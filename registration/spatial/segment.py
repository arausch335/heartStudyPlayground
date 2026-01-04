from registration.spatial.segmentation.retractor import *
from registration.spatial.segmentation.epicardium import *
from registration.spatial.segmentation.heart import *
from registration.spatial.transforms.annotation import *
from registration.utilities.utilities import *
import plotly.graph_objects as go


class Segment(object):
    def __init__(self, env):
        self.env = env
        self.parent_frame = None
        self.label = None

        # indices into frame.raw_points (authoritative)
        self.subset_indices = None

        self.segmented = False
        self.metrics = {}

    def attach(self, frame):
        self.parent_frame = frame
        return self

    def get_points(self, index_space="active", coord_space="world"):
        """
        Return this segment's points in a requested index/coordinate space.

        Index spaces
        ------------
        - "raw":     subset_indices are applied directly to frame.raw_points
        - "active":  subset_indices (raw) are mapped into the frame's active set
                    using frame.active_indices, then applied to frame active points

        Coord spaces
        ------------
        Delegated to frame.get_points(...):
        - "world": active/raw points in world coordinates
        - "or":    applies registration matrices (stage="registration") if present
        """
        f = self.parent_frame
        if f is None:
            raise RuntimeError("Segment has no parent_frame")
        if getattr(f, "raw_points", None) is None:
            raise RuntimeError("Parent frame has no raw_points")
        if self.subset_indices is None:
            raise RuntimeError("Segment has no subset_indices (raw)")

        index_space = str(index_space).lower()
        coord_space = str(coord_space).lower()

        raw_idx = np.asarray(self.subset_indices, dtype=int).reshape(-1)

        # --- RAW index space: trivial ---
        if index_space == "raw":
            raw_pts = f.get_points(index_space="raw", coord_space=coord_space)
            return raw_pts[raw_idx]

        # --- ACTIVE index space: map raw subset -> active subset indices ---
        if index_space == "active":
            if getattr(f, "active_indices", None) is None:
                raise RuntimeError(
                    f"Frame {getattr(f, 'n', '?')}: active_indices not set; cannot map raw->active."
                )

            active_raw = np.asarray(f.active_indices, dtype=int).reshape(-1)

            # Build raw->active lookup (raw index -> position in active array)
            lut = {int(r): i for i, r in enumerate(active_raw)}

            # Keep only those raw indices that are still present in the active set
            active_subset_idx = np.asarray(
                [lut.get(int(r), -1) for r in raw_idx], dtype=int
            )
            active_subset_idx = active_subset_idx[active_subset_idx >= 0]

            active_pts = f.get_points(index_space="active", coord_space=coord_space)
            return active_pts[active_subset_idx]

        raise ValueError(f"Unknown index_space '{index_space}'. Expected 'raw' or 'active'.")



class Retractor(Segment):
    def __init__(self, env):
        super().__init__(env=env)
        self.label = "retractor"

        self.base_plane = None
        self.origin_world = None
        self.axes_world = None

        # world -> retractor (will be set by ALIGN step, not segmentation)
        self.T_world_to_retractor = None

    def IO_isolate(self):
        """
        Runs segmentation on ACTIVE points,
        stores subset_indices as RAW indices,
        and records a SegmentationStep on frame.transforms.
        """
        f = self.parent_frame
        if f is None:
            raise RuntimeError("Retractor has no parent_frame")
        if f.raw_points is None:
            raise RuntimeError("Frame.raw_points is None")
        if getattr(f, "transforms", None) is None:
            raise RuntimeError("Frame.transforms is None")
        if getattr(f, "active_indices", None) is None:
            raise RuntimeError("Frame.active_indices is None (needed for active->raw mapping)")

        # --- segment on ACTIVE points ---
        active_pts = f.get_points(index_space="active", coord_space="world")
        active_to_raw = np.asarray(f.active_indices, dtype=int).reshape(-1)
        previous_index_count = int(active_to_raw.size)

        params = {
            "distance_threshold": 0.05,
            "min_inliers": 1000,
            "max_planes": 1,
            "max_iterations": 1000,
            "num_planes_to_combine": 1,
        }

        result = retractor_segmentation_algorithm(active_pts, **params)

        active_idx = np.asarray(result["indices"], dtype=int).reshape(-1)
        if active_idx.size == 0:
            raise RuntimeError("Retractor segmentation produced 0 indices")

        # --- map ACTIVE indices -> RAW indices (authoritative storage) ---
        if active_idx.min() < 0 or active_idx.max() >= previous_index_count:
            raise ValueError("Segmentation indices out of range for current active set")

        raw_idx = active_to_raw[active_idx]

        # store on segment object
        self.subset_indices = raw_idx
        self.segmented = True

        # optional: store algorithm outputs (NOTE: these are in ACTIVE coordinates right now)
        self.base_plane = result.get("base_plane", None)
        self.origin_world = result.get("origin_world", None)
        self.axes_world = result.get("axes_world", None)
        self.T_world_to_retractor = result.get("T_world_to_retractor", None)
        self.metrics = result.get("metrics", {})

        # --- record segmentation annotation step on frame.transforms ---
        step = SegmentationStep(
            name="segment/retractor",
            label="retractor",
            subset_raw_indices=raw_idx,
            stage="segmentation",
        )

        step.set_metadata(
            source_space="active",
            raw_indices_count=int(raw_idx.size),
            active_indices_at_time=active_to_raw,     # raw mapping snapshot
            previous_index_count=previous_index_count,
            algorithm="ransac_plane",
            params=params,
            metrics={
                "raw_count": int(raw_idx.size),
                "active_count": int(active_idx.size),
            },
        )

        f.transforms.add(step)

        return self

    def visualize(self):
        try:
            pts = self.registered_points
            pts_type = 'Registered'
        except AttributeError:
            pts = self.original_points
            pts_type = 'Original'

        x_mesh, y_mesh, z_mesh, v_mesh = convert_point_to_meshgrid(pts)

        fig = go.Figure(data=[go.Surface(x=x_mesh, y=y_mesh, z=z_mesh, opacity=0.5)])
        fig.update_layout(title=dict(text=f'Frame {self.frame_n} Retractor - {pts_type} Points'))
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                          highlightcolor="limegreen", project_z=True))

        fig.show()


class Epicardium(Segment):
    def __init__(self, env):
        super().__init__(env=env)
        self.label = "epicardium"
        self.metrics = {}

    def IO_isolate(self, frame):
        """
        Call retractor-specific helper on this frame's points,
        then construct a Retractor segment.
        """

        result = epicardium_segmentation_algorithm(frame)

        self.subset_indices = result["indices"]
        self.metrics = result.get("metrics", {})


class Heart(Segment):
    def __init__(self):
        super().__init__()
        self.label = "heart"
        self.metrics = {}

    def PO_isolate(self, frame):
        """
        Call retractor-specific helper on this frame's points,
        then construct a Retractor segment.
        """

        result = heart_segmentation_algorithm(frame)

        self.subset_indices = result["indices"]
        self.metrics = result.get("metrics", {})

