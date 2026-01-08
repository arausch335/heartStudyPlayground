from registration.spatial.segmentation.retractor import *
from registration.spatial.segmentation.epicardium import *
from registration.spatial.segmentation.heart import *
from registration.spatial.transforms.annotation import *
from registration.spatial.utilities.utilities import *
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

    def get_points(self, stage: str = "processed"):
        """
        Return this segment's points for a requested stage.

        Stages
        ------
        - "raw"
        - "processed"
        - "registered"
        """
        f = self.parent_frame
        if f is None:
            raise RuntimeError("Segment has no parent_frame")
        if getattr(f, "raw_points", None) is None:
            raise RuntimeError("Parent frame has no raw_points")
        if self.subset_indices is None:
            raise RuntimeError("Segment has no subset_indices (raw)")

        raw_idx = np.asarray(self.subset_indices, dtype=int).reshape(-1)
        stage_points, stage_raw_indices = f._get_stage_points_with_indices(stage)
        stage_idx = f._map_raw_to_stage_indices(raw_idx, stage_raw_indices)
        return stage_points[stage_idx]


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
        active_pts = f.get_active_points(stage="processed")
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
            source_space="processed",
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

    def visualize(self, stage="registered"):
        pts = self.get_points(stage=stage)
        x_mesh, y_mesh, z_mesh, v_mesh = convert_point_to_meshgrid(pts)

        fig = go.Figure(data=[go.Surface(x=x_mesh, y=y_mesh, z=z_mesh, opacity=0.5)])
        fig.update_layout(title=dict(text=f"Retractor - {stage} points"))
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
