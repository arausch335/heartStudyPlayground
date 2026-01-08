import numpy as np

from registration.spatial.frame import Frame
from registration.spatial.segment import Retractor
from registration.spatial.utilities.utilities import (
    build_subset_indices_dict,
)
from registration.spatial.utilities.visualization import plot_surface_with_subsets


class IOFrame(Frame):
    def __init__(self, env):
        super().__init__(env=env)
        self.retractor = None
        self.epicardium = None

    def segment(self):
        self.retractor = Retractor(self.env)
        self.retractor.attach(self)
        self.retractor.IO_isolate()
        print(f'Frame {self.n} - Retractor Isolated')

        # self.epicardium = Epicardium()
        # self.epicardium.attach(self.n)

    def align_to_retractor_plane(self, stage="registered"):
        """
        Define OR by aligning WORLD -> retractor-plane coordinates.

        Uses retractor geometry computed during segmentation.
        Stores transform ONLY in self.transforms (source-of-truth).
        """
        import numpy as np
        from registration.spatial.transforms.matrix import MatrixStep

        r = self.retractor

        if r is None or not getattr(r, "segmented", False):
            raise RuntimeError(f"Frame {self.n}: retractor not segmented; cannot align.")

        if self.transforms is None:
            raise RuntimeError(f"Frame {self.n}: transforms not set; cannot record alignment.")

        # Require geometry computed at segmentation time
        if r.origin_world is None or r.axes_world is None:
            raise RuntimeError(
                f"Frame {self.n}: retractor geometry missing "
                "(origin_world / axes_world not set during segmentation)."
            )

        # WORLD -> retractor frame
        x = np.asarray(r.axes_world["x"], dtype=float)
        y = np.asarray(r.axes_world["y"], dtype=float)
        z = np.asarray(r.axes_world["z"], dtype=float)
        origin = np.asarray(r.origin_world, dtype=float)

        R = np.vstack([x, y, z])  # rows = axes (matches your convention)
        t = -R @ origin

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t

        step = MatrixStep(
            name="define_or/align_to_retractor_plane",
            T=T,
            stage=stage,
        )
        step.set_metadata(
            description="Define OR from segmented retractor plane (processed -> registered)",
            from_space="processed",
            to_space="registered",
            method="retractor_axes_from_segmentation",
            params={
                "retractor_label": r.label,
            },
            metrics={}
        )

        self.transforms.add(step)

        # Optional convenience flag only
        self.is_registered = True
        self.validate_points_consistency(strict=False)
        return self

    def visualize(self, retractor=False, epicardium=False, *, stage="registered"):
        """
        Visualize the frame surface, optionally highlighting segment subsets.

        - Uses get_points(stage)
        - Segment subset_indices are RAW indices and are mapped into stage index space.
        """
        self.validate_points_consistency(strict=False)
        pts, stage_raw_indices = self._get_stage_points_with_indices(stage)

        # --- build subset indices in the SAME index_space as `pts` ---
        def _raw_to_stage_indices(raw_idx):
            if raw_idx is None:
                return None
            raw_idx = np.asarray(raw_idx, dtype=int).reshape(-1)
            mapped = self._map_raw_to_stage_indices(raw_idx, stage_raw_indices)
            return mapped if mapped.size > 0 else np.array([], dtype=int)

        retractor_idx = None
        epicardium_idx = None

        if retractor and self.retractor is not None and getattr(self.retractor, "subset_indices", None) is not None:
            retractor_idx = _raw_to_stage_indices(self.retractor.subset_indices)

        if epicardium and self.epicardium is not None and getattr(self.epicardium, "subset_indices", None) is not None:
            epicardium_idx = _raw_to_stage_indices(self.epicardium.subset_indices)

        subset_indices_dict = build_subset_indices_dict(
            retractor=retractor_idx,
            epicardium=epicardium_idx,
        )
        subset_indices_dict = subset_indices_dict or None
        fig = plot_surface_with_subsets(
            pts,
            subset_indices=subset_indices_dict,
            title=f"Frame {self.n}  ({stage})",
        )
        fig.show()
