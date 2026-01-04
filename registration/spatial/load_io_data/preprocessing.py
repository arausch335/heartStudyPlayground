import numpy as np
from registration.spatial.transforms.transforms import (
    Transforms,
    MatrixStep,
    SubsetStep,
    apply_T,
)


class PreprocessingPipeline:
    def __init__(self, raw_points):
        raw_points = np.asarray(raw_points)
        if raw_points.ndim != 2 or raw_points.shape[1] != 3:
            raise ValueError("raw_points must be (N,3)")
        if len(raw_points) == 0:
            raise ValueError("empty raw_points")

        self.raw_points = raw_points
        self.active_points = raw_points
        self.active_indices = np.arange(raw_points.shape[0], dtype=int)

        self.processed_points = None
        self.transforms = Transforms()

        self._finalized = False

    # -------------------------
    # Wrapper
    # -------------------------
    def preprocess(
        self,
        *,
        outlier_quantile=0.99,
        iso_radius_frac=0.02,
        min_neighbors=3,
    ):
        self.clean(
            outlier_quantile=outlier_quantile,
            iso_radius_frac=iso_radius_frac,
            min_neighbors=min_neighbors,
        )
        self.normalize()
        self.finalize()
        return self

    ### --- PREPROCESSING --- ###
    def clean(self, outlier_quantile, iso_radius_frac, min_neighbors):
        """
        Removes outliers and isolated points from the active points.
        Records a SubsetStep (preprocessing stage) with standardized metadata.
        """
        self._require_not_finalized()

        points = self.active_points
        N_prev = int(len(points))
        if N_prev == 0:
            raise ValueError("empty point cloud")

        center = points.mean(axis=0)
        pts_centered = points - center
        dists = np.linalg.norm(pts_centered, axis=1)

        # ---- outlier gate (by radius quantile) ----
        if outlier_quantile is not None and 0.0 < float(outlier_quantile) < 1.0:
            r_cut = float(np.quantile(dists, float(outlier_quantile)))
            mask_outlier = dists <= r_cut
        else:
            r_cut = None
            mask_outlier = np.ones(N_prev, dtype=bool)

        idx_outlier = np.where(mask_outlier)[0]
        pts_filtered = points[idx_outlier]

        # if everything got nuked, fall back
        if len(pts_filtered) == 0:
            idx_outlier = np.arange(N_prev, dtype=int)
            pts_filtered = points

        # ---- isolated point removal ----
        ranges = pts_filtered.max(axis=0) - pts_filtered.min(axis=0)
        max_extent = float(np.max(ranges))
        iso_radius = float(iso_radius_frac) * max_extent

        if iso_radius > 0 and len(pts_filtered) > 1:
            diff = pts_filtered[:, None, :] - pts_filtered[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            neighbor_counts = (d2 < iso_radius**2).sum(axis=1) - 1
            mask_iso = neighbor_counts >= int(min_neighbors)
        else:
            mask_iso = np.ones(len(pts_filtered), dtype=bool)

        kept_idx = idx_outlier[np.where(mask_iso)[0]]
        if kept_idx.size == 0:
            kept_idx = idx_outlier

        kept_idx = np.asarray(kept_idx, dtype=int).reshape(-1)

        # ---- validate indices against current active set ----
        if kept_idx.size > 0:
            if kept_idx.min() < 0 or kept_idx.max() >= N_prev:
                raise ValueError(
                    f"Subset indices out of range for N={N_prev}: "
                    f"[{kept_idx.min()}, {kept_idx.max()}]"
                )

        # ---- apply now (mutate active) ----
        self.active_points = self.active_points[kept_idx]
        self.active_indices = self.active_indices[kept_idx]

        # ---- record step (standardized metadata) ----
        step = SubsetStep(
            name="remove_outliers_and_isolated_points",
            stage="preprocessing",
            indices=kept_idx,
        )
        step.set_metadata(
            description="Remove radial outliers and isolated points in the point cloud",
            rule="distance_quantile, neighbor_count",
            previous_index_count=N_prev,
            params={
                "outlier_quantile": None if outlier_quantile is None else float(outlier_quantile),
                "outlier_r_cut": r_cut,
                "iso_radius_frac": float(iso_radius_frac),
                "iso_radius": float(iso_radius),
                "min_neighbors": int(min_neighbors),
                "kept_count": int(len(kept_idx)),
                "removed_count": int(N_prev - len(kept_idx)),
            },
        )
        self.transforms.add(step)

        return self

    def normalize(self):
        """
        Normalize the *current* active set by centering and scaling to unit-ish size.
        Records a MatrixStep (preprocessing stage) with standardized metadata.
        """
        self._require_not_finalized()

        pts = self.active_points
        if len(pts) == 0:
            raise ValueError("empty point cloud")

        center = pts.mean(axis=0)
        ranges = pts.max(axis=0) - pts.min(axis=0)
        scale = float(np.max(ranges))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        # x_norm = (x - center) / scale
        s = 1.0 / float(scale)
        T = np.eye(4, dtype=float)
        T[0, 0] = s
        T[1, 1] = s
        T[2, 2] = s
        T[0, 3] = -float(center[0]) * s
        T[1, 3] = -float(center[1]) * s
        T[2, 3] = -float(center[2]) * s

        # ---- apply now (mutate active) ----
        self.active_points = apply_T(T, self.active_points)

        # ---- record step (standardized metadata) ----
        step = MatrixStep(
            name="normalize(center_then_scale)",
            stage="preprocessing",
            T=T,
        )
        step.set_metadata(
            description="Center active points and scale by max extent",
            from_space="active_pre_norm",
            to_space="active_norm",
            method="center_then_uniform_scale",
            params={
                "center_world": np.asarray(center, dtype=float).tolist(),
                "scale": float(scale),
            },
            metrics={},
        )
        self.transforms.add(step)

        return self

    ### --- FINALIZE AND REPORT --- ###
    def finalize(self):
        self.processed_points = self.active_points
        self._finalized = True
        return self

    def summary(self, display=True):
        d = {
            "raw_n": int(len(self.raw_points)),
            "active_n": int(len(self.active_points)),
            "processed_n": None if self.processed_points is None else int(len(self.processed_points)),
            "n_steps": int(len(self.transforms.steps)),
        }
        if display:
            for k, v in d.items():
                print(f"{k}: {v}")
            print(self.transforms.summary())
        return d

    ### --- GUARDS --- ###
    def _require_not_finalized(self):
        if self._finalized:
            raise RuntimeError("Pipeline already finalized; create a new pipeline to add more steps.")
