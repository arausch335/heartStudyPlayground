import numpy as np

from registration.spatial.utilities.transforms_utils import apply_T


class Frame(object):
    def __init__(self, env):
        self.env = env

        # points
        self.raw_points = None
        self.points = None
        self.points_stage = None
        self.active_indices = None
        self.index_maps = {}

        # metadata
        self.n = None
        self.file_path = None

        # spatial elements
        self.preprocessing_pipeline = None
        self.transforms = None
        self.global_transform = None

        self.is_registered = False
        self.quality_metrics = {}

        # temporal elements
        self.timestamp = None
        self.cycle_n = None
        self.phase_index = None

    def set_raw_points(self, points):
        self.raw_points = points

    def set_metadata(self, n, file_path):
        self.n = n
        self.file_path = file_path

    def get_points(self, stage: str = "registered") -> np.ndarray:
        """
        Unified point accessor using strict stage semantics.

        Stages
        ------
        - "raw"        : raw points (immutable)
        - "processed"  : raw -> processed transforms (subset + matrix steps)
        - "registered" : raw -> processed -> registered transforms
        """
        stage = str(stage).lower()
        if stage not in ("raw", "processed", "registered"):
            raise ValueError(f"Unknown stage '{stage}'. Expected raw/processed/registered.")

        raw = self._require_raw_points()
        if stage == "raw":
            return raw

        indices, T = self._get_stage_indices_and_T(stage)
        pts = raw[indices]
        return apply_T(pts, T)

    def get_active_points(self, stage: str = "processed") -> np.ndarray:
        if self.active_indices is None:
            raise RuntimeError(f"Frame {self.n}: active_indices not set.")
        active_raw = np.asarray(self.active_indices, dtype=int).reshape(-1)
        if active_raw.size == 0:
            return np.empty((0, 3), dtype=float)

        if stage == "raw":
            raw = self._require_raw_points()
            return raw[active_raw]

        stage_points, stage_raw_indices = self._get_stage_points_with_indices(stage)
        active_stage_idx = self._map_raw_to_stage_indices(active_raw, stage_raw_indices)
        return stage_points[active_stage_idx]

    def _require_raw_points(self) -> np.ndarray:
        if self.raw_points is None:
            raise RuntimeError(f"Frame {self.n}: raw_points not set.")
        raw = np.asarray(self.raw_points)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise RuntimeError(f"Frame {self.n}: raw_points must be (N,3).")
        return raw

    def _get_stage_indices_and_T(self, stage: str):
        raw = self._require_raw_points()
        if self.transforms is None:
            raise RuntimeError(f"Frame {self.n}: transforms not set.")

        stages = ["processed"] if stage == "processed" else ["processed", "registered"]
        collapsed = self.transforms.collapse(
            stages=stages,
            n_raw_points=len(raw),
        )
        indices = np.asarray(collapsed["collapsed_indices"], dtype=int).reshape(-1)
        T = np.asarray(collapsed["collapsed_T"], dtype=float)

        if stage == "registered":
            has_registration = any(
                getattr(step, "kind", None) == "matrix" and getattr(step, "stage", None) == "registered"
                for step in self.transforms.steps
            )
            if not has_registration:
                raise RuntimeError(f"Frame {self.n}: no registration transforms available.")

        return indices, T

    def _get_stage_points_with_indices(self, stage: str):
        raw = self._require_raw_points()
        indices, T = self._get_stage_indices_and_T(stage)
        pts = apply_T(raw[indices], T)
        return pts, indices

    @staticmethod
    def _map_raw_to_stage_indices(raw_indices, stage_raw_indices):
        stage_raw_indices = np.asarray(stage_raw_indices, dtype=int).reshape(-1)
        lut = {int(raw_idx): i for i, raw_idx in enumerate(stage_raw_indices)}
        mapped = np.asarray([lut.get(int(r), -1) for r in raw_indices], dtype=int)
        mapped = mapped[mapped >= 0]
        return mapped

    def map_raw_indices_to_stage(self, raw_indices, stage: str):
        _, stage_raw_indices = self._get_stage_points_with_indices(stage)
        return self._map_raw_to_stage_indices(raw_indices, stage_raw_indices)

    def validate_points_consistency(self, strict: bool = True, atol=1e-6, rtol=1e-6):
        raw = self._require_raw_points()
        raw_out = self.get_points("raw")
        if not np.allclose(raw, raw_out, atol=atol, rtol=rtol):
            raise RuntimeError(f"Frame {self.n}: raw points mismatch in get_points('raw').")

        if self.points is not None:
            if self.points_stage is None:
                raise RuntimeError(f"Frame {self.n}: points_stage must be set when points cache exists.")
            expected = self.get_points(self.points_stage)
            if not np.allclose(self.points, expected, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"Frame {self.n}: cached points mismatch for stage '{self.points_stage}'."
                )

        stages = ["raw", "processed"]
        if any(
            getattr(step, "kind", None) == "matrix" and getattr(step, "stage", None) == "registered"
            for step in (self.transforms.steps if self.transforms is not None else [])
        ):
            stages.append("registered")

        for stage in stages:
            pts_a = self.get_points(stage)
            pts_b = self.get_points(stage)
            if not np.allclose(pts_a, pts_b, atol=atol, rtol=rtol):
                raise RuntimeError(f"Frame {self.n}: non-deterministic get_points('{stage}').")

        if strict and self.active_indices is not None:
            active_raw = np.asarray(self.active_indices, dtype=int).reshape(-1)
            if active_raw.size > 0:
                if active_raw.min() < 0 or active_raw.max() >= len(raw):
                    raise RuntimeError(
                        f"Frame {self.n}: active_indices out of range for raw_points (N={len(raw)})."
                    )
            mapped = self.index_maps.get(("raw", "processed"))
            if mapped is not None:
                mapped = np.asarray(mapped, dtype=int).reshape(-1)
                if not np.array_equal(mapped, active_raw):
                    raise RuntimeError(
                        f"Frame {self.n}: index_maps('raw','processed') does not match active_indices."
                    )

        return True

    def validate_transforms(self, atol=1e-6, rtol=1e-6, max_report=5, verbose=True):
        raw = self._require_raw_points()
        collapsed = self.transforms.collapse(
            stages=["processed"],
            n_raw_points=len(raw),
        )
        collapsed_indices = np.asarray(collapsed["collapsed_indices"]).astype(int).reshape(-1)
        collapsed_transform = collapsed["collapsed_T"]

        if self.points is not None and self.points_stage == "processed":
            pts = np.asarray(self.points)
        else:
            pts = self.get_points("processed")

        active_idx = np.asarray(self.active_indices).astype(int).reshape(-1)
        indices_match = np.array_equal(active_idx, collapsed_indices)

        expected = apply_T(raw[collapsed_indices], collapsed_transform) if collapsed_indices.size > 0 else raw[collapsed_indices]

        shape_match = expected.shape == pts.shape
        if shape_match and expected.size > 0:
            diff = pts - expected
            abs_err = np.linalg.norm(diff, axis=1)
            max_err = float(abs_err.max())
            mean_err = float(abs_err.mean())
            ok_points = bool(np.allclose(pts, expected, atol=atol, rtol=rtol))
        else:
            max_err = None
            mean_err = None
            ok_points = False if not shape_match else True

        ok = bool(indices_match and ok_points and shape_match)

        report = {
            "ok": ok,
            "frame_n": self.n,
            "indices_match": bool(indices_match),
            "shape_match": bool(shape_match),
            "ok_points": bool(ok_points),
            "n_raw": int(len(raw)),
            "n_active": int(len(pts)),
            "max_point_error": max_err,
            "mean_point_error": mean_err,
        }

        if verbose:
            if ok:
                print(f"[Frame {self.n}] transform validation: OK")
            else:
                print(f"[Frame {self.n}] transform validation: FAIL")
                print(f"  indices_match: {indices_match}")
                print(f"  shape_match:   {shape_match}")
                print(f"  ok_points:     {ok_points}")
                if max_err is not None:
                    print(f"  max_err:       {max_err:.6g}")
                    print(f"  mean_err:      {mean_err:.6g}")

                # show a few worst offenders
                if shape_match and expected.size > 0:
                    worst = np.argsort(abs_err)[::-1][:max_report]
                    print("  worst points (idx_in_active, raw_idx, abs_err):")
                    for i in worst:
                        raw_i = int(collapsed_indices[i])
                        print(f"    {int(i):>5}  {raw_i:>7}  {float(abs_err[i]):.6g}")

        return report

