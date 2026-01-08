import numpy as np
import plotly.graph_objects as go

from registration.spatial.segment import Retractor
from registration.spatial.transforms.transforms import MatrixStep
from registration.spatial.utils.transforms_utils import apply_T
from registration.utilities.utilities import (
    build_subset_indices_dict,
    convert_point_to_meshgrid,
    indices_to_subset_ids,
)


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

        fig = go.Figure()

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
        subset_ids, subset_names = indices_to_subset_ids(len(pts), subset_indices_dict)

        x_mesh, y_mesh, z_mesh, v_mesh = convert_point_to_meshgrid(pts, values=subset_ids)

        fig.add_trace(
            go.Surface(
                x=x_mesh, y=y_mesh, z=z_mesh,
                opacity=0.5,
            )
        )
        fig.update_traces(
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )

        if sum([retractor, epicardium]) > 0:
            for sid, name in subset_names.items():
                mask = (v_mesh == sid)
                z_masked = np.where(mask, z_mesh, np.nan)

                fig.add_trace(
                    go.Surface(
                        x=x_mesh,
                        y=y_mesh,
                        z=z_masked,
                        opacity=0.7,
                        showscale=False,
                        colorscale="gray",
                        name=name,
                    )
                )

        fig.update_layout(
            title=f"Frame {self.n}  ({stage})"
        )
        fig.show()


class POFrame:
    """
    PO "frame" view into a POSeries.

    Design goal: mirror your IOFrame usage:
        f = POFrame()
        f.attach(series)
        f.set_phase_indices(phase_index=..., cine_phase_index=...)
    """

    def __init__(self):
        self.parent_series = None
        self.phase_index = None
        self.cine_phase_index = None
        self.label = None

    # -----------------------------
    # Setup
    # -----------------------------

    def attach(self, series):
        self.parent_series = series
        return self

    def set_phase_indices(self, *, phase_index=None, cine_phase_index):
        self.phase_index = phase_index
        self.cine_phase_index = int(cine_phase_index)
        return self

    def set_label(self, label):
        self.label = label
        return self

    # -----------------------------
    # Core accessors
    # -----------------------------

    def id(self):
        if self.parent_series is None:
            raise ValueError("POFrame.parent is None. Call attach(series) first.")
        if self.cine_phase_index is None:
            raise ValueError(
                "POFrame.cine_phase_index is None. Call set_phase_indices(...) first."
            )

        return f"{self.parent_series.series_uid}:p{self.phase_index}:t{self.cine_phase_index}"

    def volume3d(self):
        if self.parent_series is None:
            raise ValueError("POFrame.parent is None. Call attach(series) first.")
        if self.cine_phase_index is None:
            raise ValueError(
                "POFrame.cine_phase_index is None. Call set_phase_indices(...) first."
            )

        data = self.parent_series.get_data()
        t = self.cine_phase_index

        if data.ndim == 3:
            return data
        if data.ndim == 4:
            return data[..., t]

        raise ValueError(
            f"Unexpected POSeries data ndim={data.ndim}; expected 3 or 4."
        )

    def affine(self):
        if self.parent_series is None:
            raise ValueError("POFrame.parent is None. Call attach(series) first.")

        if self.parent_series.affine is None:
            self.parent_series.load_nifti_metadata()

        if self.parent_series.affine is None:
            raise ValueError("Affine could not be loaded from POSeries header.")

        return self.parent_series.affine

    def spacing_mm(self):
        if self.parent_series is None:
            raise ValueError("POFrame.parent is None. Call attach(series) first.")

        if self.parent_series.spacing_mm is None:
            try:
                self.parent_series.load_nifti_metadata()
            except Exception:
                return None

        return self.parent_series.spacing_mm

    def time_ms(self):
        if self.parent_series is None:
            raise ValueError("POFrame.parent is None. Call attach(series) first.")
        if self.cine_phase_index is None:
            raise ValueError(
                "POFrame.cine_phase_index is None. Call set_phase_indices(...) first."
            )

        return self.parent_series.time_ms_for_cine_index(self.cine_phase_index)

    ### --- SUMMARY --- ###
    def summary(self, display=True):
        if self.parent_series is None:
            raise ValueError("POFrame.parent is None. Call attach(series) first.")
        if self.cine_phase_index is None:
            raise ValueError(
                "POFrame.cine_phase_index is None. Call set_phase_indices(...) first."
            )

        vol = self.volume3d()

        summary_dict = {
            "id": self.id(),
            "subject_id": self.parent_series.subject_id,
            "series_uid": self.parent_series.series_uid,
            "view": self.parent_series.view,
            "sequence_type": self.parent_series.sequence_type,
            "phase_index": self.phase_index,
            "cine_phase_index": self.cine_phase_index,
            "shape_3d": tuple(int(x) for x in vol.shape),
            "time_ms": self.time_ms(),
        }

        if display:
            for k, v in summary_dict.items():
                print(f"{k}: {v}")

        return summary_dict

    def visualize(self, colormap="twilight", gamma=0.35):
        vol = self.volume3d()
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

        # Convert (X, Y, Z) -> (Z, Y, X)
        vol_zyx = np.transpose(vol, (2, 1, 0))

        viewer = napari.Viewer()
        viewer.add_image(
            vol_zyx,
            name=self.id(),
            colormap=colormap,
            gamma=gamma
        )
        napari.run()
