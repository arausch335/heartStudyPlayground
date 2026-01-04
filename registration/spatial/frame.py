from registration.spatial.segment import Retractor
from registration.spatial.transforms.transforms import MatrixStep
from registration.utilities.utilities import *
import numpy as np
import plotly.graph_objects as go
import napari


class Frame(object):
    def __init__(self, env):
        self.env = env

        # points
        self.raw_points = None
        self.points = None
        self.active_indices = None

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

    def get_points(self, index_space="active", coord_space="world", *, require_registered=True):
        """
        Unified point accessor.

        Axes
        ----
        index_space:
          - "raw"    : indices into raw_points (no subsetting)
          - "active" : current active subset (uses active_indices)

        coord_space:
          - "world"  : scanner/world coordinates
          - "or"     : OR coordinates (requires registration stage)
          - aliases:
              "registered" -> ("active","or")
              "or"         -> ("active","or")

        Policy / Assumptions
        --------------------
        - Registration stage matrices represent WORLD -> OR.
        - self.raw_points are WORLD coordinates.
        - self.active_indices maps active index -> raw index.
        - self.points may be *preprocessed coordinates* (e.g., normalized), so it is NOT used
          for world/or calculations unless you explicitly ask for it.

        Parameters
        ----------
        require_registered : bool
            If coord_space="or", enforce that registration exists and/or is_registered is True.

        Returns
        -------
        (N,3) np.ndarray
        """
        # -----------------------------
        # Normalize inputs / aliases
        # -----------------------------
        if coord_space is None:
            coord_space = "world"

        index_space = str(index_space).lower()
        coord_space = str(coord_space).lower()

        # convenience aliases
        if coord_space in ("registered", "reg", "or_registered"):
            index_space = "active"
            coord_space = "or"

        if index_space in ("registered", "reg"):
            index_space = "active"
            coord_space = "or"

        # -----------------------------
        # Require core data
        # -----------------------------
        if self.raw_points is None:
            raise RuntimeError(f"Frame {self.n}: raw_points not set.")

        raw = np.asarray(self.raw_points)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise RuntimeError(f"Frame {self.n}: raw_points must be (N,3).")

        # -----------------------------
        # Build base WORLD points for requested index_space
        # -----------------------------
        if index_space == "raw":
            pts_world = raw

        elif index_space == "active":
            if self.active_indices is None:
                raise RuntimeError(f"Frame {self.n}: active_indices not set (cannot get active points).")

            active_idx = np.asarray(self.active_indices, dtype=int).reshape(-1)
            if active_idx.size > 0:
                if active_idx.min() < 0 or active_idx.max() >= len(raw):
                    raise RuntimeError(
                        f"Frame {self.n}: active_indices out of range for raw_points (N={len(raw)})."
                    )

            # IMPORTANT: active/world is defined as "raw subset in world coords"
            pts_world = raw[active_idx]

        else:
            raise ValueError(
                f"Unknown index_space '{index_space}'. Expected 'raw' or 'active'."
            )

        # -----------------------------
        # Coordinate transform
        # -----------------------------
        if coord_space == "world":
            return pts_world

        if coord_space == "or":
            if require_registered and not getattr(self, "is_registered", False):
                raise RuntimeError(f"Frame {self.n} is not registered (is_registered=False).")

            if self.transforms is None:
                raise RuntimeError(f"Frame {self.n}: transforms not set (cannot compute OR points).")

            # get registration collapse only
            if not hasattr(self.transforms, "collapse"):
                raise RuntimeError(f"Frame {self.n}: transforms has no collapse() method.")

            T_or_from_world = self.transforms.collapse(stages=["registration"])["collapsed_T"]
            if T_or_from_world is None:
                raise RuntimeError(f"Frame {self.n}: no registration transform found (stage='registration').")

            return apply_T(T_or_from_world, pts_world)

        # Optional: if you want access to the stored self.points regardless of coordinate meaning
        if coord_space in ("active_coords", "stored"):
            if index_space != "active":
                raise ValueError("coord_space='active_coords' only makes sense with index_space='active'.")
            if self.points is None:
                raise RuntimeError(f"Frame {self.n}: points not set.")
            return np.asarray(self.points)

        raise ValueError(
            f"Unknown coord_space '{coord_space}'. Expected 'world', 'or', or 'active_coords'."
        )

    def validate_transforms(self, atol=1e-6, rtol=1e-6, max_report=5, verbose=True):
        collapsed_transform, collapsed_indices = self.transforms.collapsed
        collapsed_indices = np.asarray(collapsed_indices).astype(int).reshape(-1)

        raw = np.asarray(self.raw_points)
        pts = np.asarray(self.points)
        active_idx = np.asarray(self.active_indices).astype(int).reshape(-1)

        indices_match = np.array_equal(active_idx, collapsed_indices)

        expected = apply_T(collapsed_transform, raw[collapsed_indices]) if collapsed_indices.size > 0 else raw[collapsed_indices]

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

    def align_to_retractor_plane(self, stage="registration"):
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
            description="Define OR from segmented retractor plane",
            from_space="world",
            to_space="or",
            method="retractor_axes_from_segmentation",
            params={
                "retractor_label": r.label,
            },
            metrics={}
        )

        self.transforms.add(step)

        # Optional convenience flag only
        self.is_registered = True
        return self

    def visualize(self, retractor=False, epicardium=False, *, index_space="active", coord_space="or"):
        """
        Visualize the frame surface, optionally highlighting segment subsets.

        - Uses your new get_points(index_space, coord_space)
        - Segment subset_indices are RAW indices, so we must map RAW->ACTIVE index space
          for coloring when index_space="active".
        """
        pts = self.get_points(index_space=index_space, coord_space=coord_space)

        fig = go.Figure()

        # --- build subset indices in the SAME index_space as `pts` ---
        def _raw_to_active_indices(raw_idx):
            """Map RAW indices -> indices into the current active set."""
            if raw_idx is None:
                return None
            if self.active_indices is None:
                raise RuntimeError(f"Frame {self.n}: active_indices not set; cannot map raw->active.")
            raw_idx = np.asarray(raw_idx, dtype=int).reshape(-1)
            active_raw = np.asarray(self.active_indices, dtype=int).reshape(-1)

            # raw->active lookup
            lut = {int(r): i for i, r in enumerate(active_raw)}
            mapped = [lut.get(int(r), -1) for r in raw_idx]
            mapped = np.asarray([m for m in mapped if m >= 0], dtype=int)
            return mapped if mapped.size > 0 else np.array([], dtype=int)

        retractor_idx = None
        epicardium_idx = None

        if retractor and self.retractor is not None and getattr(self.retractor, "subset_indices", None) is not None:
            if index_space == "raw":
                retractor_idx = np.asarray(self.retractor.subset_indices, dtype=int).reshape(-1)
            else:
                retractor_idx = _raw_to_active_indices(self.retractor.subset_indices)

        if epicardium and self.epicardium is not None and getattr(self.epicardium, "subset_indices", None) is not None:
            if index_space == "raw":
                epicardium_idx = np.asarray(self.epicardium.subset_indices, dtype=int).reshape(-1)
            else:
                epicardium_idx = _raw_to_active_indices(self.epicardium.subset_indices)

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
            title=f"Frame {self.n}  ({index_space}/{coord_space})"
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


