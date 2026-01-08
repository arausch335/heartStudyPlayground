import numpy as np


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
