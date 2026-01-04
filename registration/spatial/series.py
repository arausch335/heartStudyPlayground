from pathlib import Path

import numpy as np
import nibabel as nib
import napari


class POSeries:
    """
    Preoperative imaging series (one acquisition, e.g., cine SAX bSSFP).

    Canonical geometry/timing should be derived from the NIfTI you produce.
    DICOM metadata is used for provenance & MR sequence parameters.
    """

    def __init__(self):
        # ---- Identity / provenance ----
        self.subject_id = None
        self.series_uid = None
        self.modality = None
        self.sequence_type = None
        self.view = None
        self.intent = "mechanics_reference"
        self.acquisition_datetime_iso = None

        # DICOM UIDs (useful provenance, not PHI)
        self.dicom_study_uid = None
        self.dicom_series_uid = None

        # DICOM labels
        self.series_description = None
        self.protocol_name = None

        # ---- File references ----
        self.nifti_path = None
        self.dicom_dir = None
        self.series_dir = None

        # ---- Geometry (canonical from NIfTI) ----
        self.affine = None            # 4x4 RAS
        self.spacing_mm = None        # (sx, sy, sz)
        self.shape_xyzt = None        # (X, Y, Z, T)
        self.orientation_convention = "RAS"

        # Optional DICOM geometry hints
        self.pixel_spacing_rowcol_mm = None
        self.slice_thickness_mm = None
        self.spacing_between_slices_mm = None

        # ---- Timing ----
        self.frame_dt_ms = None
        self.frame_times_ms = None
        self.RR_ms = None
        self.heart_rate_bpm = None
        self.gating = None

        # ---- MR params ----
        self.TR_ms = None
        self.TE_ms = None
        self.flip_angle_deg = None
        self.field_strength_T = None
        self.manufacturer = None
        self.model = None

        # ---- Curated metadata ----
        self.dicom_metadata = {}

        # ---- Derived assets / bookkeeping ----
        self.masks = {}
        self.preprocess_steps = []
        self.landmarks_world = {}

        # ---- Internal caches ----
        self._img_cache = None
        self._data_cache = None

    # -------------------------
    # Builder-style setters
    # -------------------------
    def set_metadata(
        self,
        *,
        subject_id,
        series_uid,
        modality,
        sequence_type,
        view,
        intent="mechanics_reference",
        acquisition_datetime_iso=None,
    ):
        self.subject_id = subject_id
        self.series_uid = series_uid
        self.modality = modality
        self.sequence_type = sequence_type
        self.view = view
        self.intent = intent
        self.acquisition_datetime_iso = acquisition_datetime_iso
        return self

    def set_paths(self, *, nifti_path=None, dicom_dir=None, series_dir=None):
        self.nifti_path = Path(nifti_path) if nifti_path is not None else None
        self.dicom_dir = Path(dicom_dir) if dicom_dir is not None else None
        self.series_dir = Path(series_dir) if series_dir is not None else None
        return self

    def set_timing(
        self,
        *,
        frame_dt_ms=None,
        frame_times_ms=None,
        gating=None,
        heart_rate_bpm=None,
        RR_ms=None,
    ):
        if frame_dt_ms is not None:
            self.frame_dt_ms = float(frame_dt_ms)
        self.frame_times_ms = frame_times_ms
        self.gating = gating
        if heart_rate_bpm is not None:
            self.heart_rate_bpm = float(heart_rate_bpm)
        if RR_ms is not None:
            self.RR_ms = float(RR_ms)
        return self

    def set_orientation_convention(self, convention):
        self.orientation_convention = convention
        return self

    def set_dicom_metadata(self, dicom_metadata):
        self.dicom_metadata = dict(dicom_metadata) if dicom_metadata is not None else {}
        dcm = self.dicom_metadata

        self.manufacturer = self.manufacturer or dcm["provenance"]["Manufacturer"]
        self.model = self.model or dcm["provenance"]["ManufacturerModelName"]
        self.field_strength_T = self.field_strength_T or float(
            dcm["provenance"]["MagneticFieldStrength"]
        )

        self.dicom_study_uid = self.dicom_study_uid or dcm["series"]["StudyInstanceUID"]
        self.dicom_series_uid = self.dicom_series_uid or dcm["series"]["SeriesInstanceUID"]

        self.series_description = self.series_description or dcm["series"]["SeriesDescription"]
        self.protocol_name = self.protocol_name or dcm["series"]["ProtocolName"]

        self.TR_ms = self.TR_ms or float(dcm["mr_params"]["RepetitionTime"])
        self.TE_ms = self.TE_ms or float(dcm["mr_params"]["EchoTime"])
        self.flip_angle_deg = self.flip_angle_deg or float(dcm["mr_params"]["FlipAngle"])

        self.heart_rate_bpm = self.heart_rate_bpm or float(dcm["timing"]["HeartRate"])
        self.RR_ms = self.RR_ms or float(dcm["timing"]["NominalInterval"])

        ps = dcm["geometry"]["PixelSpacing"]
        if self.pixel_spacing_rowcol_mm is None and len(ps) >= 2:
            self.pixel_spacing_rowcol_mm = (float(ps[0]), float(ps[1]))

        self.slice_thickness_mm = self.slice_thickness_mm or float(
            dcm["geometry"]["SliceThickness"]
        )
        self.spacing_between_slices_mm = self.spacing_between_slices_mm or float(
            dcm["geometry"]["SpacingBetweenSlices"]
        )

        return self

    # -------------------------
    # Loading / access
    # -------------------------
    def load_nifti_metadata(self):
        if self.nifti_path is None:
            raise ValueError("POSeries.nifti_path is None; cannot load header.")

        img = nib.load(str(self.nifti_path))
        self._img_cache = img

        if self.affine is None:
            self.affine = np.array(img.affine, dtype=float)

        zooms = img.header.get_zooms()
        if self.spacing_mm is None and len(zooms) >= 3:
            self.spacing_mm = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        if self.frame_dt_ms is None and len(zooms) >= 4:
            dt = float(zooms[3])
            if np.isfinite(dt) and dt > 0:
                self.frame_dt_ms = dt

        shp = tuple(int(x) for x in img.shape)
        if len(shp) == 3:
            self.shape_xyzt = (shp[0], shp[1], shp[2], 1)
        elif len(shp) == 4:
            self.shape_xyzt = (shp[0], shp[1], shp[2], shp[3])
        else:
            self.shape_xyzt = None

        return self

    def load_data(self):
        if self._img_cache is None:
            self.load_nifti_metadata()
        self._data_cache = np.asanyarray(self._img_cache.dataobj)
        return self

    def get_data(self):
        if self._data_cache is not None:
            return self._data_cache
        if self._img_cache is None:
            self.load_nifti_metadata()
        return np.asanyarray(self._img_cache.dataobj)

    # -------------------------
    # Introspection
    # -------------------------
    def shape(self):
        if self.shape_xyzt is not None:
            X, Y, Z, T = self.shape_xyzt
            return (X, Y, Z) if T == 1 else (X, Y, Z, T)
        return tuple(int(x) for x in self.get_data().shape)

    def is_cine(self):
        return len(self.shape()) == 4

    def n_frames(self):
        shp = self.shape()
        return int(shp[3]) if len(shp) == 4 else 1

    def time_ms_for_cine_index(self, cine_phase_index):
        if self.frame_times_ms is not None:
            if 0 <= cine_phase_index < len(self.frame_times_ms):
                return float(self.frame_times_ms[cine_phase_index])
            return None
        if self.frame_dt_ms is not None:
            return float(cine_phase_index) * float(self.frame_dt_ms)
        return None

    # -------------------------
    # QA / summary / viz
    # -------------------------
    def validate(self, *, strict=True):
        issues = []

        if self.subject_id is None:
            issues.append("subject_id is None")
        if self.series_uid is None:
            issues.append("series_uid is None")

        if self.nifti_path is None:
            issues.append("nifti_path is None")
        elif not self.nifti_path.exists():
            issues.append(f"nifti_path does not exist: {self.nifti_path}")

        if self.affine is not None:
            A = np.asarray(self.affine)
            if A.shape != (4, 4):
                issues.append(f"affine shape is {A.shape}, expected (4,4)")
            if not np.isfinite(A).all():
                issues.append("affine contains non-finite values")

        if self.spacing_mm is not None:
            if min(self.spacing_mm) <= 0:
                issues.append(f"spacing_mm has non-positive values: {self.spacing_mm}")

        if self.is_cine():
            if self.frame_dt_ms is None:
                issues.append("cine series but frame_dt_ms is None")
            if self.RR_ms is None:
                issues.append("cine series but RR_ms is None")

        if strict and issues:
            raise ValueError("POSeries validation failed:\n- " + "\n- ".join(issues))

        return issues

    def summary(self, display=True):
        if self._img_cache is None and self.nifti_path is not None:
            self.load_nifti_metadata()

        shp = self.shape()

        summary_dict = {
            "subject_id": self.subject_id,
            "series_uid": self.series_uid,
            "modality": self.modality,
            "sequence_type": self.sequence_type,
            "view": self.view,
            "intent": self.intent,
            "shape": shp,
            "spacing_mm": self.spacing_mm,
            "n_frames": self.n_frames(),
            "TR_ms": self.TR_ms,
            "TE_ms": self.TE_ms,
            "flip_angle_deg": self.flip_angle_deg,
            "heart_rate_bpm": self.heart_rate_bpm,
        }

        if display:
            for k, v in summary_dict.items():
                print(f"{k}: {v}")

        return summary_dict

    def visualize(self, colormap="twilight", gamma=0.35):
        if not self.nifti_path:
            raise ValueError("nifti_path must be set")

        img = nib.load(self.nifti_path)
        data = np.asanyarray(img.dataobj)
        data_tzyx = np.transpose(data, (3, 2, 1, 0))

        viewer = napari.Viewer()
        viewer.add_image(
            data_tzyx,
            name="cine",
            colormap=colormap,
            gamma=gamma)
        napari.run()
