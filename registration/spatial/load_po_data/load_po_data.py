import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import shutil

from registration.spatial.load_po_data.converter import convert_dicom
from registration.spatial.series import POSeries
from registration.spatial.load_po_data.po_frame import POFrame

PathLike = Union[str, Path]


def load_po_series(
    data_dir: PathLike,
    load_view: str,
    *,
    subject_id: Optional[str] = None,
    make_4d_nifti: bool = True,
    make_3d_series: bool = False,
    make_nhdr: bool = False,
    make_seq: bool = True,
    sequence_type: str = "cine_bSSFP",
    modality: str = "CMR",
    intent: str = "mechanics_reference",
    replace: bool = False
) -> List[POSeries]:
    """
    Convert and load PO series for one or many subjects.

    Design pattern mirrors IO:
        s = POSeries()
        s.set_metadata(...)
        s.set_paths(...)
        s.set_header_subset(...)
        s.load_header()
        s.set_timing(...)
        s.validate()

    Returns:
        list[POSeries] (one per subject)
    """
    data_dir = Path(data_dir)

    view_code_dict = {"short_axis": "SAX", "long_axis": "LAX", "sax": "SAX", "lax": "LAX"}
    view_code = view_code_dict.get(load_view, load_view)

    if not data_dir.exists():
        raise IOError(f"PO - data_dir does not exist: {data_dir}")

    subjects = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if len(subjects) == 0:
        raise IOError(f"PO - No subjects found in {data_dir}")

    if subject_id is not None:
        if subject_id not in subjects:
            raise IOError(f"PO - Subject '{subject_id}' not found in {data_dir}")
        subjects = [subject_id]

    out: List[POSeries] = []

    for sid in subjects:
        subject_dir = data_dir / sid
        dicom_view_dir = _find_view_dir(subject_dir, view_code)

        # Where we write converted outputs
        results_root = data_dir.parent / "converted_data" / sid / dicom_view_dir.name

        if os.path.isdir(results_root) and replace:
            shutil.rmtree(results_root)

        # Convert DICOM -> NIfTI/NHDR/SEQ as requested
        res = convert_dicom(
            dicom_dir=str(dicom_view_dir),
            results_root=str(results_root),
            name=load_view,
            make_4d_nifti=make_4d_nifti,
            make_3d_series=make_3d_series,
            make_nhdr=make_nhdr,
            make_seq=make_seq,
        )

        # Prefer 4D nifti if present, else 3D directory import can be handled separately
        nifti4d_path = res.get("paths", {}).get("nifti4d", None)
        series_dir = res.get("series_dir", None)

        # ---- Build POSeries (builder-style, IO-like) ----
        s = POSeries()
        s.set_metadata(
            subject_id=sid,
            series_uid=f"{sid}_{view_code}",
            modality=modality,
            sequence_type=sequence_type,
            view=view_code,
            intent=intent,
        )
        s.set_paths(
            nifti_path=nifti4d_path,
            dicom_dir=dicom_view_dir,
            series_dir=series_dir,
        )
        s.set_dicom_metadata(res.get("header_subset", {}))

        # Load canonical geometry/timing from NIfTI header (affine, zooms, dt if present)
        s.load_nifti_metadata()

        # Fill timing from converter meta as a backstop (useful if NIfTI dt is missing/mangled)
        meta = res.get("meta", {}) or {}
        rr = meta.get("RR_ms", None)
        dt = meta.get("dt_ms", None)

        # Your builder expects dt in ms already
        s.set_timing(
            frame_dt_ms=dt,
            RR_ms=rr,
            heart_rate_bpm=s.heart_rate_bpm,  # already promoted from header_subset if present
        )

        # Validate after all ingestion
        s.validate(strict=True)

        out.append(s)

    return out


def load_po_frames(
    series: POSeries,
    *,
    cache_series_data: bool = False,
    use_scene_phase_index: bool = False,
    scene_phase_map: Optional[Dict[int, int]] = None,
) -> List[POFrame]:
    """
    Create POFrame objects from a POSeries.

    - If series is 4D cine: returns one POFrame per cine timepoint.
    - If series is 3D: returns a single POFrame.

    Naming convention:
      - POFrame.phase_index = your PROJECT phase index (optional; standardized grid)
      - POFrame.cine_phase_index = the native cine index (0..T-1)
    """
    # Ensure header is loaded (and optionally data)
    series.load_nifti_metadata()
    if cache_series_data:
        series.load_data()

    nT = series.n_frames()
    frames: List[POFrame] = []

    # If not cine, make one frame
    if not series.is_cine():
        print('No cine detected')
        f = POFrame()
        f.attach(series)
        f.set_phase_indices(
            phase_index=0 if use_scene_phase_index else None,
            cine_phase_index=0,
        )
        frames.append(f)
        return frames

    for cine_t in range(nT):
        f = POFrame()
        f.attach(series)
        f.cine_phase_index = cine_t
        frames.append(f)

    return frames


def _find_view_dir(subject_dir: Path, view_code: str) -> Path:
    """
    Find the directory inside subject_dir matching view_code.
    Uses substring matching (view_code in dirname).
    """
    if not subject_dir.exists():
        raise IOError(f"PO - Subject directory does not exist: {subject_dir}")

    candidates: List[Path] = []
    for name in os.listdir(subject_dir):
        p = subject_dir / name
        if p.is_dir() and (view_code in name):
            candidates.append(p)

    if len(candidates) == 0:
        raise IOError(f"PO - View '{view_code}' not found in {subject_dir}")

    # If multiple, pick the shortest name match to reduce accidental partial matches
    candidates.sort(key=lambda x: len(x.name))
    return candidates[0]
