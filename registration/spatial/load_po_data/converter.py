import os
from typing import Any, Dict

from registration.spatial.load_po_data.cine_builder import build_cine_from_dicom
from registration.spatial.load_po_data.exporter import (
    save_multivolume_nhdr,
    save_nifti_4d,
    save_nifti_per_frame,
    save_slicer_sequence_nrrd,
)


def convert_dicom(
    dicom_dir: str,
    results_root: str,
    name: str,
    make_4d_nifti: bool = True,
    make_3d_series: bool = True,
    make_nhdr: bool = True,
    make_seq: bool = True,
) -> Dict[str, Any]:
    """
    Build cine once and save outputs under:
      <results_root>/<name or name (k)>/{4d_nifti,3d_nifti_series,NHDR,seq}/...

    Returns dict includes:
      - header_subset: curated DICOM tags (no PHI)
      - meta: computed meta (dt, RR, voxel sizes, counts, etc.)
      - paths: output artifacts
    """
    # 1) Build cine once
    vol4d, affine, t_grid_ms, meta, header_subset = build_cine_from_dicom(dicom_dir)
    X, Y, Z, T = vol4d.shape

    # 2) Create series dir and subfolders
    series_dir = _unique_series_dir(results_root, name)
    subdirs = _make_subdirs(series_dir)

    # 3) Save into subfolders
    paths = {
        "nifti4d": None,
        "nifti3d_dir": None,
        "nifti3d_files": None,
        "nhdr": None,
        "nhdr_raw": None,
        "seq": None,
    }

    if make_4d_nifti:
        out_base = os.path.join(subdirs["nifti4d"], f"{name}_4d")
        paths["nifti4d"] = save_nifti_4d(vol4d, affine, t_grid_ms, out_base)

    if make_3d_series:
        frames_dir = os.path.join(subdirs["nifti3d"], name)
        paths["nifti3d_dir"] = frames_dir
        paths["nifti3d_files"] = save_nifti_per_frame(vol4d, affine, t_grid_ms, frames_dir, prefix=name)

    if make_nhdr:
        out_base = os.path.join(subdirs["nhdr"], f"{name}_4d")
        nhdr_path, raw_path = save_multivolume_nhdr(vol4d, affine, t_grid_ms, out_base)
        paths["nhdr"], paths["nhdr_raw"] = nhdr_path, raw_path

    if make_seq:
        out_base = os.path.join(subdirs["seq"], f"{name}_4d")
        paths["seq"] = save_slicer_sequence_nrrd(vol4d, affine, t_grid_ms, out_base)

    return {
        "series_dir": series_dir,
        "subdirs": subdirs,
        "shape": (X, Y, Z, T),
        "meta": meta,
        "header_subset": header_subset,
        "paths": paths,
    }


# utilities
def _unique_series_dir(results_root: str, name: str) -> str:
    """
    Create a unique directory under results_root with the given name.
    If 'name' exists, append ' (1)', ' (2)', ... until free.
    Returns absolute path to the created dir.
    """
    results_root = os.path.abspath(results_root)
    os.makedirs(results_root, exist_ok=True)

    base = os.path.join(results_root, name)
    candidate = base
    i = 1
    while os.path.exists(candidate):
        candidate = f"{base} ({i})"
        i += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate


def _make_subdirs(series_dir: str) -> Dict[str, str]:
    subdirs = {
        "nifti4d": os.path.join(series_dir, "4d_nifti"),
        "nifti3d": os.path.join(series_dir, "3d_nifti_series"),
        "nhdr": os.path.join(series_dir, "NHDR"),
        "seq": os.path.join(series_dir, "seq"),
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)
    return subdirs

