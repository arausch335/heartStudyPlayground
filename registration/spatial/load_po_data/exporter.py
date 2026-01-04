import os
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import nrrd


# ======================
# Export: 4D NIfTI
# ======================
def save_nifti_4d(vol4d_XYZT: np.ndarray, affine_RAS_4x4: np.ndarray, t_grid_ms: np.ndarray, out_path_no_ext: str) -> str:
    """
    Saves a single 4D NIfTI (.nii.gz) with time intent/units.
    """
    X, Y, Z, T = vol4d_XYZT.shape
    out_path = out_path_no_ext if out_path_no_ext.lower().endswith((".nii", ".nii.gz")) else out_path_no_ext + ".nii.gz"

    vx, vy, vz = voxel_sizes_from_affine(affine_RAS_4x4)
    dt_ms = float((t_grid_ms[1] - t_grid_ms[0]) if T > 1 else 0.0)

    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units("mm", "msec")
    hdr["dim"][0] = 4
    hdr["dim"][1:5] = (X, Y, Z, T)
    hdr["pixdim"][1] = vx
    hdr["pixdim"][2] = vy
    hdr["pixdim"][3] = vz
    hdr["pixdim"][4] = dt_ms
    hdr.set_intent("NIFTI_INTENT_TIME_SERIES")

    img = nib.Nifti1Image(vol4d_XYZT.astype(np.float32, copy=False), affine_RAS_4x4, header=hdr)
    nib.save(img, out_path)
    return os.path.abspath(out_path)


# ======================
# Export: 3D NIfTIs / frame
# ======================
def save_nifti_per_frame(
    vol4d_XYZT: np.ndarray,
    affine_RAS_4x4: np.ndarray,
    t_grid_ms: np.ndarray,
    out_dir: str,
    prefix: str = "cine",
) -> List[str]:
    """
    Saves one 3D NIfTI per time frame into out_dir and writes index.csv
    Returns list of file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    vx, vy, vz = voxel_sizes_from_affine(affine_RAS_4x4)
    X, Y, Z, T = vol4d_XYZT.shape

    paths: List[str] = []
    index_lines = ["frame,time_ms,filename"]

    for k in range(T):
        vol3d = vol4d_XYZT[..., k]
        t_ms = float(t_grid_ms[k])
        t_tag = f"{int(round(t_ms)):04d}ms"
        fname = f"{prefix}_{t_tag}.nii.gz"
        fpath = os.path.join(out_dir, fname)

        hdr3 = nib.Nifti1Header()
        hdr3.set_xyzt_units("mm", "unknown")
        hdr3["dim"][0] = 3
        hdr3["dim"][1:4] = (X, Y, Z)
        hdr3["pixdim"][1] = vx
        hdr3["pixdim"][2] = vy
        hdr3["pixdim"][3] = vz

        nib.save(nib.Nifti1Image(vol3d.astype(np.float32, copy=False), affine_RAS_4x4, header=hdr3), fpath)
        index_lines.append(f"{k},{t_ms:.3f},{fname}")
        paths.append(fpath)

    with open(os.path.join(out_dir, "index.csv"), "w") as f:
        f.write("\n".join(index_lines))

    return paths


# ======================
# Export: NHDR (detached)
# ======================
def save_multivolume_nhdr(
    vol4d_XYZT: np.ndarray,
    affine_RAS_4x4: np.ndarray,
    t_grid_ms: np.ndarray,
    out_path_no_ext: str,
) -> Tuple[str, str]:
    """
    Saves a detached NHDR multi-volume: <name>.nhdr + <name>.raw.gz
    """
    X, Y, Z, T = vol4d_XYZT.shape
    out_nhdr = os.path.abspath(out_path_no_ext if out_path_no_ext.lower().endswith(".nhdr") else out_path_no_ext + ".nhdr")
    out_dir = os.path.dirname(out_nhdr)
    base = os.path.splitext(out_nhdr)[0]
    out_raw = base + ".raw.gz"
    os.makedirs(out_dir, exist_ok=True)

    dX = list(map(float, affine_RAS_4x4[:3, 0]))
    dY = list(map(float, affine_RAS_4x4[:3, 1]))
    dZ = list(map(float, affine_RAS_4x4[:3, 2]))
    origin = tuple(map(float, affine_RAS_4x4[:3, 3]))

    dt_s = float((t_grid_ms[1] - t_grid_ms[0]) / 1000.0) if T > 1 else 0.0

    header = {
        "type": "float",
        "dimension": 4,
        "sizes": (X, Y, Z, T),
        "space": "right-anterior-superior",
        "space directions": [dX, dY, dZ, [0.0, 0.0, 0.0]],
        "space origin": origin,
        "kinds": ["domain", "domain", "domain", "list"],
        "endian": "little",
        "encoding": "gzip",
        # 3 entries for spatial units (space dimension = 3)
        "space units": ["mm", "mm", "mm"],
        "data file": os.path.basename(out_raw),
        # time axis meta
        "axis 3 label": "time",
        "axis 3 spacing": dt_s,
        "time step": dt_s,
        "frame times": " ".join(f"{t / 1000.0:.6f}" for t in t_grid_ms),
    }

    nrrd.write(out_nhdr, vol4d_XYZT, header, compression_level=1)
    return out_nhdr, out_raw


# ===========================
# Export: Slicer Sequence NRRD
# ===========================
def save_slicer_sequence_nrrd(
    vol4d_XYZT: np.ndarray,
    affine_RAS_4x4: np.ndarray,
    t_grid_ms: np.ndarray,
    out_path_seq: str,
) -> str:
    """
    Saves a single-file Slicer Sequence: <name>.seq.nrrd
    """
    X, Y, Z, T = vol4d_XYZT.shape
    t_sec = np.asarray(t_grid_ms, dtype=np.float64) / 1000.0
    assert t_sec.shape[0] == T

    # Slicer sequence layout is (list, X, Y, Z) => (T, X, Y, Z)
    data_TXYZ = np.transpose(vol4d_XYZT, (3, 0, 1, 2)).astype("<f4", copy=False)

    dX = list(map(float, affine_RAS_4x4[:3, 0]))
    dY = list(map(float, affine_RAS_4x4[:3, 1]))
    dZ = list(map(float, affine_RAS_4x4[:3, 2]))
    origin = tuple(map(float, affine_RAS_4x4[:3, 3]))

    index_values_str = " ".join(f"{v:.6f}" for v in t_sec)

    header = {
        "type": "float",
        "dimension": 4,
        "sizes": (T, X, Y, Z),
        "kinds": ["list", "domain", "domain", "domain"],
        "space": "right-anterior-superior",
        "space directions": ([0.0, 0.0, 0.0], dX, dY, dZ),
        "space origin": origin,
        "endian": "little",
        "encoding": "gzip",
        "space units": ["mm", "mm", "mm"],
        # Sequence metadata (no colons in keys)
        "axis 0 label": "time",
        "axis 0 index type": "numeric",
        "axis 0 index unit": "s",
        "axis 0 index values": index_values_str,
    }

    root, ext = os.path.splitext(out_path_seq)
    out_seq = root + ".seq.nrrd" if not ext.lower().endswith(".seq.nrrd") else out_path_seq
    nrrd.write(out_seq, data_TXYZ, header, compression_level=1)
    return os.path.abspath(out_seq)


def voxel_sizes_from_affine(affine_ras: np.ndarray) -> Tuple[float, float, float]:
    vx = float(np.linalg.norm(affine_ras[:3, 0]))
    vy = float(np.linalg.norm(affine_ras[:3, 1]))
    vz = float(np.linalg.norm(affine_ras[:3, 2]))
    return vx, vy, vz
