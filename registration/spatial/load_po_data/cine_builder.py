import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom

pydicom.config.enforce_valid_values = False  # tolerate UID quirks


# -------------------------
# Cine builder
# -------------------------
def build_cine_from_dicom(dicom_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Build cine from a DICOM folder.

    Returns:
      vol4d_XYZT: (X,Y,Z,T) float32
      affine_RAS: (4,4)
      t_grid_ms:  (T,)
      meta: dict (computed + light description)
      header_subset: dict (curated DICOM tags; no PHI)
    """
    hdrs = _load_headers(dicom_dir)
    if not hdrs:
        raise RuntimeError("No DICOMs with sufficient geometry found.")

    # choose the largest series under this dir
    series_map: Dict[Tuple[str, str], List[Tuple[str, Any]]] = defaultdict(list)
    for fp, ds in hdrs:
        key = (str(_safe_get(ds, "SeriesDescription", "UNKNOWN")),
               str(_safe_get(ds, "SeriesNumber", "NA")))
        series_map[key].append((fp, ds))

    (ser_desc, ser_num), items = max(series_map.items(), key=lambda kv: len(kv[1]))

    # group frames per slice by geometry
    slice_groups: Dict[Tuple[Any, ...], List[Tuple[float, int, str, Any]]] = defaultdict(list)
    for fp, ds in items:
        slice_groups[_group_key(ds)].append((_time_key(ds), int(_safe_get(ds, "InstanceNumber", 0) or 0), fp, ds))

    # build per-slice stacks sorted by time
    slices: List[Tuple[List[float], np.ndarray, Any]] = []
    for frames in slice_groups.values():
        frames.sort(key=lambda t: (t[0], t[1]))
        times = [t for t, _, _, _ in frames]
        fps = [fp for _, _, fp, _ in frames]
        imgs = np.stack([_load_pixel(fp) for fp in fps], axis=0)  # [T,H,W]
        slices.append((times, imgs, frames[0][3]))

    if not slices:
        raise RuntimeError("No pixel data found.")

    # order slices base→apex and compute dz
    projs = np.array([_slice_sort_val(ds)[0] for _, _, ds in slices], float)
    order = np.argsort(projs)
    dzs = np.diff(projs[order])
    dz = float(np.median(np.abs(dzs))) if len(dzs) else float(_safe_get(slices[0][2], "SliceThickness", 0.0) or 0.0)

    # uniform time grid
    sample_ds = slices[0][2]
    RR = None
    ni = _safe_get(sample_ds, "NominalInterval", None)
    if ni is not None:
        try:
            RR = float(ni)
        except Exception:
            RR = None
    if RR is None:
        tmax = max(max(times) for times, _, _ in slices)
        RR = float(np.ceil(tmax / 10.0) * 10.0)

    T = int(slices[0][1].shape[0])
    t_grid_ms = np.linspace(0.0, RR, T, endpoint=False).astype(np.float64)

    # interpolate each slice to the common grid and stack -> (X,Y,Z,T)
    interped = [_interp_time(slices[i][0], slices[i][1], t_grid_ms, RR) for i in order]  # list of [T,H,W]
    vol_ZTHW = np.stack(interped, axis=0)            # [Z,T,H,W]
    vol_XYZT = np.transpose(vol_ZTHW, (3, 2, 0, 1))  # (X,Y,Z,T)
    vol_XYZT = np.nan_to_num(vol_XYZT, copy=False).astype("<f4", copy=False)

    # geometry / affine
    ds_first = slices[order[0]][2]
    affine = affine_from_dicom_ras(ds_first, dz)

    # curated DICOM header subset
    header_subset = extract_header_subset(ds_first, series_items=items)

    # computed meta (compact)
    vx, vy, vz = voxel_sizes_from_affine(affine)
    dt_ms = float((t_grid_ms[1] - t_grid_ms[0]) if len(t_grid_ms) > 1 else 0.0)

    meta = {
        "selected_series": {"series_description": ser_desc, "series_number": ser_num, "n_instances": len(items)},
        "dz_mm": float(dz),
        "RR_ms": float(RR),
        "dt_ms": float(dt_ms),
        "voxel_sizes_mm": (float(vx), float(vy), float(vz)),
        "num_slices": int(vol_XYZT.shape[2]),
        "num_frames": int(vol_XYZT.shape[3]),
    }

    return vol_XYZT, affine, t_grid_ms, meta, header_subset


# -------------------------
# Geometry
# -------------------------
def affine_from_dicom_ras(ds0, dz: float) -> np.ndarray:
    """
    Return 4x4 RAS affine from DICOM IOP/IPP/spacing.

    Note: DICOM PixelSpacing = [row, col].
    """
    iop = [float(x) for x in ds0.ImageOrientationPatient]
    r = np.array(iop[:3], dtype=np.float64)
    c = np.array(iop[3:], dtype=np.float64)
    r /= (np.linalg.norm(r) + 1e-12)
    c /= (np.linalg.norm(c) + 1e-12)
    n = np.cross(r, c)
    n /= (np.linalg.norm(n) + 1e-12)

    row_spacing = float(ds0.PixelSpacing[0])
    col_spacing = float(ds0.PixelSpacing[1])

    ipp = np.array([float(x) for x in ds0.ImagePositionPatient], dtype=np.float64)

    # LPS affine (voxel indices i,j,k -> world)
    M_lps = np.eye(4, dtype=np.float64)
    M_lps[:3, 0] = r * col_spacing
    M_lps[:3, 1] = c * row_spacing
    M_lps[:3, 2] = n * dz
    M_lps[:3, 3] = ipp

    # LPS -> RAS
    LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
    return LPS2RAS @ M_lps


def voxel_sizes_from_affine(affine_ras: np.ndarray) -> Tuple[float, float, float]:
    vx = float(np.linalg.norm(affine_ras[:3, 0]))
    vy = float(np.linalg.norm(affine_ras[:3, 1]))
    vz = float(np.linalg.norm(affine_ras[:3, 2]))
    return vx, vy, vz


# -------------------------
# Header subset extraction (curated, no PHI)
# -------------------------
def extract_header_subset(ds_first, *, series_items: Optional[List[Tuple[str, Any]]] = None) -> Dict[str, Any]:
    hs: Dict[str, Any] = {
        "provenance": {
            "Manufacturer": _safe_get(ds_first, "Manufacturer"),
            "ManufacturerModelName": _safe_get(ds_first, "ManufacturerModelName"),
            "SoftwareVersions": _safe_get(ds_first, "SoftwareVersions"),
            "MagneticFieldStrength": _to_float(_safe_get(ds_first, "MagneticFieldStrength")),
        },
        "series": {
            "Modality": _safe_get(ds_first, "Modality"),
            "StudyInstanceUID": _safe_get(ds_first, "StudyInstanceUID"),
            "SeriesInstanceUID": _safe_get(ds_first, "SeriesInstanceUID"),
            "SeriesDescription": _safe_get(ds_first, "SeriesDescription"),
            "SeriesNumber": str(_safe_get(ds_first, "SeriesNumber", "NA")),
            "ProtocolName": _safe_get(ds_first, "ProtocolName"),
            "SequenceName": _safe_get(ds_first, "SequenceName"),
            "ImageType": _safe_get(ds_first, "ImageType"),
        },
        "geometry": {
            "Rows": int(_safe_get(ds_first, "Rows")) if _safe_get(ds_first, "Rows") is not None else None,
            "Columns": int(_safe_get(ds_first, "Columns")) if _safe_get(ds_first, "Columns") is not None else None,
            "PixelSpacing": _to_list_float(_safe_get(ds_first, "PixelSpacing")),
            "SliceThickness": _to_float(_safe_get(ds_first, "SliceThickness")),
            "SpacingBetweenSlices": _to_float(_safe_get(ds_first, "SpacingBetweenSlices")),
            "ImageOrientationPatient": _to_list_float(_safe_get(ds_first, "ImageOrientationPatient")),
            "ImagePositionPatient": _to_list_float(_safe_get(ds_first, "ImagePositionPatient")),
            "PatientPosition": _safe_get(ds_first, "PatientPosition"),
        },
        "timing": {
            "HeartRate": _to_float(_safe_get(ds_first, "HeartRate")),
            "NominalInterval": _to_float(_safe_get(ds_first, "NominalInterval")),
            "CardiacNumberOfImages": _to_float(_safe_get(ds_first, "CardiacNumberOfImages")),
            "ImagesInAcquisition": _to_float(_safe_get(ds_first, "ImagesInAcquisition")),
            "TriggerTime_present": _safe_get(ds_first, "TriggerTime") is not None,
        },
        "mr_params": {
            "RepetitionTime": _to_float(_safe_get(ds_first, "RepetitionTime")),
            "EchoTime": _to_float(_safe_get(ds_first, "EchoTime")),
            "FlipAngle": _to_float(_safe_get(ds_first, "FlipAngle")),
            "MRAcquisitionType": _safe_get(ds_first, "MRAcquisitionType"),
            "ScanningSequence": _safe_get(ds_first, "ScanningSequence"),
            "SequenceVariant": _safe_get(ds_first, "SequenceVariant"),
            "InPlanePhaseEncodingDirection": _safe_get(ds_first, "InPlanePhaseEncodingDirection"),
        },
    }

    if series_items is not None:
        trig = []
        inst = []
        for _, ds in series_items:
            tt = _safe_get(ds, "TriggerTime")
            if tt is not None:
                try:
                    trig.append(float(tt))
                except Exception:
                    pass
            inum = _safe_get(ds, "InstanceNumber")
            if inum is not None:
                try:
                    inst.append(int(inum))
                except Exception:
                    pass

        hs["series_stats"] = {
            "n_instances": len(series_items),
            "n_trigger_times": len(trig),
            "n_unique_trigger_times": len(set(trig)) if trig else 0,
            "instance_number_min": min(inst) if inst else None,
            "instance_number_max": max(inst) if inst else None,
        }

    return hs


# -------------------------
# Small utilities
# -------------------------
def _round_tup(vals, nd: int) -> Tuple[float, ...]:
    return tuple(np.round([float(x) for x in vals], nd).tolist())


def _safe_get(ds, name: str, default=None):
    return getattr(ds, name, default)


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_list_float(x) -> Optional[List[float]]:
    if x is None:
        return None
    try:
        return [float(v) for v in x]
    except Exception:
        return None


# -------------------------
# DICOM grouping / sorting
# -------------------------
def _group_key(ds) -> Tuple[Any, ...]:
    """
    Per-slice key to group frames into a single slice.
    Includes IOP/IPP/PixelSpacing, plus a series label.
    """
    iop = _round_tup(ds.ImageOrientationPatient, 6)
    ipp = _round_tup(ds.ImagePositionPatient, 3)
    ps = _round_tup(ds.PixelSpacing, 4)
    return (
        str(_safe_get(ds, "SeriesDescription", "UNKNOWN")),
        str(_safe_get(ds, "SeriesNumber", "NA")),
        iop,
        ipp,
        ps,
    )


def _time_key(ds) -> float:
    """
    Time ordering key for frames within a slice.

    Prefer TriggerTime. If missing, fall back to InstanceNumber ordering.
    Avoid interpreting AcquisitionTime as float (HHMMSS.frac) because it is not linear time.
    """
    t = _safe_get(ds, "TriggerTime", None)
    if t is not None:
        try:
            return float(t)
        except Exception:
            pass
    return float(_safe_get(ds, "InstanceNumber", 0) or 0)


def _load_headers(dicom_dir: str) -> List[Tuple[str, Any]]:
    headers: List[Tuple[str, Any]] = []
    needed = ["Rows", "Columns", "PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient"]

    for dp, _, files in os.walk(dicom_dir):
        for f in files:
            if not f.lower().endswith(".dcm"):
                continue
            fp = os.path.join(dp, f)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not all(hasattr(ds, k) for k in needed):
                continue
            headers.append((fp, ds))
    return headers


def _load_pixel(fp: str) -> np.ndarray:
    ds = pydicom.dcmread(fp, force=True)
    arr = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(_safe_get(ds, "RescaleSlope", 1.0))
    inter = float(_safe_get(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or inter != 0.0:
        arr = arr * slope + inter
    return arr


def _slice_sort_val(ds) -> Tuple[float, np.ndarray]:
    """
    Projection of IPP on slice normal. Used to order slices base→apex.
    """
    r = np.array(ds.ImageOrientationPatient[:3], float)
    c = np.array(ds.ImageOrientationPatient[3:], float)
    r /= (np.linalg.norm(r) + 1e-12)
    c /= (np.linalg.norm(c) + 1e-12)
    n = np.cross(r, c)
    n /= (np.linalg.norm(n) + 1e-12)
    ipp = np.array(ds.ImagePositionPatient, float)
    return float(np.dot(ipp, n)), n


def _interp_time(times_ms, images_T_HW, t_grid, RR) -> np.ndarray:
    """
    Periodic linear interpolation in time to a common grid.
    times_ms: list length T' (may be unsorted)
    images_T_HW: array [T', H, W]
    t_grid: array [T]
    RR: cycle length in ms (for wrap-around)
    """
    times = np.asarray(times_ms, float)
    imgs = images_T_HW.astype(np.float32, copy=False)

    times = np.mod(times, RR)
    idx = np.argsort(times)
    times = times[idx]
    imgs = imgs[idx]

    # periodic extension for wrap interpolation
    times_ext = np.r_[times - RR, times, times + RR]
    imgs_ext = np.concatenate([imgs, imgs, imgs], axis=0)

    out = np.empty((len(t_grid),) + imgs.shape[1:], np.float32)
    for i, t in enumerate(t_grid):
        j = np.searchsorted(times_ext, t) - 1
        j = np.clip(j, 0, len(times_ext) - 2)
        t0, t1 = times_ext[j], times_ext[j + 1]
        if t1 == t0:
            out[i] = imgs_ext[j]
        else:
            w = (t - t0) / (t1 - t0)
            out[i] = (1 - w) * imgs_ext[j] + w * imgs_ext[j + 1]
    return out  # [T,H,W]

