"""Generate synthetic point clouds with three rectangular regions for testing.

This script avoids third-party dependencies so it can run in constrained
environments while still producing ``.ply`` and ``.npy`` outputs.
"""
from __future__ import annotations

import math
import random
import struct
from pathlib import Path
from typing import Iterable, Sequence, Tuple

ROOT = Path(__file__).resolve().parent


# ---- Small vector helpers ----

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _scale(a: Sequence[float], s: float) -> Tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _normalize(v: Sequence[float]) -> Tuple[float, float, float]:
    norm = math.sqrt(_dot(v, v))
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _apply_rotation(rot: Sequence[Sequence[float]], v: Sequence[float]) -> Tuple[float, float, float]:
    return (
        rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
        rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
        rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
    )


# ---- Sampling helpers ----

def sample_rectangle(center: Tuple[float, float, float], normal: Tuple[float, float, float],
                     u_axis: Tuple[float, float, float], v_axis: Tuple[float, float, float],
                     half_lengths: Tuple[float, float], count: int, noise: float) -> list[Tuple[float, float, float]]:
    normal_v = _normalize(normal)
    u = _normalize(u_axis)
    v = _normalize(v_axis)
    rng = random.Random(12345)

    points: list[Tuple[float, float, float]] = []
    for _ in range(count):
        u_s = rng.uniform(-half_lengths[0], half_lengths[0])
        v_s = rng.uniform(-half_lengths[1], half_lengths[1])
        base = _add(_add(center, _scale(u, u_s)), _scale(v, v_s))
        jitter = _scale(normal_v, rng.gauss(0.0, noise))
        points.append(_add(base, jitter))
    return points


def axis_angle_rotation(axis: Tuple[float, float, float], angle_rad: float) -> list[list[float]]:
    axis_n = _normalize(axis)
    x, y, z = axis_n
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def build_cloud() -> list[Tuple[float, float, float]]:
    rectangles = [
        # Parallel pair (z ~ 0 and z ~ 0.08)
        ((0.0, 0.0, 0.0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0.06, 0.2)),
        ((0.0, 0.05, 0.08), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0.05, 0.18)),
        # Perpendicular rectangle on the x = -0.08 plane
        ((-0.08, 0.02, 0.04), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.18, 0.08)),
    ]

    parts = [sample_rectangle(c, n, u, v, hl, count=900, noise=0.0015) for c, n, u, v, hl in rectangles]
    cloud = [p for part in parts for p in part]

    # Add light background noise to make the detection more realistic
    rng = random.Random(2024)
    for _ in range(400):
        cloud.append((
            rng.uniform(-0.15, 0.15),
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.05, 0.12),
        ))
    return cloud


# ---- Writers ----

def write_ply(path: Path, points: Iterable[Sequence[float]]) -> None:
    pts = list(points)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(pts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("end_header\n")
        for x, y, z in pts:
            fh.write(f"{x} {y} {z}\n")


def write_obj(path: Path, points: Iterable[Sequence[float]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for x, y, z in points:
            fh.write(f"v {x} {y} {z}\n")


def write_npy(path: Path, points: Iterable[Sequence[float]]) -> None:
    pts = list(points)
    header_dict = {"descr": "<f8", "fortran_order": False, "shape": (len(pts), 3)}
    header_str = str(header_dict)

    magic = b"\x93NUMPY"
    version = b"\x01\x00"  # Version 1.0
    preamble = len(magic) + len(version) + 2  # 2 bytes for header length

    header_body = header_str.encode("latin1")
    padding = 16 - ((preamble + len(header_body) + 1) % 16)
    if padding == 16:
        padding = 0
    header = header_body + b" " * padding + b"\n"

    with path.open("wb") as fh:
        fh.write(magic)
        fh.write(version)
        fh.write(struct.pack("<H", len(header)))
        fh.write(header)
        for x, y, z in pts:
            fh.write(struct.pack("<ddd", x, y, z))


# ---- Entry point ----

def main() -> None:
    target_points = build_cloud()

    rotation = axis_angle_rotation(axis=(0.2, 0.6, 0.1), angle_rad=math.radians(18))
    translation = (0.015, -0.008, 0.02)

    source_points = [_add(_apply_rotation(rotation, p), translation) for p in target_points]

    ROOT.mkdir(exist_ok=True)
    write_ply(ROOT / "sample_target.ply", target_points)
    write_ply(ROOT / "sample_source.ply", source_points)
    write_obj(ROOT / "sample_target.obj", target_points)
    write_obj(ROOT / "sample_source.obj", source_points)
    write_npy(ROOT / "sample_target.npy", target_points)
    write_npy(ROOT / "sample_source.npy", source_points)

    print("Wrote sample point clouds to:")
    for name in ["sample_target.ply", "sample_source.ply", "sample_target.obj", "sample_source.obj", "sample_target.npy", "sample_source.npy"]:
        print(f"  {ROOT / name}")


if __name__ == "__main__":
    main()
