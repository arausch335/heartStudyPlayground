"""Utilities for loading, representing, and preprocessing point clouds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PointCloud:
    """Simple point cloud container with optional normals."""

    points: np.ndarray  # (N, 3)
    normals: np.ndarray | None = None  # (N, 3)

    def copy(self) -> "PointCloud":
        return PointCloud(points=self.points.copy(), normals=None if self.normals is None else self.normals.copy())

    def with_normals(self, normals: np.ndarray) -> "PointCloud":
        return PointCloud(points=self.points, normals=normals)

    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> "PointCloud":
        rotated = (rotation @ self.points.T).T + translation
        normals = None if self.normals is None else (rotation @ self.normals.T).T
        return PointCloud(points=rotated, normals=normals)


SUPPORTED_EXTENSIONS = {".npy", ".ply", ".obj"}


def _load_ply(path: Path) -> np.ndarray:
    """Load an ASCII PLY file containing vertex positions.

    Only ASCII PLY files are supported in order to keep dependencies minimal.
    If normals are present they are ignored; faces and other elements are
    skipped.
    """

    with path.open("r", encoding="utf-8") as fh:
        header_lines = []
        for line in fh:
            header_lines.append(line.strip())
            if line.strip() == "end_header":
                break

        vertex_count = None
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                try:
                    vertex_count = int(parts[2])
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Invalid vertex count in PLY header: {line}") from exc
        if vertex_count is None:
            raise ValueError("PLY file missing vertex count in header.")

        vertices = []
        for _ in range(vertex_count):
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected end of PLY file while reading vertices.")
            parts = line.strip().split()
            if len(parts) < 3:
                raise ValueError(f"Vertex line has insufficient components: {line.strip()}")
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.asarray(vertices, dtype=float)


def _load_obj(path: Path) -> np.ndarray:
    """Load vertex positions from an OBJ file.

    Faces and other records are ignored; only ``v`` records are read. Normals
    and texture coordinates are ignored to keep the loader lightweight.
    """

    vertices = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.lstrip().startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"OBJ vertex line has insufficient components: {line.strip()}")
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError("OBJ file contained no vertex records.")
    return np.asarray(vertices, dtype=float)


def load_point_cloud(path: str | Path) -> PointCloud:
    """Load a point cloud from ``.npy``, ASCII ``.ply``, or ``.obj`` files."""

    file_path = Path(path)
    if file_path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported point cloud extension: {file_path.suffix}. Expected one of {SUPPORTED_EXTENSIONS}")

    if file_path.suffix == ".npy":
        points = np.load(file_path)
    elif file_path.suffix == ".ply":
        points = _load_ply(file_path)
    else:
        points = _load_obj(file_path)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must be an (N, 3) array. Got {points.shape} instead.")
    return PointCloud(points=points.astype(float))


def save_point_cloud(path: str | Path, cloud: PointCloud) -> None:
    """Persist a point cloud as ``.npy``, ASCII ``.ply``, or ``.obj``.

    Normals are written when present for ``.ply`` and ``.obj`` outputs. Faces
    are not exported.
    """

    file_path = Path(path)
    suffix = file_path.suffix
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported point cloud extension: {suffix}. Expected one of {SUPPORTED_EXTENSIONS}")

    if suffix == ".npy":
        np.save(file_path, cloud.points)
    elif suffix == ".ply":
        normals = cloud.normals
        with file_path.open("w", encoding="utf-8") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(cloud.points)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            if normals is not None:
                fh.write("property float nx\nproperty float ny\nproperty float nz\n")
            fh.write("end_header\n")
            for idx, point in enumerate(cloud.points):
                if normals is None:
                    fh.write(f"{point[0]} {point[1]} {point[2]}\n")
                else:
                    normal = normals[idx]
                    fh.write(f"{point[0]} {point[1]} {point[2]} {normal[0]} {normal[1]} {normal[2]}\n")
    else:  # .obj
        with file_path.open("w", encoding="utf-8") as fh:
            for point in cloud.points:
                fh.write(f"v {point[0]} {point[1]} {point[2]}\n")
            if cloud.normals is not None:
                for normal in cloud.normals:
                    fh.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")


def estimate_normals(cloud: PointCloud, k_neighbors: int = 30) -> PointCloud:
    """Estimate per-point normals using PCA on nearest neighbors.

    The function adds consistently oriented normals to the returned cloud. For
    small datasets, ``k_neighbors`` can be reduced to keep computations cheap.
    """

    points = cloud.points
    if len(points) < 3:
        raise ValueError("At least three points are required to estimate normals.")

    tree = cKDTree(points)
    normals = np.zeros_like(points)

    for idx, point in enumerate(points):
        distances, indices = tree.query(point, k=min(k_neighbors, len(points)))
        neighborhood = points[indices]
        covariance = np.cov(neighborhood.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # Orient normals to point outward from the centroid for consistency
        centroid_direction = point - neighborhood.mean(axis=0)
        if np.dot(normal, centroid_direction) < 0:
            normal = -normal
        normals[idx] = normal / np.linalg.norm(normal)

    return cloud.with_normals(normals)


def center_cloud(cloud: PointCloud) -> Tuple[PointCloud, np.ndarray]:
    """Center the cloud around its centroid, returning the centered cloud and centroid."""

    centroid = cloud.points.mean(axis=0)
    return PointCloud(points=cloud.points - centroid, normals=cloud.normals), centroid


def apply_mask(cloud: PointCloud, mask: Iterable[bool]) -> PointCloud:
    """Return a new cloud that only contains points where ``mask`` is True."""

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape[0] != cloud.points.shape[0]:
        raise ValueError("Mask length does not match number of points.")
    filtered_points = cloud.points[mask_array]
    filtered_normals = None if cloud.normals is None else cloud.normals[mask_array]
    return PointCloud(points=filtered_points, normals=filtered_normals)
