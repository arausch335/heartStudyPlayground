"""Routines for extracting rectangular regions from point clouds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .point_cloud import PointCloud


@dataclass
class Plane:
    point: np.ndarray  # on-plane point
    normal: np.ndarray  # unit normal


@dataclass
class Rectangle:
    plane: Plane
    center: np.ndarray
    u_axis: np.ndarray  # in-plane unit vector
    v_axis: np.ndarray  # in-plane unit vector orthogonal to u_axis
    half_lengths: Tuple[float, float]

    @property
    def normal(self) -> np.ndarray:
        return self.plane.normal

    def as_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "u_axis": self.u_axis.tolist(),
            "v_axis": self.v_axis.tolist(),
            "half_lengths": list(self.half_lengths),
        }


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return v / norm


def estimate_plane(points: np.ndarray) -> Plane:
    centroid = points.mean(axis=0)
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    return Plane(point=centroid, normal=_normalize(normal))


def ransac_plane(points: np.ndarray, iterations: int = 500, distance_threshold: float = 1e-2,
                 normal_hint: Optional[np.ndarray] = None, angle_tolerance: float = np.deg2rad(10)) -> Tuple[Plane, np.ndarray]:
    """Fit a plane using RANSAC with optional normal alignment constraints."""

    if len(points) < 3:
        raise ValueError("At least three points are required for plane estimation.")

    best_inliers: np.ndarray = np.array([], dtype=int)
    best_plane: Optional[Plane] = None

    rng = np.random.default_rng()

    for _ in range(iterations):
        sample_indices = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[sample_indices]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) == 0:
            continue
        normal = _normalize(normal)

        if normal_hint is not None:
            angle = np.arccos(np.clip(np.abs(np.dot(normal, normal_hint)), -1.0, 1.0))
            if angle > angle_tolerance:
                continue

        plane_point = p0
        distances = np.abs((points - plane_point) @ normal)
        inliers = np.where(distances <= distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = Plane(point=plane_point, normal=normal)

    if best_plane is None:
        raise RuntimeError("Failed to fit a plane to the provided points.")

    # Refine using inliers with PCA
    refined_plane = estimate_plane(points[best_inliers])
    return refined_plane, best_inliers


def _rectangle_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis_candidate = np.array([1.0, 0.0, 0.0]) if np.abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = _normalize(np.cross(normal, axis_candidate))
    v_axis = _normalize(np.cross(normal, u_axis))
    return u_axis, v_axis


def _project_to_plane(points: np.ndarray, plane: Plane, u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    centered = points - plane.point
    u_coords = centered @ u_axis
    v_coords = centered @ v_axis
    return np.column_stack([u_coords, v_coords])


def rectangle_from_inliers(points: np.ndarray, plane: Plane) -> Rectangle:
    u_axis, v_axis = _rectangle_axes(plane.normal)
    uv = _project_to_plane(points, plane, u_axis, v_axis)
    min_uv = uv.min(axis=0)
    max_uv = uv.max(axis=0)
    center_uv = 0.5 * (min_uv + max_uv)
    half_lengths = 0.5 * (max_uv - min_uv)
    center = plane.point + center_uv[0] * u_axis + center_uv[1] * v_axis
    return Rectangle(
        plane=plane,
        center=center,
        u_axis=u_axis,
        v_axis=v_axis,
        half_lengths=(float(half_lengths[0]), float(half_lengths[1])),
    )


def rectangle_mask(cloud: PointCloud, rectangle: Rectangle, distance_threshold: float) -> np.ndarray:
    centered = cloud.points - rectangle.center
    distance_to_plane = centered @ rectangle.normal
    planar_distance = np.abs(distance_to_plane)

    u_coords = centered @ rectangle.u_axis
    v_coords = centered @ rectangle.v_axis

    within_plane = planar_distance <= distance_threshold
    within_u = np.abs(u_coords) <= rectangle.half_lengths[0] + distance_threshold
    within_v = np.abs(v_coords) <= rectangle.half_lengths[1] + distance_threshold

    return within_plane & within_u & within_v


def detect_rectangles(cloud: PointCloud, distance_threshold: float = 1e-2,
                      angle_tolerance: float = np.deg2rad(10)) -> List[Rectangle]:
    """Detect two parallel rectangles and a perpendicular rectangle via iterative RANSAC."""

    remaining_indices = np.arange(len(cloud.points))
    rectangles: List[Rectangle] = []
    points = cloud.points

    # First rectangle
    plane1, inliers1 = ransac_plane(points[remaining_indices], distance_threshold=distance_threshold)
    rectangles.append(rectangle_from_inliers(points[remaining_indices][inliers1], plane1))
    remaining_indices = np.setdiff1d(remaining_indices, remaining_indices[inliers1], assume_unique=True)

    # Second rectangle parallel to first
    plane2, inliers2 = ransac_plane(
        points[remaining_indices],
        distance_threshold=distance_threshold,
        normal_hint=rectangles[0].normal,
        angle_tolerance=angle_tolerance,
    )
    rectangles.append(rectangle_from_inliers(points[remaining_indices][inliers2], plane2))
    remaining_indices = np.setdiff1d(remaining_indices, remaining_indices[inliers2], assume_unique=True)

    # Third rectangle perpendicular to first
    perpendicular_hint = np.cross(rectangles[0].normal, rectangles[1].normal)
    if np.linalg.norm(perpendicular_hint) < 1e-6:
        # Fall back to an orthogonal in-plane axis of the first rectangle if the
        # two normals are nearly identical.
        perpendicular_hint = _rectangle_axes(rectangles[0].normal)[0]
    plane3, inliers3 = ransac_plane(
        points[remaining_indices],
        distance_threshold=distance_threshold,
        normal_hint=perpendicular_hint,
        angle_tolerance=angle_tolerance,
    )
    rectangles.append(rectangle_from_inliers(points[remaining_indices][inliers3], plane3))

    return rectangles


def combined_mask(cloud: PointCloud, rectangles: Sequence[Rectangle], distance_threshold: float) -> np.ndarray:
    mask = np.zeros(len(cloud.points), dtype=bool)
    for rectangle in rectangles:
        mask |= rectangle_mask(cloud, rectangle, distance_threshold)
    return mask


def rectangle_corners(rectangle: Rectangle) -> np.ndarray:
    """Return the four 3D corners of a rectangle as (4, 3)."""

    u = rectangle.u_axis * rectangle.half_lengths[0]
    v = rectangle.v_axis * rectangle.half_lengths[1]
    center = rectangle.center
    return np.array([
        center + u + v,
        center + u - v,
        center - u - v,
        center - u + v,
    ])
