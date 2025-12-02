"""Point-to-plane ICP implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from .point_cloud import PointCloud, estimate_normals


@dataclass
class ICPResult:
    rotation: np.ndarray
    translation: np.ndarray
    correspondences: np.ndarray
    iterations: int
    converged: bool


class PointToPlaneICP:
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-5, correspondence_threshold: float = 0.05):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.correspondence_threshold = correspondence_threshold

    def register(self, source: PointCloud, target: PointCloud) -> Tuple[ICPResult, PointCloud]:
        if target.normals is None:
            target = estimate_normals(target)

        rotation = np.eye(3)
        translation = np.zeros(3)
        previous_error = np.inf
        transformed = source
        last_correspondences = np.array([], dtype=int)

        target_tree = cKDTree(target.points)

        for iteration in range(1, self.max_iterations + 1):
            transformed_points = (rotation @ source.points.T).T + translation
            distances, indices = target_tree.query(transformed_points)

            valid = distances < self.correspondence_threshold
            if not np.any(valid):
                break

            last_correspondences = indices[valid]

            matched_source = transformed_points[valid]
            matched_target = target.points[indices[valid]]
            matched_normals = target.normals[indices[valid]]

            A_rows = []
            b_values = []

            for src_transformed, tgt, normal in zip(matched_source, matched_target, matched_normals):
                cross = np.cross(src_transformed, normal)
                A_rows.append(np.hstack((cross, normal)))
                b_values.append(np.dot(normal, tgt - src_transformed))

            A = np.vstack(A_rows)
            b = np.array(b_values)

            # Solve for twist (rx, ry, rz, tx, ty, tz)
            try:
                twist, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                break

            rotation_update = _skew_to_rotation(twist[:3])
            translation_update = twist[3:]

            rotation = rotation_update @ rotation
            translation = rotation_update @ translation + translation_update

            error = np.mean(np.abs(b))
            if np.abs(previous_error - error) < self.tolerance:
                converged = True
                transformed = PointCloud(points=transformed_points)
                return ICPResult(rotation=rotation, translation=translation, correspondences=last_correspondences, iterations=iteration, converged=converged), transformed
            previous_error = error

        transformed_points = (rotation @ source.points.T).T + translation
        transformed = PointCloud(points=transformed_points)
        return ICPResult(rotation=rotation, translation=translation, correspondences=last_correspondences, iterations=iteration, converged=False), transformed


def _skew_to_rotation(omega: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    axis = omega / theta
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
