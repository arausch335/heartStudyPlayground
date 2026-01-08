import numpy as np


def to_hom(points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got {points.shape}")
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.concatenate([points, ones], axis=1).T
    return points_h


def from_hom(points_h: np.ndarray) -> np.ndarray:
    points_h = np.asarray(points_h, dtype=float)
    if points_h.ndim != 2 or points_h.shape[0] != 4:
        raise ValueError(f"Expected (4,N) homogeneous points, got {points_h.shape}")
    return points_h.T[:, :3]


def assert_T_valid(T: np.ndarray, *, rigid: bool = True, atol: float = 1e-6) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4) transform, got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("Transform contains non-finite values")
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        raise ValueError("Transform last row must be [0,0,0,1] within tolerance")
    if rigid:
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=atol):
            raise ValueError("Transform rotation is not orthonormal")
        det = np.linalg.det(R)
        if not np.allclose(det, 1.0, atol=atol):
            raise ValueError(f"Transform rotation determinant must be 1, got {det}")
    return T


def apply_T(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    T = assert_T_valid(T, rigid=False)
    points_h = to_hom(points_xyz)
    out_h = T @ points_h
    return from_hom(out_h)


def compose_T(transforms) -> np.ndarray:
    T_total = np.eye(4, dtype=float)
    for T in transforms:
        if T is None:
            continue
        T = assert_T_valid(T, rigid=False)
        T_total = T @ T_total
    return T_total


def invert_T(T: np.ndarray) -> np.ndarray:
    T = assert_T_valid(T, rigid=False)
    return np.linalg.inv(T)
