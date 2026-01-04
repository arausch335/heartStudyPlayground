from typing import Tuple
from scipy.interpolate import griddata
import numpy as np


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector. If the norm is tiny, return the input unchanged.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def fit_plane_svd(points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit plane ax + by + cz + d = 0 via SVD.

    Parameters
    ----------
    points_3d : (N, 3) array

    Returns
    -------
    normal : (3,) unit vector
    centroid : (3,) point lying on the plane
    """
    centroid = points_3d.mean(axis=0)
    centered = points_3d - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1, :]
    normal = normalize_vec(normal)
    return normal, centroid


def build_plane_basis(normal: np.ndarray):
    """
    Given a plane normal, build an orthonormal basis (u, v, n),
    where u, v lie in the plane and n is the (normalized) normal.

    Returns
    -------
    u, v, n : (3,) vectors
    """
    n = normalize_vec(normal)
    # choose a reference vector not parallel to n
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u = normalize_vec(u)
    v = np.cross(n, u)
    return u, v, n


def center_of_mass(points: np.ndarray) -> np.ndarray:
    """
    Compute center of mass (simple mean) of a point cloud.

    points : (N, 3)

    Returns
    -------
    (3,) array
    """
    return points.mean(axis=0)


### --- VISUALIZE --- ###


def build_subset_indices_dict(retractor=None, epicardium=None):
    subset_indices_dict = dict()
    if retractor is not None:
        subset_indices_dict['retractor'] = retractor
    if epicardium is not None:
        subset_indices_dict['epicardium'] = epicardium

    return subset_indices_dict


def indices_to_subset_ids(num_points, subset_indices):

    subset_ids = -np.ones(num_points, dtype=int)
    subset_names = {}

    for sid, (name, idxs) in enumerate(subset_indices.items()):
        subset_ids[idxs] = sid
        subset_names[sid] = name

    return subset_ids, subset_names


def convert_point_to_meshgrid(points, values=None):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = np.linspace(y.min(), y.max(), 100)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)

    z_mesh = griddata((x, y), z, (x_mesh, y_mesh), method='nearest')

    v_mesh = None
    if values is not None:
        values = np.asarray(values)
        v_mesh = griddata((x, y), values, (x_mesh, y_mesh), method='nearest')

    return x_mesh, y_mesh, z_mesh, v_mesh


def apply_T(T, points):
    """
    Apply a 4x4 homogeneous transform to a set of 3D points.

    Parameters
    ----------
    T : (4,4) array-like
        Homogeneous transform matrix.
        Convention: points are treated as column vectors [x y z 1]^T,
        and the transform is applied as: p' = T @ p
    points : (N,3) array-like
        Input points.

    Returns
    -------
    points_out : (N,3) ndarray
        Transformed points.
    """
    T = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float)

    if T.shape != (4, 4):
        raise ValueError(f"T must be (4,4), got {T.shape}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")

    if pts.shape[0] == 0:
        # empty input, return empty output with correct shape
        return pts.copy()

    # Homogeneous coordinates
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack([pts, ones])           # (N,4)

    # Apply transform: (4,4) @ (4,N) -> (4,N)
    pts_h_t = (T @ pts_h.T).T                # (N,4)

    # Drop homogeneous coordinate
    return pts_h_t[:, :3]

