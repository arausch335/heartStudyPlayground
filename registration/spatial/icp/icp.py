# icp.py

from typing import Tuple, Dict, Optional
import numpy as np
import open3d as o3d


def _make_o3d_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert (N,3) numpy array to an Open3D PointCloud.
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return cloud


def _estimate_normals_in_place(
    cloud: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int = 30,
) -> None:
    """
    Estimate normals for a point cloud in-place.
    """
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    cloud.normalize_normals()


def point_to_plane_icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_correspondence_distance: float = 5.0,
    max_iterations: int = 50,
    normal_radius: Optional[float] = None,
    init_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Point-to-plane ICP using Open3D.

    We solve for a rigid transform that aligns source -> target, minimizing
    point-to-plane distances on the target surface.

    Parameters
    ----------
    source : (Ns, 3)
        Source points (e.g., retractor from current frame, in local/world coords).
    target : (Nt, 3)
        Target points (e.g., retractor from target frame, in OR coords).
    max_correspondence_distance : float
        Maximum distance (in same units as points) to consider point pairs as correspondences.
    max_iterations : int
        Maximum number of ICP iterations.
    normal_radius : float or None
        Neighborhood radius used to estimate target normals. If None, choose
        a radius based on target's bounding box size.
    init_transform : (4,4) or None
        Optional initial guess for the source->target transform.

    Returns
    -------
    T : (4, 4) np.ndarray
        Rigid transform mapping source -> target coordinates.
    rmse : float
        Inlier RMSE from Open3D's registration result.
    info : dict
        Extra info, e.g. {"iterations": int, "fitness": float, "rmse": float}.
    """
    # --- 1. Build Open3D point clouds ---
    source_cloud = _make_o3d_cloud(source)
    target_cloud = _make_o3d_cloud(target)

    # --- 2. Estimate normals for target (required for point-to-plane) ---
    if normal_radius is None:
        # pick a radius based on scale of target cloud
        target_bounds = np.asarray(target_cloud.get_max_bound()) - np.asarray(
            target_cloud.get_min_bound()
        )
        normal_radius = float(np.linalg.norm(target_bounds) * 0.1)  # 10% of bbox diag

    _estimate_normals_in_place(target_cloud, radius=normal_radius)

    # --- 3. Initial transform ---
    if init_transform is None:
        init_transform = np.eye(4)

    # --- 4. Run Open3D point-to-plane ICP ---
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations
    )

    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance,
        init_transform,
        estimation_method=estimation,
        criteria=criteria,
    )

    T = result.transformation  # 4x4 source -> target
    rmse = float(result.inlier_rmse)
    info = {
        "iterations": max_iterations,      # Open3D doesn't expose actual count directly
        "fitness": float(result.fitness),  # inlier fraction
        "rmse": rmse,
    }

    return T, rmse, info
