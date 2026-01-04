# icp.py (improved)

from typing import Tuple, Dict, Optional, List
import numpy as np
import open3d as o3d


def _make_o3d_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return cloud


def _estimate_normals_in_place(
    cloud: o3d.geometry.PointCloud, radius: float, max_nn: int = 50
) -> None:
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    cloud.normalize_normals()


def point_to_plane_icp_multiscale(
    source: np.ndarray,
    target: np.ndarray,
    *,
    voxel_sizes: List[float] = (5.0, 2.5, 1.0),          # mm-ish
    max_corr_distances: List[float] = (20.0, 10.0, 5.0),
    max_iterations: List[int] = (40, 30, 20),
    normal_radius_factor: float = 5.0,
    init_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Multiscale point-to-plane ICP using Open3D.
    """
    assert len(voxel_sizes) == len(max_corr_distances) == len(max_iterations)

    # base clouds
    source_full = _make_o3d_cloud(source)
    target_full = _make_o3d_cloud(target)

    # estimate normals on full-res target once
    bbox = np.asarray(target_full.get_max_bound()) - np.asarray(target_full.get_min_bound())
    normal_radius = float(np.linalg.norm(bbox) / normal_radius_factor)
    _estimate_normals_in_place(target_full, radius=normal_radius)

    if init_transform is None:
        T = np.eye(4)
    else:
        T = init_transform.copy()

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        loss=o3d.pipelines.registration.loss.RobustKernel(
            o3d.pipelines.registration.loss.RobustKernelType.Huber,  # soft outlier rejection
            1.0,
        )
    )

    rmse_history = []

    for voxel, max_corr, max_iter in zip(voxel_sizes, max_corr_distances, max_iterations):
        # downsample clouds
        src = source_full.voxel_down_sample(voxel)
        tgt = target_full.voxel_down_sample(voxel)

        # reuse normals from full cloud
        tgt.normals = target_full.normals

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        )

        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            max_corr,
            T,
            estimation_method=estimation,
            criteria=criteria,
        )

        T = result.transformation
        rmse_history.append(float(result.inlier_rmse))

    info = {
        "rmse_history": rmse_history,
        "final_rmse": rmse_history[-1] if rmse_history else None,
    }
    return T, rmse_history[-1] if rmse_history else 0.0, info
