import numpy as np

from registration.spatial.utilities.utilities import (
    normalize_vec,
    fit_plane_svd,
    build_plane_basis,
    center_of_mass,
)


def ransac_plane(points, distance_threshold=2.0, max_iterations=1000):
    """
    Fit one dominant plane using RANSAC.

    Returns
    -------
    normal : (3,)
    point_on_plane : (3,)
    inlier_mask : (N,) bool
    """
    N = len(points)
    if N < 3:
        raise ValueError("Need at least 3 points for plane RANSAC")

    rng = np.random.default_rng()
    best_inlier_count = 0
    best_inlier_mask = None

    for _ in range(int(max_iterations)):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = points[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-6:
            continue

        n = normalize_vec(n)

        dists = np.abs((points - p0) @ n)
        inliers = dists < float(distance_threshold)
        count = int(inliers.sum())

        if count > best_inlier_count:
            best_inlier_count = count
            best_inlier_mask = inliers

    if best_inlier_mask is None or best_inlier_count < 3:
        raise RuntimeError("RANSAC failed to find a plane")

    inlier_points = points[best_inlier_mask]
    normal, point_on_plane = fit_plane_svd(inlier_points)

    # refine inliers using SVD plane
    dists = np.abs((points - point_on_plane) @ normal)
    inlier_mask = dists < float(distance_threshold)

    return normal, point_on_plane, inlier_mask


def ransac_extract_planes(
    points,
    distance_threshold=2.0,
    min_inliers=500,
    max_planes=1,
    max_iterations=1000,
):
    """
    Iteratively extract multiple planes (usually max_planes=1 for retractor).
    Returns:
      planes: list of {"normal": (3,), "point": (3,)}
      indices_per_plane: list of np.ndarray (indices into input points)
    """
    remaining_mask = np.ones(len(points), dtype=bool)
    planes = []
    indices_per_plane = []

    for _ in range(int(max_planes)):
        pts_remaining = points[remaining_mask]
        if len(pts_remaining) < int(min_inliers):
            break

        normal, point_on_plane, inlier_mask_local = ransac_plane(
            pts_remaining,
            distance_threshold=distance_threshold,
            max_iterations=max_iterations,
        )

        inlier_indices_global = np.where(remaining_mask)[0][inlier_mask_local]
        if len(inlier_indices_global) < int(min_inliers):
            break

        planes.append({"normal": normal, "point": point_on_plane})
        indices_per_plane.append(inlier_indices_global)

        remaining_mask[inlier_indices_global] = False

    return planes, indices_per_plane


def orient_plane_normals_up_from_center(planes, points):
    """
    Flip normals so the point cloud COM lies "below" the plane.
    """
    center = center_of_mass(points)
    for p in planes:
        n = p["normal"]
        p0 = p["point"]
        d = float(np.dot(n, center - p0))
        if d > 1e-6:
            p["normal"] = -n


def _principal_axis_in_plane(points, normal):
    """
    Estimate the dominant in-plane axis using PCA on the projected points.
    """
    if points.shape[0] < 2:
        return None

    n = normalize_vec(normal)
    centered = points - points.mean(axis=0)
    projected = centered - np.outer(centered @ n, n)
    _, _, vh = np.linalg.svd(projected, full_matrices=False)
    axis = vh[0]
    axis = axis - np.dot(axis, n) * n
    if np.linalg.norm(axis) < 1e-8:
        return None
    return normalize_vec(axis)


def build_retractor_frame_from_plane(points, indices, normal, point_on_plane):
    """
    Given a plane (normal, point) + inlier indices, define a stable retractor XY frame.

    Returns
    -------
    origin_world : (3,)
    axes_world : {"x":(3,), "y":(3,), "z":(3,)}
    T_world_to_retractor : (4,4)
    """
    # Ensure normal points "up" away from COM (robust)
    center = center_of_mass(points)
    if float(np.dot(normal, center - point_on_plane)) > 1e-6:
        normal = -normal

    # Build an orthonormal basis aligned with the retractor geometry
    pts = points[indices]
    principal_axis = _principal_axis_in_plane(pts, normal)
    if principal_axis is None:
        u, v, n = build_plane_basis(normal)
        x_axis = normalize_vec(u)
        y_axis = normalize_vec(v)
        z_axis = normalize_vec(n)
    else:
        z_axis = normalize_vec(normal)
        x_axis = principal_axis - np.dot(principal_axis, z_axis) * z_axis
        x_axis = normalize_vec(x_axis)
        y_axis = normalize_vec(np.cross(z_axis, x_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            u, v, n = build_plane_basis(normal)
            x_axis = normalize_vec(u)
            y_axis = normalize_vec(v)
            z_axis = normalize_vec(n)

    # Choose origin = centroid of inlier points projected onto plane basis (x,y)
    pts_rel = pts - point_on_plane
    uv = np.stack([pts_rel @ x_axis, pts_rel @ y_axis], axis=1)
    uv_center = uv.mean(axis=0)

    origin_world = point_on_plane + uv_center[0] * x_axis + uv_center[1] * y_axis

    axes_world = {"x": x_axis, "y": y_axis, "z": z_axis}

    # world -> retractor: rows are axes (matches your earlier convention)
    R = np.vstack([x_axis, y_axis, z_axis])  # (3,3)
    t = -R @ origin_world                    # (3,)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t

    return origin_world, axes_world, T


def retractor_segmentation_algorithm(
    points,
    *,
    distance_threshold=2.0,
    min_inliers=500,
    max_planes=1,
    max_iterations=1000,
    num_planes_to_combine=1,
):
    """
    Segment retractor plane(s) and define a retractor XY frame.

    IMPORTANT:
      - This runs on the point cloud you pass in (typically ACTIVE/preprocessed points).
      - The returned indices are indices into THAT input point array.

    Returns
    -------
    {
      "indices": np.ndarray,                 # indices into `points` input
      "base_plane": {"normal":(3,), "point":(3,)} or None
      "origin_world": (3,) or None
      "axes_world": {"x":(3,), "y":(3,), "z":(3,)} or None
      "T_world_to_retractor": (4,4) or None
    }
    """
    planes, indices_per_plane = ransac_extract_planes(
        points,
        distance_threshold=distance_threshold,
        min_inliers=min_inliers,
        max_planes=max_planes,
        max_iterations=max_iterations,
    )

    if planes:
        orient_plane_normals_up_from_center(planes, points)

    if not indices_per_plane:
        return {
            "indices": np.array([], dtype=int),
            "base_plane": None,
            "origin_world": None,
            "axes_world": None,
            "T_world_to_retractor": None,
        }

    k = min(int(num_planes_to_combine), len(indices_per_plane))
    combined_indices = np.unique(np.concatenate(indices_per_plane[:k])).astype(int)

    if combined_indices.size == 0:
        return {
            "indices": combined_indices,
            "base_plane": None,
            "origin_world": None,
            "axes_world": None,
            "T_world_to_retractor": None,
        }

    # Refine plane via SVD on combined inliers
    combined_points = points[combined_indices]
    normal, point_on_plane = fit_plane_svd(combined_points)

    base_plane = {"normal": normal, "point": point_on_plane}

    origin_world, axes_world, T = build_retractor_frame_from_plane(
        points=points,
        indices=combined_indices,
        normal=normal,
        point_on_plane=point_on_plane,
    )

    return {
        "indices": combined_indices,
        "base_plane": base_plane,
        "origin_world": origin_world,
        "axes_world": axes_world,
        "T_world_to_retractor": T,
    }
