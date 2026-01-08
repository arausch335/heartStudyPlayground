import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import cv2
from typing import List, Dict, Tuple, Optional

from registration.utilities.utilities import build_plane_basis


# -----------------------------
# Shared helpers
# -----------------------------

def _new_3d_ax(figsize=(7, 7), title: str = "") -> Axes3D:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=30, azim=-60)
    return ax


# -----------------------------
# 1. Raw 3D point cloud / frame
# -----------------------------

def plot_point_cloud_3d(points: np.ndarray,
                        color: str = "k",
                        size: float = 1.0,
                        title: str = "Point Cloud",
                        ax: Optional[Axes3D] = None) -> Axes3D:
    """
    Basic 3D scatter of a point cloud.

    points: (N, 3)
    """
    if ax is None:
        ax = _new_3d_ax(title=title)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=size, c=color, alpha=0.7)
    plt.tight_layout()
    return ax


def plot_frame_with_segments_3d(frame,
                                show_retractor: bool = True,
                                show_epicardium: bool = False,
                                sample_every: int = 1,
                                stage: str = "processed",
                                title: str = "Frame (3D)") -> None:
    """
    Visualize a frame's point cloud and optionally highlight retractor/epicardium.
    """
    pts = frame.get_points(stage)[::sample_every]

    ax = _new_3d_ax(title=title)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=0.5, c="lightgray", alpha=0.5, label="All points")

    if show_retractor and getattr(frame, "retractor", None) is not None:
        idx = frame.retractor.subset_indices
        stage_idx = frame.map_raw_indices_to_stage(idx, stage)
        rpts = frame.get_points(stage)[stage_idx]
        ax.scatter(rpts[:, 0], rpts[:, 1], rpts[:, 2],
                   s=2.0, c="red", alpha=0.9, label="Retractor")

    if show_epicardium and getattr(frame, "epicardium", None) is not None:
        idx = frame.epicardium.subset_indices
        stage_idx = frame.map_raw_indices_to_stage(idx, stage)
        epts = frame.get_points(stage)[stage_idx]
        ax.scatter(epts[:, 0], epts[:, 1], epts[:, 2],
                   s=1.5, c="blue", alpha=0.9, label="Epicardium")

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 2. Overhead projection & rectangles
# -----------------------------

def show_overhead_image(image_color: np.ndarray,
                        rectangles: Optional[List[np.ndarray]] = None,
                        title: str = "Overhead Projection") -> None:
    """
    Display overhead image with optional rectangle overlays.

    image_color: (H, W, 3) uint8
    rectangles: list of (4, 2) arrays of pixel coordinates (col, row)
    """
    img = image_color.copy()
    if rectangles is not None:
        for rect in rectangles:
            pts = rect.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 3. Points grouped by rectangles (3D)
# -----------------------------

def plot_rectangles_point_sets_3d(points: np.ndarray,
                                  rectangles: List[np.ndarray],
                                  pixel_to_indices: Dict[Tuple[int, int], List[int]],
                                  bbox: Tuple[float, float, float, float],
                                  resolution_mm: float = 1.0,
                                  title: str = "Rectangle Point Sets (3D)") -> None:
    """
    Visualize, in 3D, which points belong to each detected rectangle.
    """
    ax = _new_3d_ax(title=title)

    # draw all points faintly
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=0.5, c="lightgray", alpha=0.3)

    colors = plt.cm.get_cmap("tab10", len(rectangles))

    for i, rect in enumerate(rectangles):
        rect = rect.astype(int)
        cols = rect[:, 0]
        rows = rect[:, 1]
        cmin, cmax = cols.min(), cols.max()
        rmin, rmax = rows.min(), rows.max()

        indices = set()
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if (r, c) in pixel_to_indices:
                    indices.update(pixel_to_indices[(r, c)])

        if not indices:
            continue

        idx = np.array(list(indices), dtype=int)
        pts_rect = points[idx]

        ax.scatter(pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2],
                   s=4.0, c=[colors(i)], alpha=0.9,
                   label=f"Rect {i} ({len(idx)} pts)")

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 4. Planes fitted to points (3D)
# -----------------------------

def plot_planes_with_points(points: np.ndarray,
                            plane_defs: List[dict],
                            per_plane_indices: List[np.ndarray],
                            title: str = "Retractor Planes (3D)") -> None:
    """
    Visualize fitted planes and their inlier points.

    plane_defs: list of dicts like {"normal": (3,), "point": (3,)}
    per_plane_indices: list of (K_i,) index arrays, each corresponding to a plane
    """
    ax = _new_3d_ax(title=title)

    # all points faint
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=0.5, c="lightgray", alpha=0.2)

    colors = plt.cm.get_cmap("tab10", len(plane_defs))

    for i, (plane, idx) in enumerate(zip(plane_defs, per_plane_indices)):
        pts_plane = points[idx]
        ax.scatter(pts_plane[:, 0], pts_plane[:, 1], pts_plane[:, 2],
                   s=3.0, c=[colors(i)], alpha=0.9,
                   label=f"Plane {i} ({len(idx)} pts)")

        # draw a small plane patch using the normal + point
        n = np.asarray(plane["normal"])
        p0 = np.asarray(plane["point"])
        u, v, _ = build_plane_basis(n)

        size = 10.0
        uu = np.linspace(-size, size, 5)
        vv = np.linspace(-size, size, 5)
        U, V = np.meshgrid(uu, vv)
        plane_pts = p0[None, None, :] + U[..., None] * u[None, None, :] + V[..., None] * v[None, None, :]

        ax.plot_surface(plane_pts[..., 0], plane_pts[..., 1], plane_pts[..., 2],
                        color=colors(i), alpha=0.2, edgecolor="none")

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 5. Retractor-specific views
# -----------------------------

def plot_left_right_strips(points: np.ndarray,
                           left_indices: np.ndarray,
                           right_indices: np.ndarray,
                           title: str = "Retractor – left/right blades"):
    """
    Show whole cloud faintly, with left/right blade points highlighted.
    """
    ax = _new_3d_ax(title=title)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=0.2, c="lightgray", alpha=0.05, label="All points")

    if len(left_indices) > 0:
        pts_left = points[left_indices]
        ax.scatter(pts_left[:, 0], pts_left[:, 1], pts_left[:, 2],
                   s=3.0, c="red", alpha=0.9, label=f"Left ({len(left_indices)} pts)")

    if len(right_indices) > 0:
        pts_right = points[right_indices]
        ax.scatter(pts_right[:, 0], pts_right[:, 1], pts_right[:, 2],
                   s=3.0, c="blue", alpha=0.9, label=f"Right ({len(right_indices)} pts)")

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_retractor_plane_frame(points: np.ndarray,
                               R_world_to_plane: np.ndarray,
                               origin_world: np.ndarray,
                               left_indices: np.ndarray,
                               right_indices: np.ndarray,
                               blade_boxes: Dict[str, Dict],
                               title: str = "Retractor – plane frame (XY = retractor)"):
    """
    Visualize retractor in its own coordinate system:
      - XY = retractor plane
      - origin = center of combined blades
      - rectangles for left/right boxes.
    """
    pts_plane = (R_world_to_plane @ (points - origin_world).T).T  # (N,3)
    x_all = pts_plane[:, 0]
    y_all = pts_plane[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_all, y_all, s=0.2, c="lightgray", alpha=0.1, label="All points")

    if len(left_indices) > 0:
        pl = pts_plane[left_indices]
        ax.scatter(pl[:, 0], pl[:, 1], s=2.0, c="red", alpha=0.9, label="Left blade")

    if len(right_indices) > 0:
        pr = pts_plane[right_indices]
        ax.scatter(pr[:, 0], pr[:, 1], s=2.0, c="blue", alpha=0.9, label="Right blade")

    # Draw boxes in (a,b) coords (plane XY axes)
    for label, color in [("left", "red"), ("right", "blue")]:
        box = blade_boxes.get(label)
        if not box:
            continue
        a_min = box["a_min"]
        a_max = box["a_max"]
        b_min = box["b_min"]
        b_max = box["b_max"]
        xs = [a_min, a_max, a_max, a_min, a_min]
        ys = [b_min, b_min, b_max, b_max, b_min]
        ax.plot(xs, ys, color=color, linewidth=2)

    ax.set_xlabel("X (along retractor)")
    ax.set_ylabel("Y (across blades)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. Registration
# -----------------------------


def plot_two_registered_retractors(frame_a, frame_b, sample=1):
    """
    Show two registered point clouds in the OR coordinate system.
    Useful for debugging alignment quality.
    """

    idx0 = frame_a.retractor.subset_indices
    idx1 = frame_b.retractor.subset_indices

    pts_a = frame_a.get_points("registered")
    pts_b = frame_b.get_points("registered")
    idx0_stage = frame_a.map_raw_indices_to_stage(idx0, "registered")
    idx1_stage = frame_b.map_raw_indices_to_stage(idx1, "registered")
    pts_a = pts_a[idx0_stage][::sample]
    pts_b = pts_b[idx1_stage][::sample]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
               s=1.0, c="red", alpha=0.6, label=f"Frame {frame_a.n}")

    ax.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
               s=1.0, c="blue", alpha=0.6, label=f"Frame {frame_b.n}")

    ax.set_title("Two Registered Point Clouds")
    ax.set_xlabel("X (OR)")
    ax.set_ylabel("Y (OR)")
    ax.set_zlabel("Z (OR)")
    ax.legend()
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.show()
