"""Lightweight matplotlib visualizations for point cloud processing."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .plane_extraction import Rectangle, rectangle_corners
from .point_cloud import PointCloud


_DEF_FIGSIZE = (8, 6)


def _new_axes():
    fig = plt.figure(figsize=_DEF_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def _set_equal_aspect(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    max_range = (points.max(axis=0) - points.min(axis=0)).max()
    mid = points.mean(axis=0)
    for axis, coordinate in zip("xyz", mid):
        getattr(ax, f"set_{axis}lim")(coordinate - max_range / 2, coordinate + max_range / 2)


def _finalize(ax, title: str | None, output_path: Path) -> None:
    if title:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path)
    plt.close(ax.figure)


def plot_mask(cloud: PointCloud, mask: Iterable[bool], output_path: Path, title: str | None = None) -> None:
    points = cloud.points
    mask_array = np.asarray(mask, dtype=bool)
    fig, ax = _new_axes()
    ax.scatter(points[~mask_array, 0], points[~mask_array, 1], points[~mask_array, 2], s=1, c="lightgray", alpha=0.4, label="background")
    ax.scatter(points[mask_array, 0], points[mask_array, 1], points[mask_array, 2], s=2, c="tab:blue", label="mask")
    ax.legend(loc="upper right")
    _set_equal_aspect(ax, points)
    _finalize(ax, title, Path(output_path))


def plot_rectangles(cloud: PointCloud, rectangles: Sequence[Rectangle], output_path: Path, title: str | None = None) -> None:
    points = cloud.points
    fig, ax = _new_axes()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="lightgray", alpha=0.3)

    for idx, rectangle in enumerate(rectangles):
        corners = rectangle_corners(rectangle)
        # close loop for plotting
        loop = np.vstack([corners, corners[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], lw=2, label=f"Rectangle {idx+1}")

        poly = Poly3DCollection([corners], alpha=0.1)
        poly.set_facecolor("C" + str(idx % 10))
        ax.add_collection3d(poly)

    if rectangles:
        centers = np.stack([r.center for r in rectangles])
        _set_equal_aspect(ax, np.vstack([points, centers]))
    else:
        _set_equal_aspect(ax, points)

    ax.legend(loc="upper right")
    _finalize(ax, title, Path(output_path))


def plot_alignment(target: PointCloud, aligned_source: PointCloud, output_path: Path, title: str = "Aligned point clouds") -> None:
    fig, ax = _new_axes()
    ax.scatter(target.points[:, 0], target.points[:, 1], target.points[:, 2], s=1, c="tab:gray", alpha=0.5, label="target")
    ax.scatter(aligned_source.points[:, 0], aligned_source.points[:, 1], aligned_source.points[:, 2], s=1, c="tab:red", alpha=0.6, label="aligned source")
    _set_equal_aspect(ax, np.vstack([target.points, aligned_source.points]))
    ax.legend(loc="upper right")
    _finalize(ax, title, Path(output_path))
