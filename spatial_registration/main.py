from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from .icp import PointToPlaneICP
from .plane_extraction import combined_mask, detect_rectangles
from .point_cloud import PointCloud, apply_mask, estimate_normals, load_point_cloud, save_point_cloud
from .visualization import plot_alignment, plot_mask, plot_rectangles


def extract_retractors(cloud: PointCloud, distance_threshold: float):
    rectangles = detect_rectangles(cloud, distance_threshold=distance_threshold)
    mask = combined_mask(cloud, rectangles, distance_threshold=distance_threshold)
    return rectangles, mask, apply_mask(cloud, mask)


def run_pipeline(
    target_path: Path,
    source_path: Path,
    distance_threshold: float,
    icp_iterations: int,
    correspondence_threshold: float,
    normal_neighbors: int,
    output_dir: Path,
) -> None:
    target_cloud = load_point_cloud(target_path)
    source_cloud = load_point_cloud(source_path)

    target_rectangles, target_mask, target_masked = extract_retractors(target_cloud, distance_threshold=distance_threshold)
    source_rectangles, source_mask, source_masked = extract_retractors(source_cloud, distance_threshold=distance_threshold)

    target_with_normals = estimate_normals(target_masked, k_neighbors=normal_neighbors)

    icp = PointToPlaneICP(max_iterations=icp_iterations, correspondence_threshold=correspondence_threshold)
    result, aligned = icp.register(source_masked, target_with_normals)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_point_cloud(output_dir / "target_masked.npy", target_masked)
    save_point_cloud(output_dir / "target_masked.ply", target_masked)
    save_point_cloud(output_dir / "source_masked.npy", source_masked)
    save_point_cloud(output_dir / "source_masked.ply", source_masked)
    save_point_cloud(output_dir / "aligned_source.npy", aligned)
    save_point_cloud(output_dir / "aligned_source.ply", aligned)

    np.save(output_dir / "target_mask.npy", target_mask.astype(bool))
    np.save(output_dir / "source_mask.npy", source_mask.astype(bool))

    (output_dir / "target_rectangles.json").write_text(json.dumps([r.as_dict() for r in target_rectangles], indent=2))
    (output_dir / "source_rectangles.json").write_text(json.dumps([r.as_dict() for r in source_rectangles], indent=2))

    plot_mask(target_cloud, target_mask, output_dir / "target_mask.png", title="Target mask")
    plot_mask(source_cloud, source_mask, output_dir / "source_mask.png", title="Source mask")
    plot_rectangles(target_cloud, target_rectangles, output_dir / "target_rectangles.png", title="Target rectangle fits")
    plot_rectangles(source_cloud, source_rectangles, output_dir / "source_rectangles.png", title="Source rectangle fits")
    plot_alignment(target_with_normals, aligned, output_dir / "aligned_overlay.png")

    print("ICP converged:" , result.converged)
    print("Iterations:", result.iterations)
    print("Rotation:\n", result.rotation)
    print("Translation:", result.translation)


def main(
    target_path: Path,
    source_path: Path,
    *,
    distance_threshold: float = 1e-2,
    icp_iterations: int = 60,
    correspondence_threshold: float = 0.05,
    normal_neighbors: int = 30,
    output_dir: Path = Path("data/outputs"),
) -> None:
    """Run the registration pipeline with explicit parameters.

    Supply the target/source paths and optionally adjust thresholds/iteration counts.
    This keeps the module script-friendly while still allowing programmatic use.
    """

    run_pipeline(
        target_path=target_path,
        source_path=source_path,
        distance_threshold=distance_threshold,
        icp_iterations=icp_iterations,
        correspondence_threshold=correspondence_threshold,
        normal_neighbors=normal_neighbors,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main(
        target_path=Path("data/sample_target.ply"),
        source_path=Path("data/sample_source.ply"),
    )
