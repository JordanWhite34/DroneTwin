"""Split the cleaned dense cloud into ground and non-ground points.

How it works:
- Loads `fused-keep-more-cleaned.ply` from the dense output folder.
- Estimates the median nearest-neighbor spacing so the RANSAC tolerance scales
  with the current reconstruction instead of using a fixed magic distance.
- Fits one dominant plane with Open3D RANSAC; in this aerial mapping workflow
  that plane is treated as the main ground surface.
- Writes inlier points to `ground.ply` and all remaining points to
  `non_ground.ply` for clustering, region growing, and classification.

Important caveat:
RANSAC finds the largest plane-like surface. If a scene has a wall, roof, or
other dominant flat surface larger than the terrain, visually inspect the split.
"""

from pathlib import Path
import argparse

import numpy as np
import open3d as o3d


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_DIR / "data" / "processed" / "dense" / "fused-keep-more-cleaned.ply"
GROUND_OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "dense" / "ground.ply"
NON_GROUND_OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "dense" / "non_ground.ply"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a cleaned point cloud into ground and non-ground points."
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Write outputs without opening the Open3D viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))

    if pcd.is_empty():
        raise ValueError(f"No points loaded from {INPUT_PATH}")

    print(f"Loaded {len(pcd.points):,} points")

    # Estimate point spacing so our RANSAC threshold is not totally arbitrary.
    nn_distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    median_spacing = float(np.median(nn_distances))

    print(f"Median nearest-neighbor spacing: {median_spacing:.6f}")

    # RANSAC plane segmentation. The threshold is intentionally derived from
    # point spacing because COLMAP scene scale can vary across reconstructions.
    distance_threshold = median_spacing * 8.0

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )

    a, b, c, d = plane_model
    print(f"Plane model: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"Ground/inlier points: {len(inliers):,}")

    ground_cloud = pcd.select_by_index(inliers)
    non_ground_cloud = pcd.select_by_index(inliers, invert=True)

    print(f"Non-ground/outlier points: {len(non_ground_cloud.points):,}")

    # Color for viewing
    ground_cloud.paint_uniform_color([0.2, 0.8, 0.2])      # green-ish
    non_ground_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # gray

    if not args.no_view:
        o3d.visualization.draw_geometries(
            [ground_cloud, non_ground_cloud],
            window_name="RANSAC segmentation: ground vs non-ground",
        )

    GROUND_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(str(GROUND_OUTPUT_PATH), ground_cloud)
    o3d.io.write_point_cloud(str(NON_GROUND_OUTPUT_PATH), non_ground_cloud)

    print(f"Wrote ground cloud to: {GROUND_OUTPUT_PATH}")
    print(f"Wrote non-ground cloud to: {NON_GROUND_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
