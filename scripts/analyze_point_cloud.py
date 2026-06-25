"""Inspect a cleaned point cloud and print basic scene measurements.

How it works:
- Loads the cleaned dense PLY produced by `clean_point_cloud.py`.
- Uses the axis-aligned bounding box for rough width/depth/height estimates.
- Uses Z-coordinate statistics for elevation summaries.
- Uses nearest-neighbor distances to estimate point spacing and reconstruction
  density; that spacing also informs later cleanup and segmentation thresholds.
- Colors points by normalized height so elevation structure is easy to inspect.
"""

from pathlib import Path
import argparse

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

POINT_CLOUD_PATH = Path(r"data\processed\dense\fused-keep-more-cleaned.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect point-cloud metadata, spacing, and height coloring."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=POINT_CLOUD_PATH,
        help=f"Input point cloud. Default: {POINT_CLOUD_PATH}",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Print statistics without opening Matplotlib/Open3D windows.",
    )
    return parser.parse_args()

def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise ValueError(f"No points loaded from {path}")
    
    return pcd


def print_basic_metadata(pcd: o3d.geometry.PointCloud) -> None:
    points = np.asarray(pcd.points)

    print("Point count:", len(points))
    print("Has colors:", pcd.has_colors())
    print("Has normals:", pcd.has_normals())

    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    extent = aabb.get_extent()
    diagonal = np.linalg.norm(extent)

    print("\nAxis-aligned bounding box:")
    print("Min bound:", min_bound)
    print("Max bound:", max_bound)
    print("Extent:", extent)
    print("Diagonal:", diagonal)

    x_size, y_size, z_size = extent
    print("\nApproximate spatial dimensions:")
    print(f"Width X:  {x_size:.3f}")
    print(f"Depth Y:  {y_size:.3f}")
    print(f"Height Z: {z_size:.3f}")

    footprint_area = x_size * y_size
    print(f"Approximate footprint area: {footprint_area:.3f} square scene-units")


def analyze_elevation(pcd: o3d.geometry.PointCloud, z_axis: int = 2) -> None:
    points = np.asarray(pcd.points)
    z = points[:, z_axis]

    print("\nElevation / height statistics:")
    print(f"Min:    {np.min(z):.3f}")
    print(f"Max:    {np.max(z):.3f}")
    print(f"Mean:   {np.mean(z):.3f}")
    print(f"Median: {np.median(z):.3f}")
    print(f"Std:    {np.std(z):.3f}")

    low_idx = np.argmin(z)
    high_idx = np.argmax(z)

    print("\nLowest point:", points[low_idx])
    print("Highest point:", points[high_idx])


def analyze_point_spacing(
    pcd: o3d.geometry.PointCloud,
    show_plot: bool = True,
) -> np.ndarray:
    # nearest-neighbor distance analysis
    nn_distances = np.asarray(pcd.compute_nearest_neighbor_distance())

    print("\nNearest-neighbor distance statistics:")
    print(f"Min:    {np.min(nn_distances):.6f}")
    print(f"Max:    {np.max(nn_distances):.6f}")
    print(f"Mean:   {np.mean(nn_distances):.6f}")
    print(f"Median: {np.median(nn_distances):.6f}")
    print(f"Std:    {np.std(nn_distances):.6f}")

    if show_plot:
        plt.hist(nn_distances, bins=100)
        plt.title("Nearest-neighbor distance distribution")
        plt.xlabel("Distance")
        plt.ylabel("Point count")
        plt.show()

    return nn_distances


def color_by_height(pcd: o3d.geometry.PointCloud, z_axis: int = 2) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    z = points[:, z_axis]

    z_min = np.min(z)
    z_max = np.max(z)
    z_normalized = (z - z_min) / max(z_max - z_min, 1e-12)

    colors = plt.cm.viridis(z_normalized)[:, :3]

    colored = o3d.geometry.PointCloud(pcd)
    colored.colors = o3d.utility.Vector3dVector(colors)

    return colored


def main() -> None:
    args = parse_args()
    pcd = load_point_cloud(args.input)

    # Optional: normalize cloud so its minimum bound starts near origin.
    # The book does this before geometric operations in its clustering/spatial setup.
    translation = pcd.get_min_bound()
    pcd.translate(-translation)

    print_basic_metadata(pcd)
    analyze_elevation(pcd)
    analyze_point_spacing(pcd, show_plot=not args.no_view)

    colored_by_height = color_by_height(pcd)

    if not args.no_view:
        o3d.visualization.draw_geometries(
            [colored_by_height],
            window_name="Point Cloud Colored by Height",
        )


if __name__ == "__main__":
    main()
