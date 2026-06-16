"""Region-grow non-ground points into surface patches.

This script is intentionally small: it builds a voxel proxy, connects nearby
voxels when their normals and heights are compatible, then transfers region
labels back to the full-resolution cloud.
"""

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROJECT_DIR = Path(__file__).resolve().parents[1]
DENSE_DIR = PROJECT_DIR / "data" / "processed" / "dense"
DEFAULT_INPUT = DENSE_DIR / "non_ground.ply"
DEFAULT_GROUND = DENSE_DIR / "ground.ply"
DEFAULT_OUTPUT = DENSE_DIR / "non_ground_regions.ply"
DEFAULT_PROXY_OUTPUT = DENSE_DIR / "non_ground_regions_proxy.ply"
DEFAULT_LABELS = DENSE_DIR / "non_ground_region_labels.npz"


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"No points loaded from {path}")
    return cloud


def bbox_diagonal(cloud: o3d.geometry.PointCloud) -> float:
    return float(np.linalg.norm(cloud.get_axis_aligned_bounding_box().get_extent()))


def fit_plane(path: Path) -> np.ndarray:
    points = np.asarray(load_cloud(path).points)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1] / max(float(np.linalg.norm(vh[-1])), 1e-12)
    return np.array([*normal, -float(normal @ centroid)])


def signed_height(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return points @ plane[:3] + plane[3]


def orient_plane(plane: np.ndarray, points: np.ndarray) -> np.ndarray:
    return -plane if float(np.median(signed_height(points, plane))) < 0 else plane


def estimate_normals(cloud: o3d.geometry.PointCloud, radius: float, max_nn: int) -> np.ndarray:
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    normals = np.asarray(cloud.normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def grow_regions(
    points: np.ndarray,
    normals: np.ndarray,
    heights: np.ndarray,
    radius: float,
    max_angle_deg: float,
    max_height_jump: float,
    min_region_points: int,
) -> np.ndarray:
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=radius)
    min_dot = float(np.cos(np.deg2rad(max_angle_deg)))
    labels = np.full(len(points), -2, dtype=int)
    next_label = 0

    for seed in range(len(points)):
        if labels[seed] != -2:
            continue

        labels[seed] = next_label
        queue = deque([seed])
        members = []

        while queue:
            current = queue.popleft()
            members.append(current)

            for neighbor in neighbors[current]:
                if labels[neighbor] != -2:
                    continue
                normal_ok = abs(float(normals[current] @ normals[neighbor])) >= min_dot
                height_ok = abs(float(heights[current] - heights[neighbor])) <= max_height_jump
                if normal_ok and height_ok:
                    labels[neighbor] = next_label
                    queue.append(neighbor)

        if len(members) < min_region_points:
            labels[members] = -1
        else:
            next_label += 1

    labels[labels == -2] = -1
    return labels


def transfer_labels(proxy_points: np.ndarray, proxy_labels: np.ndarray, points: np.ndarray) -> np.ndarray:
    _, nearest = cKDTree(proxy_points).query(points, k=1)
    return proxy_labels[nearest]


def colors_for_labels(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(labels), 3))
    cmap = plt.get_cmap("tab20")
    for label in np.unique(labels[labels >= 0]):
        colors[labels == label] = cmap(int(label) % 20)[:3]
    return colors


def print_summary(points: np.ndarray, labels: np.ndarray, limit: int) -> None:
    region_labels = labels[labels >= 0]
    print(f"Detected {len(np.unique(region_labels))} regions")
    print(f"Noise points: {int(np.sum(labels == -1)):,}")

    sizes = [(int(label), int(np.sum(labels == label))) for label in np.unique(region_labels)]
    for label, count in sorted(sizes, key=lambda item: item[1], reverse=True)[:limit]:
        extent = np.ptp(points[labels == label], axis=0)
        print(
            f"  region {label:>4}: {count:>8,} points | "
            f"extent=({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grow non-ground surface regions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ground", type=Path, default=DEFAULT_GROUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--proxy-output", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--voxel-fraction", type=float, default=0.005)
    parser.add_argument("--neighbor-radius-multiplier", type=float, default=1.6)
    parser.add_argument("--normal-radius-multiplier", type=float, default=3.0)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--max-normal-angle-deg", type=float, default=22.0)
    parser.add_argument("--max-height-jump", type=float, default=0.15)
    parser.add_argument("--min-region-points", type=int, default=10)
    parser.add_argument("--summary-count", type=int, default=12)
    parser.add_argument("--view", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.proxy_output and str(args.proxy_output).lower() == "default":
        args.proxy_output = DEFAULT_PROXY_OUTPUT

    cloud = load_cloud(args.input)
    points = np.asarray(cloud.points)
    voxel_size = args.voxel_size or bbox_diagonal(cloud) * args.voxel_fraction

    proxy = cloud.voxel_down_sample(voxel_size)
    proxy_points = np.asarray(proxy.points)
    plane = orient_plane(fit_plane(args.ground), proxy_points)
    heights = signed_height(proxy_points, plane)
    normals = estimate_normals(
        proxy,
        radius=voxel_size * args.normal_radius_multiplier,
        max_nn=args.normal_max_nn,
    )

    print(
        f"Region proxy voxel size: {voxel_size:.6f} | "
        f"{len(points):,} full points -> {len(proxy_points):,} proxy points"
    )

    proxy_labels = grow_regions(
        points=proxy_points,
        normals=normals,
        heights=heights,
        radius=voxel_size * args.neighbor_radius_multiplier,
        max_angle_deg=args.max_normal_angle_deg,
        max_height_jump=args.max_height_jump,
        min_region_points=args.min_region_points,
    )
    full_labels = transfer_labels(proxy_points, proxy_labels, points)

    print_summary(points, full_labels, args.summary_count)

    cloud.colors = o3d.utility.Vector3dVector(colors_for_labels(full_labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.output), cloud)
    print(f"Wrote full-resolution regions to: {args.output}")

    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.labels_output,
        proxy_points=proxy_points,
        proxy_labels=proxy_labels,
        full_labels=full_labels,
        voxel_size=np.array([voxel_size]),
        ground_plane=plane,
    )
    print(f"Wrote region labels to: {args.labels_output}")

    if args.proxy_output:
        proxy.colors = o3d.utility.Vector3dVector(colors_for_labels(proxy_labels))
        args.proxy_output.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.proxy_output), proxy)
        print(f"Wrote proxy regions to: {args.proxy_output}")

    if args.view:
        o3d.visualization.draw_geometries([cloud], window_name="Non-ground regions")


if __name__ == "__main__":
    main()
