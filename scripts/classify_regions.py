"""Classify region-grown patches into coarse scene classes.

The model is deliberately transparent: each region gets simple geometry scores
for ground, tree, building, water, and other. Dominance arguments act like
future UI sliders by biasing those scores without changing the feature code.

How it works:
- Reads the non-ground cloud plus region labels from `region_grow_non_ground.py`.
- Computes region features: point count, area estimate, height range/variation,
  roughness, planarity, scattering, horizontalness, footprint fill, aspect ratio,
  and squareness.
- Converts those features into rule-based class scores. For example, tree-like
  regions tend to be rough/tall, building-like regions tend to be planar and
  filled, and water-like regions must be flat/smooth when enabled.
- Multiplies each score by its class dominance value, then assigns the highest
  scoring class with deterministic tie-breaking.
- Writes colored semantic PLY files plus CSV reports so the decision can be
  inspected rather than treated as a black-box ML model.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROJECT_DIR = Path(__file__).resolve().parents[1]
DENSE_DIR = PROJECT_DIR / "data" / "processed" / "dense"
DEFAULT_NON_GROUND = DENSE_DIR / "non_ground.ply"
DEFAULT_GROUND = DENSE_DIR / "ground.ply"
DEFAULT_LABELS = DENSE_DIR / "non_ground_region_labels.npz"
DEFAULT_OUTPUT = DENSE_DIR / "semantic_classes.ply"
DEFAULT_NON_GROUND_OUTPUT = DENSE_DIR / "non_ground_semantic_classes.ply"
DEFAULT_REPORT = DENSE_DIR / "region_class_report.csv"
DEFAULT_LEGEND = DENSE_DIR / "semantic_class_legend.csv"

COLORS = {
    "noise": [0.0, 0.0, 0.0],
    "ground": [0.45, 0.38, 0.25],
    "tree": [0.05, 0.45, 0.12],
    "water": [0.05, 0.35, 0.95],
    "building": [0.72, 0.72, 0.72],
    "other": [0.95, 0.55, 0.12],
}


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"No points loaded from {path}")
    return cloud


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_min(value: float, minimum: float, span: float) -> float:
    return clamp01((value - minimum) / max(span, 1e-12))


def score_max(value: float, maximum: float, span: float) -> float:
    return clamp01((maximum - value) / max(span, 1e-12))


def signed_height(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return points @ plane[:3] + plane[3]


def orient_plane(plane: np.ndarray, points: np.ndarray) -> np.ndarray:
    return -plane if float(np.median(signed_height(points, plane))) < 0 else plane


def covariance_shape(points: np.ndarray, ground_normal: np.ndarray) -> dict[str, float]:
    if len(points) < 3:
        return {"roughness": 1.0, "planarity": 0.0, "scattering": 1.0, "horizontalness": 0.0}

    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(len(points) - 1, 1))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    e0, e1, e2 = eigenvalues
    normal = eigenvectors[:, 0] / max(float(np.linalg.norm(eigenvectors[:, 0])), 1e-12)
    scale = max(float(e2), 1e-12)
    return {
        "roughness": float(e0 / max(float(eigenvalues.sum()), 1e-12)),
        "planarity": float((e1 - e0) / scale),
        "scattering": float(e0 / scale),
        "horizontalness": abs(float(normal @ ground_normal)),
    }


def footprint(points: np.ndarray, voxel_size: float) -> dict[str, float]:
    xy = points[:, :2]
    if len(xy) < 3:
        return {"area": 0.0, "fill": 0.0, "aspect": 999.0, "squareness": 0.0}

    centered = xy - xy.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered / max(len(xy) - 1, 1))
    extents = np.ptp(centered @ axes, axis=0)
    width = float(max(extents))
    depth = float(min(extents))
    box_area = max(width * depth, 1e-12)
    area = float(len(points) * voxel_size * voxel_size)
    return {
        "area": area,
        "fill": min(area / box_area, 1.0),
        "aspect": width / max(depth, 1e-12),
        "squareness": min(depth / max(width, 1e-12), 1.0),
    }


def region_features(
    label: int,
    points: np.ndarray,
    heights: np.ndarray,
    voxel_size: float,
    ground_normal: np.ndarray,
) -> dict[str, float | int]:
    extents = np.ptp(points, axis=0)
    features: dict[str, float | int] = {
        "region_id": label,
        "point_count": int(len(points)),
        "height_mean": float(heights.mean()),
        "height_std": float(heights.std()),
        "extent_z": float(extents[2]),
        "centroid_x": float(points[:, 0].mean()),
        "centroid_y": float(points[:, 1].mean()),
        "centroid_z": float(points[:, 2].mean()),
    }
    features.update(covariance_shape(points, ground_normal))
    features.update(footprint(points, voxel_size))
    return features


def class_scores(features: dict[str, float | int], args: argparse.Namespace) -> dict[str, float]:
    count = int(features["point_count"])
    area = float(features["area"])
    rough = float(features["roughness"])
    scatter = float(features["scattering"])
    planar = float(features["planarity"])
    horiz = float(features["horizontalness"])
    fill = float(features["fill"])
    aspect = float(features["aspect"])
    square = float(features["squareness"])
    hstd = float(features["height_std"])
    ez = float(features["extent_z"])

    ground = np.mean([
        score_max(rough, args.ground_max_roughness, args.ground_max_roughness),
        score_max(scatter, args.ground_max_scattering, args.ground_max_scattering),
        score_min(fill, args.ground_min_fill, 1.0 - args.ground_min_fill),
        score_min(area, args.ground_seed_area * 0.25, args.ground_seed_area),
    ])

    tree = 0.0
    if count >= args.tree_min_points and area <= args.tree_max_area:
        tree = np.mean([
            score_min(rough, args.tree_min_roughness, args.tree_min_roughness * 3.0),
            score_min(scatter, args.tree_min_scattering, args.tree_min_scattering * 3.0),
            score_min(hstd, args.tree_min_height_std, args.tree_min_height_std * 3.0),
            score_min(ez, args.tree_min_extent_z, args.tree_min_extent_z * 2.0),
        ])

    building = 0.0
    if count >= args.building_min_points and area <= args.building_max_area:
        building = np.mean([
            score_max(rough, args.building_max_roughness, args.building_max_roughness),
            score_max(hstd, args.building_max_height_std, args.building_max_height_std),
            score_min(planar, args.building_min_planarity, 1.0 - args.building_min_planarity),
            score_min(horiz, args.building_min_horizontalness, 1.0 - args.building_min_horizontalness),
            score_min(fill, args.building_min_fill, 1.0 - args.building_min_fill),
            score_max(aspect, args.building_max_aspect, args.building_max_aspect),
            score_min(square, args.building_min_squareness, 1.0 - args.building_min_squareness),
        ])

    water = 0.0
    if args.enable_water and count >= args.water_min_points and area >= args.water_min_area:
        water = np.mean([
            score_max(rough, args.water_max_roughness, args.water_max_roughness),
            score_max(hstd, args.water_max_height_std, args.water_max_height_std),
            score_max(ez, args.water_max_extent_z, args.water_max_extent_z),
            score_min(horiz, args.water_min_horizontalness, 1.0 - args.water_min_horizontalness),
            score_min(fill, args.water_min_fill, 1.0 - args.water_min_fill),
        ])

    return {
        "ground": float(ground * args.ground_dominance),
        "tree": float(tree * args.tree_dominance),
        "building": float(building * args.building_dominance),
        "water": float(water * args.water_dominance),
        "other": float(args.other_score * args.other_dominance),
    }


def winner(scores: dict[str, float]) -> str:
    order = ("ground", "tree", "building", "water", "other")
    return max(order, key=lambda name: (scores[name], -order.index(name)))


def region_adjacency(points: np.ndarray, labels: np.ndarray, radius: float) -> dict[int, set[int]]:
    adjacency = {int(label): set() for label in np.unique(labels[labels >= 0])}
    neighborhoods = cKDTree(points).query_ball_point(points, r=radius)
    for index, neighbors in enumerate(neighborhoods):
        a = int(labels[index])
        if a < 0:
            continue
        for neighbor in neighbors:
            b = int(labels[neighbor])
            if b >= 0 and b != a:
                adjacency[a].add(b)
                adjacency[b].add(a)
    return adjacency


def infer_ground(rows: list[dict[str, float | int | str]], points: np.ndarray, labels: np.ndarray, voxel_size: float, args: argparse.Namespace) -> set[int]:
    if args.no_terrain_ground:
        return set()

    by_label = {int(row["region_id"]): row for row in rows}
    seeds = {
        label for label, row in by_label.items()
        if float(row["ground_score"]) >= args.ground_seed_score
        and float(row["area"]) >= args.ground_seed_area
    }
    if not seeds:
        return set()

    ground = set(seeds)
    adjacency = region_adjacency(points, labels, voxel_size * args.ground_neighbor_radius)
    queue = list(seeds)
    while queue:
        current = queue.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor in ground:
                continue
            row = by_label[neighbor]
            close_height = abs(float(row["height_mean"]) - float(by_label[current]["height_mean"])) <= args.ground_height_gap
            good_score = float(row["ground_score"]) >= args.ground_min_score
            if close_height and good_score:
                ground.add(neighbor)
                queue.append(neighbor)

    print(f"Terrain ground inference: {len(seeds)} seeds -> {len(ground)} regions")
    return ground


def absorb_ground_noise(class_names: np.ndarray, full_points: np.ndarray, proxy_points: np.ndarray, proxy_labels: np.ndarray, ground_labels: set[int], voxel_size: float, args: argparse.Namespace) -> int:
    if args.no_ground_noise_absorption or not ground_labels:
        return 0

    ground_points = proxy_points[np.isin(proxy_labels, list(ground_labels))]
    noise_indices = np.flatnonzero(class_names == "noise")
    if len(ground_points) == 0 or len(noise_indices) == 0:
        return 0

    noise_points = full_points[noise_indices]
    xy_dist, nearest = cKDTree(ground_points[:, :2]).query(noise_points[:, :2], k=1)
    z_dist = np.abs(noise_points[:, 2] - ground_points[nearest, 2])
    keep = (xy_dist <= voxel_size * args.ground_noise_radius) & (z_dist <= args.ground_noise_height_gap)
    class_names[noise_indices[keep]] = "ground"
    return int(np.sum(keep))


def colorize(class_names: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(class_names), 3))
    for name, color in COLORS.items():
        colors[class_names == name] = color
    return colors


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "region_id", "class_name", "point_count", "area", "height_mean", "height_std",
        "extent_z", "roughness", "planarity", "scattering", "horizontalness", "fill",
        "aspect", "squareness", "ground_score", "tree_score", "building_score",
        "water_score", "other_score", "centroid_x", "centroid_y", "centroid_z",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_legend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "red", "green", "blue"])
        for name, color in COLORS.items():
            writer.writerow([name, *(int(channel * 255) for channel in color)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify region-grown point-cloud patches.")
    parser.add_argument("--non-ground", type=Path, default=DEFAULT_NON_GROUND)
    parser.add_argument("--ground", type=Path, default=DEFAULT_GROUND)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--non-ground-output", type=Path, default=DEFAULT_NON_GROUND_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--legend", type=Path, default=DEFAULT_LEGEND)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--view-target", choices=("combined", "non-ground"), default="combined")

    parser.add_argument("--ground-dominance", type=float, default=1.0)
    parser.add_argument("--tree-dominance", type=float, default=1.8)
    parser.add_argument("--building-dominance", type=float, default=1.0)
    parser.add_argument("--water-dominance", type=float, default=1.0)
    parser.add_argument("--other-dominance", type=float, default=0.6)
    parser.add_argument("--other-score", type=float, default=0.15)

    parser.add_argument("--enable-water", action="store_true")
    parser.add_argument("--water-min-points", type=int, default=200)
    parser.add_argument("--water-min-area", type=float, default=12.0)
    parser.add_argument("--water-max-roughness", type=float, default=0.004)
    parser.add_argument("--water-max-height-std", type=float, default=0.035)
    parser.add_argument("--water-max-extent-z", type=float, default=0.12)
    parser.add_argument("--water-min-horizontalness", type=float, default=0.98)
    parser.add_argument("--water-min-fill", type=float, default=0.60)

    parser.add_argument("--ground-seed-area", type=float, default=10.0)
    parser.add_argument("--ground-seed-score", type=float, default=0.30)
    parser.add_argument("--ground-min-score", type=float, default=0.18)
    parser.add_argument("--ground-max-roughness", type=float, default=0.03)
    parser.add_argument("--ground-max-scattering", type=float, default=0.04)
    parser.add_argument("--ground-min-fill", type=float, default=0.45)
    parser.add_argument("--ground-neighbor-radius", type=float, default=2.2)
    parser.add_argument("--ground-height-gap", type=float, default=0.45)
    parser.add_argument("--no-terrain-ground", action="store_true")
    parser.add_argument("--ground-noise-radius", type=float, default=2.0)
    parser.add_argument("--ground-noise-height-gap", type=float, default=0.22)
    parser.add_argument("--no-ground-noise-absorption", action="store_true")

    parser.add_argument("--tree-min-points", type=int, default=20)
    parser.add_argument("--tree-max-area", type=float, default=8.0)
    parser.add_argument("--tree-min-roughness", type=float, default=0.018)
    parser.add_argument("--tree-min-scattering", type=float, default=0.025)
    parser.add_argument("--tree-min-height-std", type=float, default=0.04)
    parser.add_argument("--tree-min-extent-z", type=float, default=0.18)

    parser.add_argument("--building-min-points", type=int, default=20)
    parser.add_argument("--building-max-area", type=float, default=8.0)
    parser.add_argument("--building-max-roughness", type=float, default=0.025)
    parser.add_argument("--building-max-height-std", type=float, default=0.12)
    parser.add_argument("--building-min-planarity", type=float, default=0.35)
    parser.add_argument("--building-min-horizontalness", type=float, default=0.75)
    parser.add_argument("--building-min-fill", type=float, default=0.20)
    parser.add_argument("--building-max-aspect", type=float, default=8.0)
    parser.add_argument("--building-min-squareness", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    non_ground = load_cloud(args.non_ground)
    full_points = np.asarray(non_ground.points)
    data = np.load(args.labels)
    proxy_points = data["proxy_points"]
    proxy_labels = data["proxy_labels"]
    full_labels = data["full_labels"]
    voxel_size = float(data["voxel_size"][0])
    plane = orient_plane(data["ground_plane"], proxy_points)
    ground_normal = plane[:3] / max(float(np.linalg.norm(plane[:3])), 1e-12)
    heights = signed_height(proxy_points, plane)

    if len(full_labels) != len(full_points):
        raise ValueError("Region labels do not match non-ground point count. Rerun region_grow_non_ground.py.")

    rows = []
    region_class = {}
    for label in np.unique(proxy_labels[proxy_labels >= 0]):
        mask = proxy_labels == label
        features = region_features(int(label), proxy_points[mask], heights[mask], voxel_size, ground_normal)
        scores = class_scores(features, args)
        class_name = winner(scores)
        region_class[int(label)] = class_name
        rows.append({**features, "class_name": class_name, **{f"{k}_score": v for k, v in scores.items()}})

    ground_labels = infer_ground(rows, proxy_points, proxy_labels, voxel_size, args)
    for row in rows:
        if int(row["region_id"]) in ground_labels:
            row["class_name"] = "ground"
            region_class[int(row["region_id"])] = "ground"

    class_names = np.full(len(full_labels), "noise", dtype=object)
    for label, class_name in region_class.items():
        class_names[full_labels == label] = class_name

    absorbed = absorb_ground_noise(class_names, full_points, proxy_points, proxy_labels, ground_labels, voxel_size, args)
    if absorbed:
        print(f"Absorbed {absorbed:,} noise points into terrain ground")

    non_ground.colors = o3d.utility.Vector3dVector(colorize(class_names))
    args.non_ground_output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.non_ground_output), non_ground)

    ground = load_cloud(args.ground)
    ground.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(COLORS["ground"]), (len(ground.points), 1))
    )
    combined = ground + non_ground
    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.output), combined)
    write_report(args.report, rows)
    write_legend(args.legend)

    print("Classified non-ground points:")
    for name in ("ground", "tree", "water", "building", "other", "noise"):
        print(f"  {name:>8}: {int(np.sum(class_names == name)):,}")

    print("Classified regions:")
    for name in ("ground", "tree", "water", "building", "other"):
        print(f"  {name:>8}: {sum(row['class_name'] == name for row in rows):,}")

    print(f"Wrote semantic scene to: {args.output}")
    print(f"Wrote report to: {args.report}")

    if args.view:
        cloud = combined if args.view_target == "combined" else non_ground
        o3d.visualization.draw_geometries([cloud], window_name="Semantic class inference")


if __name__ == "__main__":
    main()
