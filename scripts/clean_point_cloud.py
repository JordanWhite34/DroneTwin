"""Clean and downsample COLMAP point clouds.

This script is the post-processing stage after COLMAP dense fusion. It keeps the
raw COLMAP output intact and writes a separate cleaned PLY file so different
cleanup settings can be compared visually.

The module can be used two ways:

1. From the command line:
   `python scripts/clean_point_cloud.py`
2. From Python:
   `PointCloudCleaner(...).run()`

The cleanup stages run in this order:

1. Optional low-percentile clipping along one axis.
2. Statistical outlier removal for sparse isolated points.
3. Optional radius outlier removal for local-density filtering.
4. Optional voxel downsampling for file-size reduction.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_DIR / "data" / "processed" / "dense" / "fused-keep-more.ply"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "dense" / "fused-keep-more-cleaned.ply"


class PointCloudCleaner:
    """Configurable point-cloud cleanup pipeline.

    Parameters are intentionally the same as the CLI options so the class can be
    imported and used with the same behavior as `scripts/clean_point_cloud.py`.

    `voxel_size` and `voxel_fraction` control downsampling:
    - `voxel_size` is an absolute scene-unit value and takes precedence.
    - `voxel_fraction` derives the voxel size from the point-cloud bounding-box
      diagonal, which is useful when COLMAP scene scale varies between runs.

    `stat_neighbors` and `stat_std_ratio` control statistical outlier removal:
    lower `stat_std_ratio` removes more points, while higher values are more
    permissive.

    `radius` and `radius_min_neighbors` control radius outlier removal. This is
    disabled by default because the right radius depends strongly on scene scale.

    `clip_low_axis` and `clip_low_percentile` are a blunt cleanup tool for
    sparse below-scene points. Confirm the COLMAP coordinate frame visually
    before enabling this, because the vertical axis is not guaranteed by name.
    """

    def __init__(
        self,
        input: Path,
        output: Path,
        voxel_size: float | None,
        voxel_fraction: float,
        stat_neighbors: int,
        stat_std_ratio: float,
        radius: float | None,
        radius_min_neighbors: int,
        clip_low_axis: str | None,
        clip_low_percentile: float,
        write_ascii: bool,
    ) -> None:
        self.input = input
        self.output = output
        self.voxel_size = voxel_size
        self.voxel_fraction = voxel_fraction
        self.stat_neighbors = stat_neighbors
        self.stat_std_ratio = stat_std_ratio
        self.radius = radius
        self.radius_min_neighbors = radius_min_neighbors
        self.clip_low_axis = clip_low_axis
        self.clip_low_percentile = clip_low_percentile
        self.write_ascii = write_ascii

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PointCloudCleaner":
        """Build a cleaner from parsed CLI arguments."""
        return cls(
            input=args.input,
            output=args.output,
            voxel_size=args.voxel_size,
            voxel_fraction=args.voxel_fraction,
            stat_neighbors=args.stat_neighbors,
            stat_std_ratio=args.stat_std_ratio,
            radius=args.radius,
            radius_min_neighbors=args.radius_min_neighbors,
            clip_low_axis=args.clip_low_axis,
            clip_low_percentile=args.clip_low_percentile,
            write_ascii=args.write_ascii,
        )

    def run(self) -> None:
        """Load, clean, downsample, and write the configured point cloud."""
        point_cloud = o3d.io.read_point_cloud(str(self.input))
        if point_cloud.is_empty():
            raise ValueError(f"No points loaded from {self.input}")

        self.describe("Loaded", point_cloud)

        if self.clip_low_axis and self.clip_low_percentile > 0:
            point_cloud = self.clip_low_percentile_filter(point_cloud)
            self.describe("After percentile clip", point_cloud)

        if self.stat_neighbors > 0:
            point_cloud, indices = point_cloud.remove_statistical_outlier(
                nb_neighbors=self.stat_neighbors,
                std_ratio=self.stat_std_ratio,
            )
            print(f"Statistical outlier removal kept {len(indices):,} points")
            self.describe("After statistical filter", point_cloud)

        if self.radius is not None and self.radius > 0:
            point_cloud, indices = point_cloud.remove_radius_outlier(
                nb_points=self.radius_min_neighbors,
                radius=self.radius,
            )
            print(f"Radius outlier removal kept {len(indices):,} points")
            self.describe("After radius filter", point_cloud)

        resolved_voxel_size = self.resolve_voxel_size(point_cloud)
        if resolved_voxel_size > 0:
            print(f"Voxel downsampling with voxel size {resolved_voxel_size:.6f}")
            point_cloud = point_cloud.voxel_down_sample(voxel_size=resolved_voxel_size)
            self.describe("After voxel downsample", point_cloud)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(
            str(self.output),
            point_cloud,
            write_ascii=self.write_ascii,
            compressed=False,
        )
        print(f"Wrote {self.output}")

    def clip_low_percentile_filter(
        self,
        point_cloud: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:
        """Remove the lowest configured percentile along the configured axis."""
        axis_index = {"x": 0, "y": 1, "z": 2}[self.clip_low_axis]
        points = np.asarray(point_cloud.points)
        threshold = float(np.percentile(points[:, axis_index], self.clip_low_percentile))
        keep_indices = np.flatnonzero(points[:, axis_index] >= threshold)
        print(
            f"Clipping lowest {self.clip_low_percentile:g}% on {self.clip_low_axis}-axis "
            f"below {threshold:.3f}: keeping {len(keep_indices):,}/{len(points):,} points"
        )
        return point_cloud.select_by_index(keep_indices)

    def resolve_voxel_size(self, point_cloud: o3d.geometry.PointCloud) -> float:
        """Return the absolute voxel size to use for downsampling.

        An explicit `voxel_size` wins. Otherwise, the size is derived from the
        point-cloud bounding-box diagonal and `voxel_fraction`. Returning `0`
        disables voxel downsampling.
        """
        if self.voxel_size is not None:
            return self.voxel_size
        if self.voxel_fraction <= 0:
            return 0.0
        return self.bounding_box_diagonal(point_cloud) * self.voxel_fraction

    @staticmethod
    def point_count(point_cloud: o3d.geometry.PointCloud) -> int:
        """Return the number of points in an Open3D point cloud."""
        return len(point_cloud.points)

    @staticmethod
    def bounding_box_diagonal(point_cloud: o3d.geometry.PointCloud) -> float:
        """Return the length of the point-cloud axis-aligned bounding-box diagonal."""
        bounds = point_cloud.get_axis_aligned_bounding_box()
        return float(np.linalg.norm(bounds.get_extent()))

    @classmethod
    def describe(cls, label: str, point_cloud: o3d.geometry.PointCloud) -> None:
        """Print point count and bounding-box bounds for a processing stage."""
        bounds = point_cloud.get_axis_aligned_bounding_box()
        min_bound = bounds.get_min_bound()
        max_bound = bounds.get_max_bound()
        print(
            f"{label}: {cls.point_count(point_cloud):,} points | "
            f"min=({min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}) | "
            f"max=({max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f})"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for point-cloud cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean and downsample a COLMAP point cloud."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input point cloud path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output point cloud path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Absolute voxel size for downsampling. Overrides --voxel-fraction.",
    )
    parser.add_argument(
        "--voxel-fraction",
        type=float,
        default=0.001,
        help=(
            "Voxel size as a fraction of the point-cloud bounding-box diagonal. "
            "Use 0 to disable downsampling. Default: 0.001"
        ),
    )
    parser.add_argument(
        "--stat-neighbors",
        type=int,
        default=20,
        help="Neighbor count for statistical outlier removal. Use 0 to disable. Default: 20",
    )
    parser.add_argument(
        "--stat-std-ratio",
        type=float,
        default=2.0,
        help="Standard deviation threshold for statistical outlier removal. Default: 2.0",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Radius for radius outlier removal. Disabled unless provided.",
    )
    parser.add_argument(
        "--radius-min-neighbors",
        type=int,
        default=6,
        help="Minimum neighbors inside --radius. Default: 6",
    )
    parser.add_argument(
        "--clip-low-axis",
        choices=("x", "y", "z"),
        default=None,
        help="Optionally remove the lowest percentile along this axis.",
    )
    parser.add_argument(
        "--clip-low-percentile",
        type=float,
        default=0.0,
        help=(
            "Lowest percentile to remove along --clip-low-axis, e.g. 0.25. "
            "Use this for sparse below-scene points. Default: 0"
        ),
    )
    parser.add_argument(
        "--write-ascii",
        action="store_true",
        help="Write ASCII PLY instead of binary. Binary is smaller and faster.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI entry point."""
    cleaner = PointCloudCleaner.from_args(parse_args())
    cleaner.run()


if __name__ == "__main__":
    main()
