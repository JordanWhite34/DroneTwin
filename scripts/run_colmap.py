"""Run the COLMAP photogrammetry pipeline from raw images to dense outputs.

How it works:
- Feature extraction finds visual keypoints in each uploaded/drone image.
- Sequential matching links keypoints between neighboring images in flight order.
- Mapping estimates camera poses and builds the sparse reconstruction.
- Image undistortion prepares the sparse model for dense stereo.
- Patch-match stereo estimates per-image depth maps.
- Stereo fusion merges depth maps into `fused-keep-more.ply`, the dense point
  cloud used by the Open3D cleanup and segmentation scripts.
- Optional meshing/texturing stages produce viewable mesh artifacts, but the
  segmentation workflow primarily uses the fused PLY point cloud.
"""

import subprocess
import argparse
import shutil
from pathlib import Path

COLMAP = Path(r"C:\Tools\COLMAP\COLMAP.bat")

PROJECT_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

DATABASE_PATH = PROCESSED_DIR / "database.db"
SPARSE_DIR = PROCESSED_DIR / "sparse"
DENSE_DIR = PROCESSED_DIR / "dense"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the COLMAP reconstruction pipeline."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previous COLMAP outputs before running.",
    )
    return parser.parse_args()


def clean_outputs() -> None:
    for path in [DATABASE_PATH, SPARSE_DIR, DENSE_DIR]:
        if path.is_dir():
            print(f"Removing directory: {path}")
            shutil.rmtree(path)
        elif path.exists():
            print(f"Removing file: {path}")
            path.unlink()


def prepare_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)

# helper function to run commands we pass in and print their results
def run_colmap(args: list[str]) -> None:
    command = [str(COLMAP), *args]

    print("\nRunning:")
    print(" ".join(str(x) for x in command))


    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        print(error.stdout)
        raise

    print(result.stdout)

def main() -> None:
    args = parse_args()

    if args.clean:
        clean_outputs()

    prepare_output_dirs()

    # extract features on GPU
    run_colmap([
        "feature_extractor",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGE_DIR),

        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",

        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
    ])

    # matcher
    run_colmap([
        "sequential_matcher",
        "--database_path", str(DATABASE_PATH),

        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
    ])

    # mapper
    run_colmap([
        "mapper",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGE_DIR),
        "--output_path", str(SPARSE_DIR)
    ])

    # image undistorter
    run_colmap([
        "image_undistorter",
        "--image_path", str(IMAGE_DIR),
        "--input_path", str(SPARSE_DIR / "0"),
        "--output_path", str(DENSE_DIR),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ])

    # patch match stereo
    run_colmap([
        "patch_match_stereo",
        "--workspace_path", str(DENSE_DIR),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.gpu_index", "0",
    ])

    # stereo fusion
    run_colmap([
        "stereo_fusion",
        "--workspace_path", str(DENSE_DIR),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--StereoFusion.min_num_pixels", "2",
        "--StereoFusion.max_reproj_error", "4",
        "--StereoFusion.max_depth_error", "0.03",
        "--StereoFusion.max_normal_error", "30",
        "--output_path", str(DENSE_DIR / "fused-keep-more.ply"),
    ])

    # poisson mesher
    run_colmap([
        "poisson_mesher",
        "--input_path", str(DENSE_DIR / "fused-keep-more.ply"),
        "--output_path", str(DENSE_DIR / "meshed-poisson-improved.ply")
    ])

    # delaunay mesher
    run_colmap([
        "delaunay_mesher",
        "--input_path", str(DENSE_DIR),
        "--output_path", str(DENSE_DIR / "meshed-delaunay.ply")
    ])

    # simplify mesh to reduce its size
    run_colmap([
        "mesh_simplifier",
        "--input_path", str(DENSE_DIR / "meshed-poisson.ply"),
        "--output_path", str(DENSE_DIR / "meshed-poisson-simplified.ply"),
        "--MeshSimplification.target_face_ratio", "0.25"
    ])

    # texture a mesh using the undistorted images
    run_colmap([
        "mesh_texturer",
        "--workspace_path", str(DENSE_DIR),
        "--input_path", str(DENSE_DIR / "meshed-poisson.ply"),
        "--output_path", str(DENSE_DIR / "textured"),
    ])


if __name__ == "__main__":
    main()
