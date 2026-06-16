# Quarry Digital Twin

This project builds a 3D digital twin of a quarry from drone imagery using
COLMAP, Open3D, and transparent geometry-based point-cloud processing.

The current goal is a clean, inspectable pipeline:

```text
drone images
  -> COLMAP dense reconstruction
  -> point-cloud cleanup
  -> spatial analysis
  -> ground / non-ground split
  -> non-ground surface regions
  -> coarse semantic classes
```

## Pipeline

Run the stages in this order from the project root:

```powershell
conda run -n drone-twin python scripts\clean_point_cloud.py
conda run -n drone-twin python scripts\analyze_point_cloud.py
conda run -n drone-twin python scripts\segment_point_cloud.py
conda run -n drone-twin python scripts\region_grow_non_ground.py --proxy-output default
conda run -n drone-twin python scripts\classify_regions.py --view
```

`segment_point_cloud.py` writes:

```text
data\processed\dense\ground.ply
data\processed\dense\non_ground.ply
```

`region_grow_non_ground.py` writes:

```text
data\processed\dense\non_ground_regions.ply
data\processed\dense\non_ground_regions_proxy.ply
data\processed\dense\non_ground_region_labels.npz
```

`classify_regions.py` writes:

```text
data\processed\dense\semantic_classes.ply
data\processed\dense\non_ground_semantic_classes.ply
data\processed\dense\region_class_report.csv
data\processed\dense\semantic_class_legend.csv
```

## Semantic Classes

The semantic classifier is intentionally rule-based and inspectable. It computes
simple region features, scores each class, then applies dominance multipliers:

```powershell
conda run -n drone-twin python scripts\classify_regions.py --view `
  --ground-dominance 1.0 `
  --tree-dominance 1.8 `
  --building-dominance 1.0 `
  --other-dominance 0.6
```

Water classification is disabled by default because the current scene does not
contain water. Enable it explicitly only for datasets where water is expected:

```powershell
conda run -n drone-twin python scripts\classify_regions.py --enable-water --view
```

## Repository Structure

```text
scripts/
  run_colmap.py              # COLMAP reconstruction wrapper
  clean_point_cloud.py       # cleanup and downsampling
  analyze_point_cloud.py     # basic spatial statistics and visualization
  segment_point_cloud.py     # ground / non-ground split
  region_grow_non_ground.py  # surface-region extraction
  classify_regions.py        # coarse semantic classification

data/
  raw/                       # source imagery, not committed
  processed/                 # generated point clouds/results, not committed
```

## Setup

Create the environment:

```powershell
conda create -n drone-twin python=3.10
conda activate drone-twin
pip install -r requirements.txt
```

COLMAP should be installed separately and available from the command line.

## Project Status

Current status:

- [x] Dataset organized
- [x] COLMAP sparse reconstruction
- [x] COLMAP dense reconstruction
- [x] Point-cloud cleanup
- [x] Basic spatial analysis
- [x] Ground / non-ground segmentation
- [x] Surface-region extraction
- [x] Coarse semantic classification
- [ ] App / interactive demo
- [ ] Demo video / writeup

## Long-Term Direction

The next maintainability step is to move stable script logic into a small
library under `src/`, leaving scripts as thin command-line entry points. That
will make the same pipeline easier to call from a future app.
