#!/usr/bin/env python3
"""
sp_cloud_diagnostic.py

Minimal site-percolation diagnostic script.
Generates a lattice, runs the preprocessing pipeline, 
and reports how many clouds (connected components) were found.

Use this to sanity-check:
  - Whether clouds appear at a given fill probability (p)
  - Whether order/connectivity choices behave as expected
"""

# python -m clouds.data_processing_scripts.sp_proc.new_pipeline_test

import numpy as np
from clouds.utils import cloud_utils
from clouds.utils.image_utils import save_lattice_png
from pathlib import Path

# -------------------------------------------------------------
# === USER CONFIGURATION ===
# (Set these before running)
# -------------------------------------------------------------
WIDTH  = 4000         # lattice width
HEIGHT = 2666         # lattice height
P_VAL  = 0.592746    # fill probability (e.g., site percolation threshold)
SEED   = 42          # RNG seed for reproducibility

ORDER  = "LF_bbox"        # "FL" or "LF_bbox"
CL     = 4           # label connectivity (foreground): 4 or 8
CF     = 8           # flood connectivity (background): 4 or 8
MIN_AREA = 100       # minimum cloud area (pixels)
MAX_AREA = 10_000_000  # maximum cloud area (pixels)
BBOX_PAD = 1         # padding around bbox for per-bbox fill

OUTPUT_DIR = Path("scratch/diagnostic_samples")  # where PNGs will be saved

# -------------------------------------------------------------
# === RUN DIAGNOSTIC ===
# -------------------------------------------------------------
print("=== Site Percolation Cloud Diagnostic ===")
print(f"Size: {HEIGHT}x{WIDTH}")
print(f"p = {P_VAL:.6f}   seed = {SEED}")
print(f"Order: {ORDER}   Label conn: {CL}   Flood conn: {CF}")
print(f"Area range: [{MIN_AREA}, {MAX_AREA}]   BBox pad: {BBOX_PAD}")
print("=========================================")

# 1) Generate lattice
lattice = cloud_utils.generate_site_percolation_lattice(
    width=WIDTH, height=HEIGHT, fill_prob=P_VAL, seed=SEED
)

# 2) Run preprocessing + cropping
cropped_clouds = cloud_utils.preprocess_and_crop_clouds(
    lattice,
    order=ORDER,
    cl=CL,
    cf=CF,
    min_area=MIN_AREA,
    max_area=MAX_AREA,
    bbox_pad=BBOX_PAD,
)

# 3) Report results
n_clouds = len(cropped_clouds)
areas = [np.count_nonzero(c) for c in cropped_clouds]

print(f"[OK] Clouds found: {n_clouds}")
if n_clouds > 0:
    print(f"    Mean area: {np.mean(areas):.1f}")
    print(f"    Min area:  {np.min(areas)}")
    print(f"    Max area:  {np.max(areas)}")
else:
    print("    No clouds found within given area range.")

# -------------------------------------------------------------
# === SAVE SAMPLE CROPS AS PNGs ===
# -------------------------------------------------------------
if n_clouds > 0:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_count = min(3, n_clouds)

    print(f"\nSaving {sample_count} sample cropped clouds to: {OUTPUT_DIR}")
    for i in range(sample_count):
        crop = cropped_clouds[i].astype(np.uint8) * 255
        area = np.count_nonzero(crop)
        png_path = OUTPUT_DIR / f"cloud_{i:02d}_area{area}_seed{SEED}.png"
        save_lattice_png(crop, png_path)
        print(f"  -> {png_path}")

print("=========================================")

