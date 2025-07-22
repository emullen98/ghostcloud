#!/usr/bin/env python3

import sys
import os
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from filelock import FileLock

# ✅ Import your helper functions exactly as before
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../utils')))
from slice_analysis_utils import filter_min_area, convert_to_log_values

# ===================
# === Parameters ====
# ===================

gamma_tag = sys.argv[1]         # e.g., "0p1975"
p_tag = sys.argv[2]             # e.g., "0p460"
slice_id = int(sys.argv[3])     # e.g., 0 or -1
base_dir = sys.argv[4]          # e.g., "/scratch/mbiju2/storm/cp_crit_exp_20250701_123456"

# ======================
# === Path setup =======
# ======================

pair_dir = Path(base_dir) / f"g_{gamma_tag}" / f"p_{p_tag}"
summary_csv_path = pair_dir / "d_summary.csv"
lock_path = pair_dir / "d_summary.csv.lock"

print(f"[INFO] Processing gamma={gamma_tag}, p={p_tag}, slice_id={slice_id}")
print(f"[INFO] Base dir: {pair_dir}")

# =====================
# === Pool data =======
# =====================

all_dfs = []

for run_id in range(1, 501):
    run_dir = pair_dir / f"run_{run_id}"
    csv_file = run_dir / "slice_data.csv.gz"

    if not csv_file.exists():
        print(f"[WARN] Missing file: {csv_file}")
        continue

    with gzip.open(csv_file, "rt") as f:
        df = pd.read_csv(f)

    # Filter for this slice
    df_slice = df[df["slice_id"] == slice_id]

    if not df_slice.empty:
        all_dfs.append(df_slice)

# Merge all slice data
if not all_dfs:
    print(f"[WARN] No data for slice {slice_id} across lattices. Skipping D estimate.")
    D_est = np.nan
else:
    pooled_df = pd.concat(all_dfs, ignore_index=True)

    # ✅ Use helper function to filter by min area
    filtered_df = filter_min_area(pooled_df, min_area=3000)

    if filtered_df.empty:
        print(f"[WARN] No data left after area filtering for slice {slice_id}.")
        D_est = np.nan
    else:
        # ✅ Use helper to create log columns
        log_df = convert_to_log_values(filtered_df)

        log_area = log_df["log_area"].values.reshape(-1, 1)
        log_perim = log_df["log_perimeter"].values
        model = LinearRegression().fit(log_area, log_perim)
        slope = model.coef_[0]
        D_est = 2 * slope  # Correct formula: D = 2 × slope

print(f"[INFO] Estimated D for slice {slice_id}: {D_est}")

# ======================
# === Append to CSV ====
# ======================

header = not summary_csv_path.exists()

lock = FileLock(str(lock_path))

with lock:
    with open(summary_csv_path, "a") as f:
        if header:
            f.write("slice_id,D_estimate\n")
        f.write(f"{slice_id},{D_est}\n")

print(f"[INFO] Appended result to {summary_csv_path}")
