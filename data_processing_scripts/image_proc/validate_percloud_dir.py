#!/usr/bin/env python3
"""
validate_percloud_dir.py

Quick sanity check for per-cloud Parquet outputs.

Usage:
    python validate_percloud_dir.py --dir scratch/threshold_wk/per_cloud/thr_wk_2013-01-06--11-26-15--500
"""

import argparse
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa
import statistics

EXPECTED_COLS = {
    "cloud_idx", "perim", "area", "threshold",
    "cr", "cr_bnd", "num_all", "den_all", "num_bnd", "den_bnd",
}

def main():
    ap = argparse.ArgumentParser(description="Validate per-cloud Parquet directory.")
    ap.add_argument("--dir", type=str, required=True, help="Directory containing .part*.parquet files")
    args = ap.parse_args()
    indir = Path(args.dir)

    part_files = sorted(indir.glob("*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"No .parquet files found in {indir}")

    print(f"[INFO] Found {len(part_files)} part files in {indir}")

    total_rows = 0
    areas, perims, thresholds = [], [], []
    cr_count = num_all_count = num_bnd_count = 0

    for pf in part_files:
        pf = Path(pf)
        table = pq.read_table(pf)

        # Schema check
        missing = EXPECTED_COLS - set(table.column_names)
        if missing:
            print(f"[WARN] {pf.name}: missing expected columns: {missing}")

        nrows = table.num_rows
        total_rows += nrows

        # Collect stats
        if "area" in table.column_names:
            areas.extend(table["area"].to_pylist())
        if "perim" in table.column_names:
            perims.extend(table["perim"].to_pylist())
        if "threshold" in table.column_names:
            thresholds.extend([t for t in table["threshold"].to_pylist() if t is not None])

        if "cr" in table.column_names:
            cr_count += sum(x is not None for x in table["cr"].to_pylist())
        if "num_all" in table.column_names:
            num_all_count += sum(x is not None for x in table["num_all"].to_pylist())
        if "num_bnd" in table.column_names:
            num_bnd_count += sum(x is not None for x in table["num_bnd"].to_pylist())

    # Print summary
    print(f"[OK] Total rows (clouds): {total_rows}")

    if areas:
        print(f"[OK] area stats  -> min: {min(areas)}, max: {max(areas)}, median: {statistics.median(areas)}")
    if perims:
        print(f"[OK] perim stats -> min: {min(perims)}, max: {max(perims)}, median: {statistics.median(perims)}")
    if thresholds:
        uniq_thr = sorted(set(thresholds))
        counts = {t: thresholds.count(t) for t in uniq_thr}
        print(f"[OK] thresholds -> {counts}")

    print(f"[OK] rows with cr saved:       {cr_count}")
    print(f"[OK] rows with num_all saved:  {num_all_count}")
    print(f"[OK] rows with num_bnd saved:  {num_bnd_count}")

    print("[DONE] Validation complete.")


if __name__ == "__main__":
    main()
