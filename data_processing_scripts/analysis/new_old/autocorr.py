#!/usr/bin/env python3
"""
Autocorr Aggregator — CSV-only, 0-based r, denom = (Σ area_i) × ring_counts

Sample runs (manual):
---------------------
# SITEPERC (all clouds, num_all/denom via area×ring)
python analysis/autocorr.py --source siteperc --which all --prefix SITEPERC_all

# SITEPERC, boundary-only numerator
python analysis/autocorr.py --source siteperc --which bnd --prefix SITEPERC_bnd

# PNG (argmax threshold) for a specific time window
python analysis/autocorr.py --source png --threshold_policy argmax_clouds --time_window 03:00-04:00 --which all --prefix PNG_argmax_0300

Notes:
------
- r is 0-based everywhere. CSVs store r_index = 0..R-1; split CSVs use column r = 0..R-1.
- Denominator ignores parquet, built as: agg_den[r] = (Σ area_i) * ring_counts_quadrant(R)[r].
- We do ONE ring_counts_quadrant(R) call per run; no disk cache; util file remains untouched.
"""

from __future__ import annotations
import os, json, glob
from pathlib import Path
from typing import List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

# Shared helpers
from clouds.utils.analysis_utils import (
    abs_path, ensure_dir, discover_png_runs, discover_sp_parquets,
    within_any_window, choose_argmax_threshold, apply_common_filters, read_parquet_cols
)

# IMPORTANT: do NOT modify this util; we just import and use it.
from clouds.utils.autocorr_utils import ring_counts_quadrant

# -------------------- Config (CLI-overridable) --------------------
SOURCE = "siteperc"     # "png" | "siteperc"
PREFIX = "SITEPERC_all" # output filename prefix
WHICH  = "all"          # "all" | "bnd"

# paths
PNG_PER_CLOUD_ROOT = "scratch/all_clouds_data/threshold_autocorr_bd/per_cloud"
PNG_META_ROOT      = "scratch/all_clouds_data/threshold_autocorr_bd"
SP_PER_CLOUD_ROOT  = "scratch/all_clouds_data/sp_autocorr_bd/per_cloud"
OUTPUT_DIR         = "scratch/all_clouds_data/analysis/autocorr"

# filters
MIN_AREA = 500
MAX_AREA = 7_500_000
MIN_PERIM: Optional[int] = None
MAX_PERIM: Optional[int] = None

# PNG knobs
TIME_WINDOWS: List[Tuple[str,str]] = []
THRESHOLD_POLICY = "argmax_clouds"   # "all" | "argmax_clouds"
THRESHOLDS_WHITELIST: Optional[List[float]] = None
ON_MISSING_METADATA = "skip_run"     # "skip_run" | "keep_all_thresholds"

# SP knobs
P_VALS: Optional[List[float]] = None

# numerics
ATOL = 1e-12
RTOL = 1e-10
EPS  = 1e-12

# resolve abs paths
PNG_PER_CLOUD_ROOT = abs_path(PNG_PER_CLOUD_ROOT)
PNG_META_ROOT      = abs_path(PNG_META_ROOT)
SP_PER_CLOUD_ROOT  = abs_path(SP_PER_CLOUD_ROOT)
OUTPUT_DIR         = abs_path(OUTPUT_DIR)

# -------------------- Validation --------------------
def _assert_source():
    if SOURCE not in {"png","siteperc"}:
        raise ValueError('SOURCE must be "png" or "siteperc"')
    if WHICH not in {"all","bnd"}:
        raise ValueError('WHICH must be "all" or "bnd"')
    if SOURCE == "png":
        if THRESHOLD_POLICY not in {"all","argmax_clouds"}:
            raise ValueError('THRESHOLD_POLICY must be "all" or "argmax_clouds"')
        if THRESHOLD_POLICY == "argmax_clouds" and ON_MISSING_METADATA not in {"skip_run","keep_all_thresholds"}:
            raise ValueError('ON_MISSING_METADATA must be "skip_run" or "keep_all_thresholds"')

# -------------------- Row iters --------------------
def _iter_png_rows():
    run_tags = discover_png_runs(PNG_PER_CLOUD_ROOT)
    for run_tag in run_tags:
        if not within_any_window(run_tag, TIME_WINDOWS): 
            continue
        shard_paths = sorted(glob.iglob(os.path.join(PNG_PER_CLOUD_ROOT, run_tag, "cloud_metrics.part*.parquet")))
        if not shard_paths: 
            continue
        chosen_thr: Optional[float] = None
        if THRESHOLD_POLICY == "argmax_clouds":
            chosen_thr = choose_argmax_threshold(PNG_META_ROOT, run_tag)
            if chosen_thr is None and ON_MISSING_METADATA == "skip_run":
                continue
        for pq in shard_paths:
            need = ["area","perim","threshold", ("num_all" if WHICH=="all" else "num_bnd")]
            df = read_parquet_cols(pq, need)
            if df.empty: 
                continue
            df = apply_common_filters(df, MIN_AREA, MAX_AREA, MIN_PERIM, MAX_PERIM)
            if df.empty: 
                continue
            if "threshold" in df.columns:
                if THRESHOLD_POLICY == "argmax_clouds":
                    if chosen_thr is not None:
                        df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=RTOL, atol=ATOL)]
                elif THRESHOLD_POLICY == "all" and THRESHOLDS_WHITELIST is not None:
                    df = df[df["threshold"].isin(THRESHOLDS_WHITELIST)]
            if df.empty: 
                continue
            for _, row in df.iterrows():
                yield row

def _iter_sp_rows():
    for pq in discover_sp_parquets(SP_PER_CLOUD_ROOT):
        need = ["area","perim","p_val", ("num_all" if WHICH=="all" else "num_bnd")]
        df = read_parquet_cols(pq, need)
        if df.empty: 
            continue
        df = apply_common_filters(df, MIN_AREA, MAX_AREA, MIN_PERIM, MAX_PERIM)
        if df.empty: 
            continue
        if P_VALS is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(P_VALS)]
        if df.empty: 
            continue
        for _, row in df.iterrows():
            yield row

# -------------------- Aggregation helpers --------------------
def _iter_num_area(rows_iter) -> Iterable[Tuple[np.ndarray, float]]:
    num_field = "num_all" if WHICH=="all" else "num_bnd"
    for row in rows_iter:
        area = row.get("area", None)
        if area is None or pd.isna(area): 
            continue
        try:
            a = float(area)
        except Exception:
            continue
        arr = row.get(num_field, None)
        if isinstance(arr, (list, np.ndarray)):
            num = np.asarray(arr, dtype=np.float64)
            # ensure num is valid and 0-based in-memory (we assume input arrays already r=0..)
            if num.size > 0 and np.all(np.isfinite(num)):
                yield num, a

def _dynamic_add(agg: np.ndarray, arr: np.ndarray) -> np.ndarray:
    if agg.size == 0:
        return arr.astype(np.float64, copy=True)
    if arr.size > agg.size:
        out = np.zeros(arr.size, dtype=np.float64)
        out[:agg.size] = agg
        out[:arr.size] += arr
        return out
    agg[:arr.size] += arr
    return agg

# -------------------- Main --------------------
def main():
    _assert_source()
    ensure_dir(OUTPUT_DIR)

    rows = _iter_png_rows() if SOURCE=="png" else _iter_sp_rows()

    agg_num = np.array([], dtype=np.float64)
    area_total = 0.0
    n_clouds = 0

    for num_vec, area in _iter_num_area(rows):
        # num_vec is assumed aligned as r=0..len-1; we keep it 0-based in aggregation
        agg_num = _dynamic_add(agg_num, num_vec)
        area_total += area
        n_clouds += 1

    if agg_num.size == 0:
        print("[INFO] No arrays after filtering; nothing to aggregate.")
        print(json.dumps({"n_clouds": n_clouds, "area_total": area_total}))
        return

    # R length = numerator length; build denom with ring_counts for r=0..R-1
    R = agg_num.size
    cnt_full = ring_counts_quadrant(R)                 # shape R+1; cnt_full[0] is for r=0
    ring_per_shell = cnt_full[0:R].astype(np.float64)  # -> length R, r = 0..R-1
    agg_den = area_total * ring_per_shell
    agg_ratio = agg_num / np.maximum(agg_den, EPS)

    r_index = np.arange(0, R, dtype=np.int64)          # 0-based in CSV

    base = os.path.join(OUTPUT_DIR, PREFIX)
    curves_csv  = f"{base}__curves.csv"
    metrics_json= f"{base}__metrics.json"
    # optional helpers (still 0-based)
    num_csv = f"{base}__autocorr_num.csv"
    den_csv = f"{base}__autocorr_den.csv"
    cr_csv  = f"{base}__autocorr_Cr.csv"

    # Save CSVs — all 0-based
    pd.DataFrame({
        "r_index": r_index, 
        "agg_num": agg_num, 
        "agg_den": agg_den, 
        "agg_ratio": agg_ratio
    }).to_csv(curves_csv, index=False)

    pd.DataFrame({"r": r_index, "y": agg_num}).to_csv(num_csv, index=False)
    pd.DataFrame({"r": r_index, "y": agg_den}).to_csv(den_csv, index=False)
    pd.DataFrame({"r": r_index, "y": agg_ratio}).to_csv(cr_csv, index=False)

    # Metrics
    metrics = {
        "source": SOURCE,
        "prefix": PREFIX,
        "which": WHICH,
        "den_strategy": "area_sum * ring_counts_quadrant",
        "area_total": float(area_total),
        "R_used": int(R),
        "n_clouds": int(n_clouds),
        "filters": {
            "min_area": MIN_AREA, "max_area": MAX_AREA, "min_perim": MIN_PERIM, "max_perim": MAX_PERIM,
            "time_windows": TIME_WINDOWS if SOURCE=="png" else None,
            "threshold_policy": THRESHOLD_POLICY if SOURCE=="png" else None,
            "thresholds_whitelist": THRESHOLDS_WHITELIST if (SOURCE=="png" and THRESHOLD_POLICY=="all") else None,
            "on_missing_metadata": ON_MISSING_METADATA if (SOURCE=="png" and THRESHOLD_POLICY=="argmax_clouds") else None,
            "p_vals": P_VALS if SOURCE=="siteperc" else None,
        },
        "outputs": {
            "curves_csv": curves_csv,
            "num_csv": num_csv,
            "den_csv": den_csv,
            "ratio_csv": cr_csv,
            "metrics_json": metrics_json,
        },
        "notes": "r (and r_index) are 0-based; CSV-only; denom ignores parquet and is rebuilt from total area × integer ring counts.",
    }
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Autocorr aggregate written (0-based r):")
    print(curves_csv)
    print(metrics_json)

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate autocorr numerators; build denom via area×ring (0-based, CSV-only).")
    ap.add_argument("--source", choices=["png","siteperc"])
    ap.add_argument("--which", choices=["all","bnd"])
    ap.add_argument("--prefix", help="Output file prefix (required).")
    ap.add_argument("--time_window", help="PNG only: 'HH:MM-HH:MM' or 'all'.")
    ap.add_argument("--threshold_policy", choices=["all","argmax_clouds"])
    ap.add_argument("--p_vals", help="Site-perc p-values: comma list or 'all'.")
    args = ap.parse_args()

    if args.source: SOURCE = args.source
    if args.which: WHICH = args.which
    if args.prefix: PREFIX = args.prefix
    if args.time_window:
        if args.time_window.lower() == "all":
            TIME_WINDOWS = []
        else:
            s,e = [x.strip() for x in args.time_window.split("-")]
            TIME_WINDOWS = [(s,e)]
    if args.threshold_policy: THRESHOLD_POLICY = args.threshold_policy
    if args.p_vals:
        if args.p_vals.lower() == "all":
            P_VALS = None
        else:
            P_VALS = [float(x) for x in args.p_vals.split(",")]

    main()
