#!/usr/bin/env python3
"""
Fractal Dimension from Perimeter–Area (log–log OLS), using shared utils.

- Strict source separation: SOURCE ∈ {"png","siteperc"} (no pooling).
- PNG filters: time_windows (list), threshold_policy ("all" | "argmax_clouds"),
               thresholds_whitelist (optional for "all"),
               on_missing_metadata ("skip_run" | "keep_all_thresholds").
- Site-perc filters: p_vals (list or None).
- Common filters: min_area, max_area, optional min_perim/max_perim.
- Fit: OLS on logP ~ m*logA + b; report D = 2m ± 2*stderr_m.

Outputs (files are prefixed by PREFIX) under OUTPUT_DIR:
  - {PREFIX}__perim_vs_area_loglog.png
  - {PREFIX}__metrics.json
"""

from __future__ import annotations
import os, json, math, glob
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clouds.utils.analysis_utils import (
    abs_path, ensure_dir, discover_png_runs, discover_sp_parquets,
    within_any_window, choose_argmax_threshold, apply_common_filters, read_parquet_cols
)

# -------------------- Config (overridable via CLI) --------------------
SOURCE = "siteperc"                # "png" | "siteperc"
PREFIX = "SITEPERC_all"            # output filename prefix (accepts --suffix_tag for compat)
# Paths (relative to $HOME or absolute)
PNG_PER_CLOUD_ROOT = "scratch/all_clouds_data/threshold_autocorr_bd/per_cloud"
PNG_META_ROOT      = "scratch/all_clouds_data/threshold_autocorr_bd"
SP_PER_CLOUD_ROOT  = "scratch/all_clouds_data/sp_autocorr_bd/per_cloud"
OUTPUT_DIR         = "scratch/all_clouds_data/analysis/fractal_dimension"

# Filters
MIN_AREA = 500
MAX_AREA = 7_500_000
MIN_PERIM: Optional[int] = None
MAX_PERIM: Optional[int] = None

# PNG knobs
TIME_WINDOWS: List[Tuple[str, str]] = []   # [] => all times
THRESHOLD_POLICY = "argmax_clouds"         # "all" | "argmax_clouds"
THRESHOLDS_WHITELIST: Optional[List[float]] = None
ON_MISSING_METADATA = "skip_run"           # "skip_run" | "keep_all_thresholds"

# Site-perc knobs
P_VALS: Optional[List[float]] = None       # None => accept all

# Plot look
FIGSIZE = (9.0, 7.0)
SCATTER_ALPHA = 0.25
SCATTER_S = 2
DPI = 300

# Float compares
ATOL = 1e-12
RTOL = 1e-10

# Resolve paths to absolute (so cwd doesn't matter)
PNG_PER_CLOUD_ROOT = abs_path(PNG_PER_CLOUD_ROOT)
PNG_META_ROOT      = abs_path(PNG_META_ROOT)
SP_PER_CLOUD_ROOT  = abs_path(SP_PER_CLOUD_ROOT)
OUTPUT_DIR         = abs_path(OUTPUT_DIR)


# -------------------- Validation --------------------
def _assert_source():
    if SOURCE not in {"png","siteperc"}:
        raise ValueError('SOURCE must be "png" or "siteperc" (never both).')
    if SOURCE == "png":
        if THRESHOLD_POLICY not in {"all","argmax_clouds"}:
            raise ValueError('For PNG, THRESHOLD_POLICY must be "all" or "argmax_clouds".')
        if THRESHOLD_POLICY == "argmax_clouds" and ON_MISSING_METADATA not in {"skip_run","keep_all_thresholds"}:
            raise ValueError('ON_MISSING_METADATA must be "skip_run" or "keep_all_thresholds".')


# -------------------- Row iterators --------------------
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
            df = read_parquet_cols(pq, ["area","perim","threshold"])
            if df.empty:
                continue
            df = apply_common_filters(df, MIN_AREA, MAX_AREA, MIN_PERIM, MAX_PERIM)
            if df.empty:
                continue
            if "threshold" in df.columns:
                if THRESHOLD_POLICY == "argmax_clouds":
                    if chosen_thr is not None:
                        df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=RTOL, atol=ATOL)]
                elif THRESHOLD_POLICY == "all":
                    if THRESHOLDS_WHITELIST is not None:
                        df = df[df["threshold"].isin(THRESHOLDS_WHITELIST)]
            if df.empty:
                continue
            for _, row in df.iterrows():
                yield row

def _iter_sp_rows():
    for pq in discover_sp_parquets(SP_PER_CLOUD_ROOT):
        df = read_parquet_cols(pq, ["area","perim","p_val"])
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


# -------------------- Collect points --------------------
def _collect_points():
    """Return (area_array, perim_array, counts_dict, meta_used_dict)."""
    areas, perims = [], []
    counts: Dict[str,int] = {"files_considered": 0, "runs_considered": 0, "scanned_rows": 0, "kept_rows": 0, "runs_skipped_no_meta": 0}
    meta_used: Dict[str,dict] = {}

    if SOURCE == "png":
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
                    counts["runs_skipped_no_meta"] += 1
                    continue
                meta_used[run_tag] = {"policy":"argmax_clouds","chosen_thr":chosen_thr,"on_missing":ON_MISSING_METADATA}
            else:
                meta_used[run_tag] = {"policy":"all","whitelist":THRESHOLDS_WHITELIST}

            scanned_in_run = 0
            kept_in_run = 0
            for pq in shard_paths:
                df = read_parquet_cols(pq, ["area","perim","threshold"])
                scanned_in_run += len(df)
                if df.empty:
                    continue
                df = apply_common_filters(df, MIN_AREA, MAX_AREA, MIN_PERIM, MAX_PERIM)
                if df.empty:
                    continue
                if "threshold" in df.columns:
                    if THRESHOLD_POLICY == "argmax_clouds":
                        if chosen_thr is not None:
                            df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=RTOL, atol=ATOL)]
                    elif THRESHOLD_POLICY == "all":
                        if THRESHOLDS_WHITELIST is not None:
                            df = df[df["threshold"].isin(THRESHOLDS_WHITELIST)]
                if df.empty:
                    continue

                areas.append(df["area"].to_numpy(dtype=float, copy=False))
                perims.append(df["perim"].to_numpy(dtype=float, copy=False))
                kept_in_run += len(df)

            if kept_in_run > 0:
                counts["runs_considered"] += 1
                counts["scanned_rows"] += scanned_in_run
                counts["kept_rows"] += kept_in_run

    else:  # siteperc
        for pq in discover_sp_parquets(SP_PER_CLOUD_ROOT):
            counts["files_considered"] += 1
            df = read_parquet_cols(pq, ["area","perim","p_val"])
            counts["scanned_rows"] += len(df)
            if df.empty:
                continue
            df = apply_common_filters(df, MIN_AREA, MAX_AREA, MIN_PERIM, MAX_PERIM)
            if df.empty:
                continue
            if P_VALS is not None and "p_val" in df.columns:
                df = df[df["p_val"].isin(P_VALS)]
            if df.empty:
                continue
            areas.append(df["area"].to_numpy(dtype=float, copy=False))
            perims.append(df["perim"].to_numpy(dtype=float, copy=False))
            counts["kept_rows"] += len(df)

    if counts["kept_rows"] == 0:
        return np.array([]), np.array([]), counts, meta_used

    return np.concatenate(areas).astype(float), np.concatenate(perims).astype(float), counts, meta_used


# -------------------- OLS + Plot --------------------
def _ols_fit(logA: np.ndarray, logP: np.ndarray) -> Tuple[float,float,float,float]:
    n = logA.size
    if n < 3:
        raise ValueError("Not enough points for OLS.")
    xm, ym = logA.mean(), logP.mean()
    Sxx = float(np.sum((logA - xm)**2))
    Sxy = float(np.sum((logA - xm)*(logP - ym)))
    m = Sxy / Sxx
    b = ym - m * xm
    yhat = m*logA + b
    SS_res = float(np.sum((logP - yhat)**2))
    SS_tot = float(np.sum((logP - ym)**2))
    R2 = 1.0 - SS_res/SS_tot if SS_tot > 0 else float("nan")
    sigma2 = SS_res / (n - 2)
    stderr_m = math.sqrt(sigma2 / Sxx)
    return m, b, stderr_m, R2

def _plot(area: np.ndarray, perim: np.ndarray, m: float, b: float, title: str, subtitle: str, out_png: str):
    mask = np.isfinite(area) & np.isfinite(perim) & (area > 0) & (perim > 0)
    area = area[mask]; perim = perim[mask]
    if area.size < 3:
        raise ValueError("Not enough finite/positive points for log plot.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(area, perim, s=SCATTER_S, alpha=SCATTER_ALPHA)
    a_min, a_max = float(np.min(area)), float(np.max(area))
    a_line = np.logspace(np.log10(a_min), np.log10(a_max), 200)
    c = math.exp(b)
    p_line = c * np.power(a_line, m)
    ax.plot(a_line, p_line, linewidth=2, color="red", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Area (A)"); ax.set_ylabel("Perimeter (P)")
    ax.set_title(title, pad=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    # room for footer
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.text(
        0.5, 0.04, subtitle, ha="center", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, linewidth=0.6)
    )

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


# -------------------- Main --------------------
def main():
    _assert_source()
    ensure_dir(OUTPUT_DIR)

    area, perim, counts, meta_used = _collect_points()
    if area.size < 3:
        print("[INFO] No points after filtering; nothing to fit.")
        print("Counts:", json.dumps(counts))
        if meta_used: print("Meta used:", json.dumps(meta_used))
        return

    logA, logP = np.log(area), np.log(perim)
    m, b, stderr_m, R2 = _ols_fit(logA, logP)
    D = 2.0 * m
    stderr_D = 2.0 * stderr_m

    # Labels
    filt_bits = [
        f"{MIN_AREA} ≤ area ≤ {MAX_AREA}",
        f"MIN_PERIM={MIN_PERIM}" if MIN_PERIM is not None else None,
        f"MAX_PERIM={MAX_PERIM}" if MAX_PERIM is not None else None,
    ]
    if SOURCE == "png":
        if TIME_WINDOWS:
            filt_bits.append(f"time_windows={TIME_WINDOWS}")
        if THRESHOLD_POLICY == "all":
            filt_bits.append("thresholds=ALL" if THRESHOLDS_WHITELIST is None else f"thresholds_whitelist={THRESHOLDS_WHITELIST}")
        else:
            filt_bits.append("threshold=argmax(num_clouds_by_threshold)")
            filt_bits.append(f"on_missing_metadata={ON_MISSING_METADATA}")
    else:
        filt_bits.append(f"p_vals={P_VALS if P_VALS is not None else 'ALL'}")
    filt_bits = [x for x in filt_bits if x]

    title = f"Perimeter vs Area (log–log) — {SOURCE.upper()}"
    subtitle = (
        f"N = {area.size}  |  slope m = {m:.4f} ± {stderr_m:.4f}  |  "
        f"D = 2m = {D:.4f} ± {stderr_D:.4f}  |  R² = {R2:.4f}\n"
        f"Mode: {'png['+THRESHOLD_POLICY+']' if SOURCE=='png' else 'siteperc'}  |  Filters: " + "; ".join(filt_bits)
    )

    # Files
    base = os.path.join(OUTPUT_DIR, PREFIX)
    plot_png = f"{base}__perim_vs_area_loglog.png"
    metrics_json = f"{base}__metrics.json"

    # Plot + metrics
    _plot(area, perim, m, b, title, subtitle, plot_png)

    metrics = {
        "source": SOURCE,
        "prefix": PREFIX,
        "counts": counts,
        "filters": {
            "min_area": MIN_AREA, "max_area": MAX_AREA,
            "min_perim": MIN_PERIM, "max_perim": MAX_PERIM,
            "time_windows": TIME_WINDOWS if SOURCE == "png" else None,
            "threshold_policy": THRESHOLD_POLICY if SOURCE == "png" else None,
            "thresholds_whitelist": THRESHOLDS_WHITELIST if (SOURCE == "png" and THRESHOLD_POLICY=="all") else None,
            "on_missing_metadata": ON_MISSING_METADATA if (SOURCE == "png" and THRESHOLD_POLICY=="argmax_clouds") else None,
            "p_vals": P_VALS if SOURCE == "siteperc" else None,
        },
        "fit": {
            "method": "OLS on logP ~ m*logA + b",
            "m": float(m),
            "stderr_m": float(stderr_m),
            "D": float(D),
            "stderr_D": float(stderr_D),
            "R2": float(R2),
            "N": int(area.size),
        },
        "meta_used": meta_used if SOURCE == "png" else {},
        "outputs": {
            "plot_png": plot_png,
            "metrics_json": metrics_json
        },
        "notes": "Uses shared utils; strict source separation; 1-sigma errors.",
    }
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] FD figure + metrics written:")
    print(plot_png)
    print(metrics_json)


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute fractal dimension (log–log OLS) using shared utils.")
    ap.add_argument("--source", choices=["png","siteperc"])
    ap.add_argument("--threshold_policy", choices=["all","argmax_clouds"])
    ap.add_argument("--time_window", help="PNG only: 'HH:MM-HH:MM' or 'all'.")
    # Naming: prefer --prefix but keep --suffix_tag for back-compat with existing wrappers
    ap.add_argument("--prefix", help="Output filename prefix.")
    ap.add_argument("--suffix_tag", help="Alias of --prefix for back-compat.")
    ap.add_argument("--p_vals", help="Site-perc p-values, comma-separated or 'all'.")
    # Optional filter overrides
    ap.add_argument("--min_area", type=int)
    ap.add_argument("--max_area", type[int])
    ap.add_argument("--min_perim", type=int)
    ap.add_argument("--max_perim", type=int)
    args = ap.parse_args()

    if args.source: SOURCE = args.source
    if args.threshold_policy: THRESHOLD_POLICY = args.threshold_policy
    if args.time_window:
        if args.time_window.lower() == "all":
            TIME_WINDOWS = []
        else:
            s, e = [x.strip() for x in args.time_window.split("-")]
            TIME_WINDOWS = [(s, e)]
    # naming
    if args.prefix: PREFIX = args.prefix
    if args.suffix_tag: PREFIX = args.suffix_tag  # back-compat alias

    # site-perc / filters
    if args.p_vals:
        if args.p_vals.lower() == "all":
            P_VALS = None
        else:
            P_VALS = [float(x) for x in args.p_vals.split(",")]
    if args.min_area is not None: MIN_AREA = int(args.min_area)
    if args.max_area is not None: MAX_AREA = int(args.max_area)
    if args.min_perim is not None: MIN_PERIM = int(args.min_perim)
    if args.max_perim is not None: MAX_PERIM = int(args.max_perim)

    main()
