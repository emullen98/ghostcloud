#!/usr/bin/env python3
"""
Autocorr aggregates (summed numerator and summed numerator/denominator),
spec-accurate and minimal, mirroring the FD script's structure.

- Strict source separation: SOURCE ∈ {"png","siteperc"} (no pooling).
- PNG filters: time_windows (list), threshold_policy ("all" | "argmax_clouds"),
               thresholds_whitelist (optional with "all"),
               on_missing_metadata ("skip_run" | "keep_all_thresholds").
- Site-perc filters: p_vals (list or None).
- Common filters: min_area, max_area, optional min_perim/max_perim.

- Aggregation:
    agg_num[i] = sum_clouds num[i]
    agg_den[i] = sum_clouds den[i]
    agg_ratio[i] = agg_num[i] / max(agg_den[i], EPS)

- Outputs under:
    scratch/all_clouds_data/analysis/autocorr/
  with filenames prefixed by SUFFIX_TAG:
    * agg_num_{linear,semilog,loglog}.png
    * agg_ratio_{linear,semilog,loglog}.png
    * r_index.npy, agg_num.npy, agg_den.npy, agg_ratio.npy, curves.csv
    * metrics.json (counts, filters, file list, etc.)

Requires: pandas, numpy, pyarrow (for parquet), matplotlib
"""

from __future__ import annotations
import os, re, glob, json
from typing import List, Tuple, Optional, Dict, Iterable, Tuple as Tup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Resolve all key paths absolutely (fixes discovery issues) ---
def _abs(p: str) -> str:
    """Return absolute, expanded, resolved path (handles relative paths safely)."""
    return str(Path(p).expanduser().resolve())

# ===================== CONFIG (EDIT) =====================

# Choose exactly one
SOURCE = "siteperc"        # "png" or "siteperc" (NEVER "both")

# Suffix tag to distinguish runs in filenames
SUFFIX_TAG = "SITEPERC_p04074"  # e.g., "PNG_allthr_alltimes", "SITEPERC_p04074"

# What to aggregate: "all" -> (num_all, den_all), "bnd" -> (num_bnd, den_bnd)
WHICH = "all"

# Common filters
MIN_AREA = 500
MAX_AREA = 7_500_000
MIN_PERIM: Optional[int] = None
MAX_PERIM: Optional[int] = None

# Site-perc filter (None => accept all present)
P_VALS: Optional[List[float]] = None

# PNG filters
# Time windows: list of ("HH:MM","HH:MM") in 24h; empty list => accept all times
TIME_WINDOWS: List[Tuple[str, str]] = []

# Threshold handling for PNG:
THRESHOLD_POLICY = "argmax_clouds"  # "all" or "argmax_clouds"

# When THRESHOLD_POLICY == "all": optional whitelist (None => accept all thresholds present)
THRESHOLDS_WHITELIST: Optional[List[float]] = None

# When THRESHOLD_POLICY == "argmax_clouds": behavior if meta JSON is missing/malformed
ON_MISSING_METADATA = "skip_run"    # "skip_run" (default) or "keep_all_thresholds"

# Roots (mirror FD)
PNG_PER_CLOUD_ROOT = "scratch/all_clouds_data/threshold_autocorr_bd/per_cloud"
PNG_META_ROOT      = "scratch/all_clouds_data/threshold_autocorr_bd"
SP_PER_CLOUD_ROOT  = "scratch/all_clouds_data/sp_autocorr_bd/per_cloud"

# Output directory (files will be prefixed by SUFFIX_TAG)
OUTPUT_DIR = "scratch/all_clouds_data/analysis/autocorr"

# Plot look
DPI = 300
FIGSIZE = (9.0, 7.0)

# Numerics
EPS = 1e-12
ATOL = 1e-12
RTOL = 1e-10

# --- Resolve to absolute paths so cwd doesn't matter ---
PNG_PER_CLOUD_ROOT = _abs(PNG_PER_CLOUD_ROOT)
PNG_META_ROOT      = _abs(PNG_META_ROOT)
SP_PER_CLOUD_ROOT  = _abs(SP_PER_CLOUD_ROOT)
OUTPUT_DIR         = _abs(OUTPUT_DIR)

# =========================================================

def _assert_source():
    if SOURCE not in {"png", "siteperc"}:
        raise ValueError('SOURCE must be "png" or "siteperc" (never "both").')
    if WHICH not in {"all", "bnd"}:
        raise ValueError('WHICH must be "all" or "bnd".')
    if SOURCE == "png" and THRESHOLD_POLICY not in {"all", "argmax_clouds"}:
        raise ValueError('For PNG, THRESHOLD_POLICY must be "all" or "argmax_clouds".')
    if SOURCE == "png" and THRESHOLD_POLICY == "argmax_clouds" and ON_MISSING_METADATA not in {"skip_run", "keep_all_thresholds"}:
        raise ValueError('ON_MISSING_METADATA must be "skip_run" or "keep_all_thresholds".')

def _ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _hm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})--(\d{2})-(\d{2})-(\d{2})")

def _parse_time_minutes_from_run_tag(run_tag: str) -> Optional[int]:
    """
    run_tag example: thr_autocorr_bd_2012-12-30--03-15-16--375
    Returns minutes since midnight or None if not parsable.
    """
    m = _TIME_RE.search(run_tag)
    if not m:
        return None
    HH, MM = int(m.group(4)), int(m.group(5))
    return HH*60 + MM

def _within_any_window(run_tag: str, windows: List[Tuple[str, str]]) -> bool:
    if not windows:
        return True
    minutes = _parse_time_minutes_from_run_tag(run_tag)
    if minutes is None:
        # If time windows specified but we can't parse time, exclude
        return False
    for start, end in windows:
        s = _hm_to_minutes(start)
        e = _hm_to_minutes(end)
        if s <= e:
            if s <= minutes < e:
                return True
        else:
            # Overnight window (e.g., 22:00–03:00)
            if minutes >= s or minutes < e:
                return True
    return False

# -------------------- Discovery (mirror FD) --------------------

def _discover_png_runs() -> List[str]:
    """
    Return list of PNG run_tags (folder names under PNG_PER_CLOUD_ROOT).
    """
    pattern = os.path.join(PNG_PER_CLOUD_ROOT, "*")
    runs = []
    for d in glob.iglob(pattern):
        if os.path.isdir(d):
            runs.append(os.path.basename(d))
    return sorted(runs)

def _discover_sp_parquets() -> List[str]:
    """
    Return list of site-perc parquet shard paths.
    """
    pattern = os.path.join(SP_PER_CLOUD_ROOT, "*", "cloud_metrics.part*.parquet")
    return sorted(glob.iglob(pattern))

def _png_meta_path(run_tag: str) -> str:
    return os.path.join(PNG_META_ROOT, f"{run_tag}_meta.json")

def _choose_argmax_threshold(run_tag: str) -> Optional[float]:
    """
    Read num_clouds_by_threshold from the meta JSON and pick argmax (tie -> smallest thr).
    Returns chosen threshold (float) or None if unavailable/malformed.
    """
    meta_path = _png_meta_path(run_tag)
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None
    d = meta.get("num_clouds_by_threshold")
    if not isinstance(d, dict) or not d:
        return None
    best_thr = None
    best_cnt = None
    for k, v in d.items():
        try:
            thr = float(k)
            cnt = int(v)
        except Exception:
            continue
        if best_cnt is None or cnt > best_cnt or (cnt == best_cnt and thr < best_thr):
            best_cnt = cnt
            best_thr = thr
    return best_thr

# -------------------- Filters (mirror FD) --------------------

def _apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    if "area" not in df.columns or "perim" not in df.columns:
        return df.iloc[0:0]
    df = df[(df["area"] > 0) & (df["perim"] > 0)]
    df = df[(df["area"] >= MIN_AREA) & (df["area"] <= MAX_AREA)]
    if MIN_PERIM is not None:
        df = df[df["perim"] >= MIN_PERIM]
    if MAX_PERIM is not None:
        df = df[df["perim"] <= MAX_PERIM]
    return df

def _load_cols(parquet_path: str, cols: List[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(parquet_path, columns=cols)
    except Exception:
        df = pd.read_parquet(parquet_path)
        keep = [c for c in cols if c in df.columns]
        return df[keep]

# -------------------- Aggregation --------------------

def _iter_num_den_rows(df: pd.DataFrame, which: str) -> Iterable[Tup[np.ndarray, np.ndarray]]:
    num_field = "num_all" if which == "all" else "num_bnd"
    den_field = "den_all" if which == "all" else "den_bnd"
    if num_field not in df.columns or den_field not in df.columns:
        return
    for _, row in df.iterrows():
        num = np.asarray(row[num_field], dtype=np.float64)
        den = np.asarray(row[den_field], dtype=np.float64)
        if num.size == 0 or den.size == 0:
            continue
        m = min(num.size, den.size)
        if m == 0:
            continue
        yield num[:m], den[:m]

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

# -------------------- Collectors (mirror FD logic) --------------------

def _collect_png_aggregates() -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, object]]:
    """
    Returns (agg_num, agg_den, counts_dict, meta_used) for PNG.
    Applies time windows, threshold_policy, thresholds_whitelist (if any).
    """
    counts = {"runs_considered": 0, "scanned_rows": 0, "kept_rows": 0, "runs_skipped_no_meta": 0}
    meta_used: Dict[str, object] = {}

    agg_num = np.array([], dtype=np.float64)
    agg_den = np.array([], dtype=np.float64)

    run_tags = _discover_png_runs()
    for run_tag in run_tags:
        if not _within_any_window(run_tag, TIME_WINDOWS):
            continue

        run_dir = os.path.join(PNG_PER_CLOUD_ROOT, run_tag)
        shard_paths = sorted(glob.iglob(os.path.join(run_dir, "cloud_metrics.part*.parquet")))
        if not shard_paths:
            continue

        chosen_thr: Optional[float] = None
        if THRESHOLD_POLICY == "argmax_clouds":
            chosen_thr = _choose_argmax_threshold(run_tag)
            if chosen_thr is None and ON_MISSING_METADATA == "skip_run":
                counts["runs_skipped_no_meta"] += 1
                continue
            meta_used[run_tag] = {"policy": "argmax_clouds", "chosen_thr": chosen_thr, "on_missing": ON_MISSING_METADATA}
        else:
            meta_used[run_tag] = {"policy": "all", "whitelist": THRESHOLDS_WHITELIST}

        kept_in_run = 0
        scanned_in_run = 0

        need_cols = ["area", "perim", "threshold", ("num_all" if WHICH=="all" else "num_bnd"),
                                           ("den_all" if WHICH=="all" else "den_bnd")]

        for pq in shard_paths:
            df = _load_cols(pq, need_cols)
            scanned_in_run += len(df)
            if df.empty:
                continue

            df = _apply_common_filters(df)
            if df.empty:
                continue

            # Threshold filtering
            if THRESHOLD_POLICY == "argmax_clouds":
                if chosen_thr is not None and "threshold" in df.columns:
                    df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=RTOL, atol=ATOL)]
            else:  # "all"
                if THRESHOLDS_WHITELIST is not None and "threshold" in df.columns:
                    df = df[df["threshold"].isin(THRESHOLDS_WHITELIST)]

            if df.empty:
                continue

            for num_arr, den_arr in _iter_num_den_rows(df, WHICH):
                agg_num = _dynamic_add(agg_num, num_arr)
                agg_den = _dynamic_add(agg_den, den_arr)
                kept_in_run += 1

        if kept_in_run > 0:
            counts["runs_considered"] += 1
            counts["scanned_rows"] += scanned_in_run
            counts["kept_rows"] += kept_in_run

    return agg_num, agg_den, counts, meta_used

def _collect_sp_aggregates() -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    counts = {"files_considered": 0, "scanned_rows": 0, "kept_rows": 0}
    agg_num = np.array([], dtype=np.float64)
    agg_den = np.array([], dtype=np.float64)

    need_cols = ["area", "perim", "p_val", ("num_all" if WHICH=="all" else "num_bnd"),
                                   ("den_all" if WHICH=="all" else "den_bnd")]

    for pq in _discover_sp_parquets():
        df = _load_cols(pq, need_cols)
        counts["files_considered"] += 1
        counts["scanned_rows"] += len(df)
        if df.empty:
            continue
        df = _apply_common_filters(df)
        if df.empty:
            continue
        if P_VALS is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(P_VALS)]
        if df.empty:
            continue

        for num_arr, den_arr in _iter_num_den_rows(df, WHICH):
            agg_num = _dynamic_add(agg_num, num_arr)
            agg_den = _dynamic_add(agg_den, den_arr)
            counts["kept_rows"] += 1

    return agg_num, agg_den, counts

# -------------------- Plotting --------------------

def _plot_series(x: np.ndarray, y: np.ndarray, scale: str, ylabel: str, out_png: str):
    plt.figure(figsize=FIGSIZE)
    if scale == "linear":
        mask = np.ones_like(y, dtype=bool)
        plt.xscale("linear"); plt.yscale("linear")
    elif scale == "semilog":
        mask = (y > 0)
        plt.xscale("linear"); plt.yscale("log")
    elif scale == "loglog":
        mask = (y > 0)
        plt.xscale("log"); plt.yscale("log")
    else:
        raise ValueError("Unknown scale")
    plt.plot(x[mask], y[mask])
    plt.xlabel("radius index (bin #)")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI)
    plt.close()

# -------------------- Main --------------------

def main():
    _assert_source()
    _ensure_outdir()

    if SOURCE == "png":
        agg_num, agg_den, counts, meta_used = _collect_png_aggregates()
        mode_descr = f'png[{THRESHOLD_POLICY}]/{WHICH}'
    else:
        agg_num, agg_den, counts = _collect_sp_aggregates()
        meta_used = {}
        mode_descr = f'siteperc/{WHICH}'

    if agg_num.size == 0:
        print("No arrays after filtering; nothing to aggregate.")
        print("Counts:", counts)
        if meta_used:
            print("Meta used:", meta_used)
        return

    # Align lengths (if needed)
    if agg_den.size < agg_num.size:
        tmp = np.zeros_like(agg_num)
        tmp[:agg_den.size] = agg_den
        agg_den = tmp

    r_index = np.arange(1, agg_num.size + 1, dtype=np.int64)
    agg_ratio = agg_num / np.maximum(agg_den, EPS)

    # File paths (prefix with SUFFIX_TAG)
    base = os.path.join(OUTPUT_DIR, SUFFIX_TAG)
    out_num_lin   = f"{base}__agg_num_linear.png"
    out_num_semi  = f"{base}__agg_num_semilog.png"
    out_num_log   = f"{base}__agg_num_loglog.png"
    out_rat_lin   = f"{base}__agg_ratio_linear.png"
    out_rat_semi  = f"{base}__agg_ratio_semilog.png"
    out_rat_log   = f"{base}__agg_ratio_loglog.png"
    out_curves    = f"{base}__curves.csv"
    out_metrics   = f"{base}__metrics.json"

    # Save arrays + CSV
    np.save(f"{base}__r_index.npy", r_index)
    np.save(f"{base}__agg_num.npy", agg_num)
    np.save(f"{base}__agg_den.npy", agg_den)
    np.save(f"{base}__agg_ratio.npy", agg_ratio)

    pd.DataFrame({
        "r_index": r_index,
        "agg_num": agg_num,
        "agg_den": agg_den,
        "agg_ratio": agg_ratio,
    }).to_csv(out_curves, index=False)

    # Plots
    _plot_series(r_index, agg_num,   "linear",  "aggregated numerator", out_num_lin)
    _plot_series(r_index, agg_num,   "semilog", "aggregated numerator", out_num_semi)
    _plot_series(r_index, agg_num,   "loglog",  "aggregated numerator", out_num_log)

    _plot_series(r_index, agg_ratio, "linear",  "aggregated num / aggregated den", out_rat_lin)
    _plot_series(r_index, agg_ratio, "semilog", "aggregated num / aggregated den", out_rat_semi)
    _plot_series(r_index, agg_ratio, "loglog",  "aggregated num / aggregated den", out_rat_log)

    # Metrics JSON (FD-style)
    metrics = {
        "source": SOURCE,
        "suffix_tag": SUFFIX_TAG,
        "mode_descr": mode_descr,
        "which": WHICH,
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
        "outputs": {
            "plots": [out_num_lin, out_num_semi, out_num_log, out_rat_lin, out_rat_semi, out_rat_log],
            "arrays": [f"{base}__r_index.npy", f"{base}__agg_num.npy", f"{base}__agg_den.npy", f"{base}__agg_ratio.npy"],
            "curves_csv": out_curves,
            "metrics_json": out_metrics,
        },
        "notes": "Radius = list index (1-based). Log plots omit non-positive y.",
    }

    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary
    print("-" * 60)
    print(" Autocorr aggregates — strict source, FD-style")
    print("-" * 60)
    print("Saved curves:", out_curves)
    print("Saved metrics:", out_metrics)
    print("Counts:", counts)

# -------------------- CLI overrides + entrypoint --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate autocorr numerators and ratio (FD-style filters).")
    parser.add_argument("--source", choices=["png", "siteperc"], help="Data source: png or siteperc.")
    parser.add_argument("--threshold_policy", choices=["all", "argmax_clouds"],
                        help="For PNG only: threshold selection policy.")
    parser.add_argument("--time_window",
                        help="Single time window 'HH:MM-HH:MM' or 'all' (for all times).")
    parser.add_argument("--suffix_tag", help="Suffix tag for output filenames (required).")
    parser.add_argument("--p_vals", help="Site-perc p-values, comma-separated or 'all'.")
    parser.add_argument("--which", choices=["all","bnd"], help="Aggregate (num_all,den_all) or (num_bnd,den_bnd).")
    args = parser.parse_args()

    # Apply overrides (mirror FD)
    if args.source:
        SOURCE = args.source
    if args.threshold_policy:
        THRESHOLD_POLICY = args.threshold_policy
    if args.time_window:
        if args.time_window.lower() == "all":
            TIME_WINDOWS = []  # no filter = all times
        else:
            start, end = args.time_window.split("-")
            TIME_WINDOWS = [(start.strip(), end.strip())]
    if args.suffix_tag:
        SUFFIX_TAG = args.suffix_tag
    if args.p_vals:
        if args.p_vals.lower() == "all":
            P_VALS = None
        else:
            P_VALS = [float(x) for x in args.p_vals.split(",")]
    if args.which:
        WHICH = args.which

    main()
