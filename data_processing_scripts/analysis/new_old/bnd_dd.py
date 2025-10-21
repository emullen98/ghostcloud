#!/usr/bin/env python3
from __future__ import annotations
import os, json, math, glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Set
import numpy as np
import pandas as pd

from clouds.utils.analysis_utils import *

# Config (overridable)
SOURCE = "siteperc"   # "png" | "siteperc"
PREFIX = "SITEPERC_all"
# paths
PNG_PER_CLOUD_ROOT = "scratch/all_clouds_data/threshold_autocorr_bd/per_cloud"
PNG_META_ROOT      = "scratch/all_clouds_data/threshold_autocorr_bd"
SP_PER_CLOUD_ROOT  = "scratch/all_clouds_data/sp_autocorr_bd/per_cloud"
OUTPUT_DIR         = "scratch/all_clouds_data/analysis/bd_distribution"
# filters
MIN_AREA = 500
MAX_AREA = 7_500_000

# PNG knobs
TIME_WINDOWS: List[Tuple[str,str]] = []
THRESHOLD_POLICY = "argmax_clouds"
THRESHOLDS_WHITELIST: Optional[List[float]] = None
ON_MISSING_METADATA = "skip_run"

# SP knobs
P_VALS: Optional[List[float]] = None

# Grid knobs
U_SPACING = "log"    # "log" | "linear"
U_NBINS = 100
U_MARGIN_FRAC = 0.05
SPLAT = "linear"     # "linear" | "nearest"
MIN_CLOUDS_PER_BIN = 10

# resolve abs
PNG_PER_CLOUD_ROOT = abs_path(PNG_PER_CLOUD_ROOT)
PNG_META_ROOT      = abs_path(PNG_META_ROOT)
SP_PER_CLOUD_ROOT  = abs_path(SP_PER_CLOUD_ROOT)
OUTPUT_DIR         = abs_path(OUTPUT_DIR)

def _area_ok(a) -> bool:
    try:
        aa = float(a)
    except Exception:
        return False
    return (aa >= MIN_AREA) and (aa <= MAX_AREA)

def _iter_png_rows():
    run_tags = discover_png_runs(PNG_PER_CLOUD_ROOT)
    for run_tag in run_tags:
        if not within_any_window(run_tag, TIME_WINDOWS): continue
        shards = sorted(glob.iglob(os.path.join(PNG_PER_CLOUD_ROOT, run_tag, "cloud_metrics.part*.parquet")))
        if not shards: continue
        chosen_thr = None
        if THRESHOLD_POLICY == "argmax_clouds":
            chosen_thr = choose_argmax_threshold(PNG_META_ROOT, run_tag)
            if chosen_thr is None and ON_MISSING_METADATA == "skip_run":
                continue
        for pq in shards:
            df = pd.read_parquet(pq)
            if "threshold" in df.columns:
                if THRESHOLD_POLICY == "argmax_clouds":
                    if chosen_thr is not None:
                        df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=1e-10, atol=1e-12)]
                elif THRESHOLD_POLICY == "all":
                    if THRESHOLDS_WHITELIST is not None:
                        df = df[df["threshold"].isin(THRESHOLDS_WHITELIST)]
            for _, row in df.iterrows():
                yield row

def _iter_sp_rows():
    for pq in discover_sp_parquets(SP_PER_CLOUD_ROOT):
        df = pd.read_parquet(pq)
        if P_VALS is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(P_VALS)]
        for _, row in df.iterrows():
            yield row

def _extract_bd_row(row: pd.Series):
    # area filter
    if "area" not in row or not _area_ok(row["area"]): return None
    # Rg_area
    R = None
    if "rg_area" in row and pd.notna(row["rg_area"]): R = float(row["rg_area"])
    elif "Rg_area" in row and pd.notna(row["Rg_area"]): R = float(row["Rg_area"])
    else: return None
    # arrays
    if "bd_r" in row and isinstance(row["bd_r"], (list, np.ndarray)):
        r = np.asarray(row["bd_r"], dtype=np.float64)
    elif "rp_r" in row and isinstance(row["rp_r"], (list, np.ndarray)):
        r = np.asarray(row["rp_r"], dtype=np.float64)
    else:
        return None
    if "bd_counts" in row and isinstance(row["bd_counts"], (list, np.ndarray)):
        c = np.asarray(row["bd_counts"], dtype=np.float64)
    elif "rp_counts" in row and isinstance(row["rp_counts"], (list, np.ndarray)):
        c = np.asarray(row["rp_counts"], dtype=np.float64)
    else:
        return None
    if r.size == 0 or c.size == 0 or r.size != c.size: return None
    # widths and totals
    if "bd_bin_width" in row and pd.notna(row["bd_bin_width"]):
        dr = float(row["bd_bin_width"])
    else:
        dif = np.diff(r)
        if dif.size == 0: return None
        dr = float(np.median(dif))
    if "bd_n" in row and pd.notna(row["bd_n"]):
        Nb = float(row["bd_n"])
    else:
        Nb = float(np.sum(c))
    if not (np.isfinite(R) and R>0 and np.isfinite(dr) and dr>0 and np.isfinite(Nb) and Nb>0):
        return None
    return R, r, c, dr, Nb

def _discover_u_bounds(rows_iter):
    umin = np.inf; umax = -np.inf
    scanned = 0; kept = 0; skipped = 0
    for row in rows_iter:
        scanned += 1
        ext = _extract_bd_row(row)
        if ext is None:
            skipped += 1; continue
        R, r, c, dr, Nb = ext
        u = r / R
        mask = np.isfinite(u) & (u>0) & np.isfinite(c) & (c>0)
        if not np.any(mask):
            skipped += 1; continue
        umin = min(umin, float(np.min(u[mask])))
        umax = max(umax, float(np.max(u[mask])))
        kept += 1
    return umin, umax, {"scanned_rows": scanned, "kept_rows": kept, "skipped_bad_rows": skipped}

def _build_u_grid(umin_obs: float, umax_obs: float, spacing: str, nbins: int, margin_frac: float):
    eps = 1e-12
    if not np.isfinite(umin_obs) or not np.isfinite(umax_obs) or umax_obs <= max(umin_obs, eps):
        raise ValueError("Insufficient valid u-range discovered.")
    if spacing == "log":
        log_min = math.log(max(umin_obs, eps))
        log_max = math.log(umax_obs)
        pad = margin_frac * (log_max - log_min if log_max > log_min else 1.0)
        log_min -= pad; log_max += pad
        edges = np.exp(np.linspace(log_min, log_max, nbins + 1))
        centers = np.sqrt(edges[:-1]*edges[1:])
    else:
        span = umax_obs - umin_obs
        pad = margin_frac * (span if span > 0 else 1.0)
        umin = max(umin_obs - pad, eps)
        umax = umax_obs + pad
        edges = np.linspace(umin, umax, nbins + 1)
        centers = 0.5*(edges[:-1]+edges[1:])
    widths = edges[1:] - edges[:-1]
    return edges, centers, widths

def _splat(u_vals: np.ndarray, w_vals: np.ndarray, edges: np.ndarray, spacing: str,
           mode: str, sums: np.ndarray, covered: Set[int]):
    eps = 1e-12
    if u_vals.size == 0: return
    if spacing == "log":
        logu = np.log(np.clip(u_vals, eps, None))
        log_edges = np.log(np.clip(edges, eps, None))
        d = log_edges[1]-log_edges[0]
        f = (logu - log_edges[0]) / d
    else:
        d = edges[1]-edges[0]
        f = (u_vals - edges[0]) / d
    i0 = np.floor(f).astype(int)
    frac = f - i0
    i1 = i0 + 1
    in0 = (i0 >= 0) & (i0 < sums.size)
    in1 = (i1 >= 0) & (i1 < sums.size)
    if mode == "nearest":
        idx = np.clip(np.where(frac <= 0.5, i0, i1), 0, sums.size - 1)
        np.add.at(sums, idx, w_vals)
        for j in np.unique(idx): covered.add(int(j))
    else:
        if np.any(in0):
            np.add.at(sums, i0[in0], w_vals[in0]*(1.0-frac[in0]))
            for j in np.unique(i0[in0]): covered.add(int(j))
        if np.any(in1):
            np.add.at(sums, i1[in1], w_vals[in1]*frac[in1])
            for j in np.unique(i1[in1]): covered.add(int(j))

def main():
    ensure_dir(OUTPUT_DIR)

    # First pass: bounds
    if SOURCE=="png":
        umin, umax, counts = _discover_u_bounds(_iter_png_rows())
        mode_descr = f"png[{THRESHOLD_POLICY}]"
    else:
        umin, umax, counts = _discover_u_bounds(_iter_sp_rows())
        mode_descr = "siteperc"

    edges, centers, widths = _build_u_grid(umin, umax, U_SPACING, U_NBINS, U_MARGIN_FRAC)

    # Second pass: accumulate
    nb = widths.size
    sums_len = np.zeros(nb, dtype=np.float64)
    sums_eq  = np.zeros(nb, dtype=np.float64)
    cov_len  = np.zeros(nb, dtype=np.int64)
    cov_eq   = np.zeros(nb, dtype=np.int64)
    n_clouds = 0

    rows_iter = _iter_png_rows() if SOURCE=="png" else _iter_sp_rows()

    for row in rows_iter:
        ext = _extract_bd_row(row)
        if ext is None: continue
        R, r, c, dr, Nb = ext
        u = r / R
        mask = np.isfinite(u) & (u>0) & np.isfinite(c) & (c>0)
        if not np.any(mask): continue
        u = u[mask]; w = c[mask]
        factor = (R / dr)
        w_len = w * factor
        w_eq  = (w / Nb) * factor

        touched = set()
        _splat(u, w_len, edges, U_SPACING, SPLAT, sums_len, touched)
        if touched: cov_len[list(touched)] += 1

        touched2 = set()
        _splat(u, w_eq, edges, U_SPACING, SPLAT, sums_eq, touched2)
        if touched2: cov_eq[list(touched2)] += 1

        n_clouds += 1

    total_len = float(np.sum(sums_len))
    total_eq  = float(np.sum(sums_eq))
    dens_len = (sums_len/total_len)/widths if total_len>0 else np.zeros_like(sums_len)
    dens_eq  = (sums_eq/total_eq)/widths if total_eq>0 else np.zeros_like(sums_eq)

    if n_clouds == 0 or (dens_len.sum()==0 and dens_eq.sum()==0):
        print("[WARN] No valid clouds after filtering; nothing to write.")
        print(json.dumps(counts))
        return

    base = os.path.join(OUTPUT_DIR, PREFIX)
    bdd_len_csv = f"{base}__bdd_len.csv"
    bdd_eq_csv  = f"{base}__bdd_eq.csv"
    metrics_json = f"{base}__bdd_metrics.json"

    pd.DataFrame({"u": centers, "y": dens_len, "coverage": cov_len, "width": widths}).to_csv(bdd_len_csv, index=False)
    pd.DataFrame({"u": centers, "y": dens_eq,  "coverage": cov_eq,  "width": widths}).to_csv(bdd_eq_csv,  index=False)

    metrics = {
        "source": SOURCE,
        "prefix": PREFIX,
        "u_grid": {"spacing": U_SPACING, "nbins": int(U_NBINS), "margin_frac": float(U_MARGIN_FRAC),
                   "u_min_obs": float(umin), "u_max_obs": float(umax)},
        "splat": SPLAT,
        "min_clouds_per_bin": int(MIN_CLOUDS_PER_BIN),
        "counts": counts,
        "totals": {"length_weighted": total_len, "equal_cloud": total_eq, "n_clouds": int(n_clouds)},
        "outputs": {"bdd_len_csv": bdd_len_csv, "bdd_eq_csv": bdd_eq_csv, "metrics_json": metrics_json},
        "notes": "BDD aggregator is data-only. Plot via plotter.py.",
    }
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] BDD aggregates written:")
    print(bdd_len_csv)
    print(bdd_eq_csv)
    print(metrics_json)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate BDD (data-only CSVs).")
    ap.add_argument("--source", choices=["png","siteperc"])
    ap.add_argument("--prefix")
    ap.add_argument("--time_window")
    ap.add_argument("--threshold_policy", choices=["all","argmax_clouds"])
    ap.add_argument("--p_vals", help="comma list or 'all'")
    ap.add_argument("--u_spacing", choices=["log","linear"])
    ap.add_argument("--u_nbins", type=int)
    ap.add_argument("--u_margin_frac", type=float)
    ap.add_argument("--splat", choices=["linear","nearest"])
    args = ap.parse_args()

    if args.source: SOURCE = args.source
    if args.prefix: PREFIX = args.prefix
    if args.time_window:
        if args.time_window.lower()=="all": TIME_WINDOWS=[]
        else:
            s,e = [x.strip() for x in args.time_window.split("-")]
            TIME_WINDOWS=[(s,e)]
    if args.threshold_policy: THRESHOLD_POLICY = args.threshold_policy
    if args.p_vals:
        if args.p_vals.lower()=="all": P_VALS = None
        else: P_VALS = [float(x) for x in args.p_vals.split(",")]
    if args.u_spacing: U_SPACING = args.u_spacing
    if args.u_nbins: U_NBINS = int(args.u_nbins)
    if args.u_margin_frac is not None: U_MARGIN_FRAC = float(args.u_margin_frac)
    if args.splat: SPLAT = args.splat

    main()
