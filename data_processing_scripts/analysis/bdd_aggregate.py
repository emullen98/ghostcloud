#!/usr/bin/env python3
"""
bdd_aggregate.py — Single-run Boundary Distance Distribution (width-aware)

Behavior (unchanged externally):
  - READ config (common schema, 3-root paths).
  - FILTER rows (area/perim; plus PNG/siteperc selectors).
  - USE bd_r, bd_counts, rg_area from CloudRow (tolerant to rp_* / Rg_area).
  - BUILD a common U-grid with (U_SPACING, U_NBINS, U_MARGIN_FRAC).
  - ACCUMULATE per-bin mass using explicit bin EDGES and WIDTHS.
  - CONVERT to PDF: density_i = (mass_i / total_mass) / width_i, then renormalize.
  - WEIGHTING: 'equal' (equal-cloud mixture) OR 'by_counts' (length/counts-weighted).
  - WRITE bdd_{TAG}.csv (r_norm, density), diagnostics json, resolved config, run-info.

Notes:
  - Serialized; no parallelism.
  - CSV schema preserved: columns = ["r_norm","density"] (centers + density).
"""

from __future__ import annotations
import argparse, json, time, math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from clouds.utils.analysis_utils import *  # type: ignore


# ----------------------------- Row Iterators -----------------------------

def _iter_png_rows(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield filtered rows with bd_r, bd_counts, rg_area from PNG per-cloud runs."""
    paths = cfg["paths"]; flt = cfg["filters"]
    runs = discover_png_runs(paths["png_per_cloud_root"])

    windows: List[Tuple[str, str]] = []
    if flt.get("time_windows"):
        for w in flt["time_windows"]:
            if isinstance(w, str) and "-" in w:
                s, e = [x.strip() for x in w.split("-")]
                windows.append((s, e))

    for run_tag in runs:
        if windows and not within_any_window(run_tag, windows):
            continue
        run_dir = Path(paths["png_per_cloud_root"]) / run_tag
        for pq in sorted(run_dir.glob("cloud_metrics.part*.parquet")):
            # be permissive on columns; we will pick what we need
            need = ["area", "perim", "bd_r", "bd_counts", "rg_area",
                    "rp_r", "rp_counts", "Rg_area", "bd_bin_width", "bd_n"]
            df = read_parquet_cols(str(pq), need)
            if df.empty:
                continue
            df = apply_common_filters(
                df,
                flt.get("min_area"), flt.get("max_area"),
                flt.get("min_perim"), flt.get("max_perim"),
            )
            if df.empty:
                continue
            for _, row in df.iterrows():
                yield row.to_dict()

def _iter_siteperc_rows(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield filtered rows with bd_r, bd_counts, rg_area from site-perc per-cloud runs."""
    paths = cfg["paths"]; flt = cfg["filters"]
    for pq in discover_sp_parquets(paths["siteperc_per_cloud_root"]):
        need = ["area", "perim", "bd_r", "bd_counts", "rg_area",
                "rp_r", "rp_counts", "Rg_area", "bd_bin_width", "bd_n", "p_val"]
        df = read_parquet_cols(pq, need)
        if df.empty:
            continue
        df = apply_common_filters(
            df,
            flt.get("min_area"), flt.get("max_area"),
            flt.get("min_perim"), flt.get("max_perim"),
        )
        if df.empty:
            continue
        if flt.get("p_vals") is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(flt["p_vals"])]
        if df.empty:
            continue
        for _, row in df.iterrows():
            yield row.to_dict()


# ----------------------------- Grid Builders -----------------------------

def _make_u_grid_with_widths(bdd_cfg: Dict[str, Any], umin_obs: float, umax_obs: float):
    """
    Construct edges, centers, widths for u-grid (u = r/rg_area), with margin.
    Log spacing pads in log-space; centers are geometric means for log grids.
    """
    spacing = bdd_cfg.get("U_SPACING", "log")
    nbins = int(bdd_cfg.get("U_NBINS", 100))
    margin = float(bdd_cfg.get("U_MARGIN_FRAC", 0.05))
    eps = 1e-12

    if not np.isfinite(umin_obs) or not np.isfinite(umax_obs) or umax_obs <= max(umin_obs, eps):
        raise ValueError("Insufficient valid u-range discovered.")

    if spacing == "log":
        lo = math.log(max(umin_obs, eps))
        hi = math.log(max(umax_obs, umin_obs*(1+1e-12)))
        pad = margin * max(hi - lo, 1.0)
        lo -= pad; hi += pad
        edges = np.exp(np.linspace(lo, hi, nbins + 1))
        centers = np.sqrt(edges[:-1] * edges[1:])  # geometric-mean centers
    else:
        span = max(umax_obs - umin_obs, eps)
        pad = margin * span
        umin = max(umin_obs - pad, eps)
        umax = umax_obs + pad
        edges = np.linspace(umin, umax, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    widths = edges[1:] - edges[:-1]
    return edges, centers, widths


# ----------------------------- Depositing Logic -----------------------------

def _splat_with_edges(u_vals: np.ndarray,
                      w_vals: np.ndarray,
                      edges: np.ndarray,
                      spacing: str,
                      mode: str,
                      sums: np.ndarray,
                      touched: Set[int]) -> None:
    """
    Splat weights into adjacent bins defined by EDGES (width-aware).
    - 'linear': split between floor and ceil bins by fractional offset.
    - 'nearest': deposit into nearest single bin.
    """
    eps = 1e-12
    if u_vals.size == 0:
        return

    if spacing == "log":
        logu = np.log(np.clip(u_vals, eps, None))
        log_edges = np.log(np.clip(edges, eps, None))
        binw = log_edges[1] - log_edges[0]
        f = (logu - log_edges[0]) / binw
    else:
        binw = edges[1] - edges[0]
        f = (u_vals - edges[0]) / binw

    i0 = np.floor(f).astype(int)
    frac = f - i0
    i1 = i0 + 1

    in0 = (i0 >= 0) & (i0 < sums.size)
    in1 = (i1 >= 0) & (i1 < sums.size)

    if mode == "nearest":
        idx = np.clip(np.where(frac <= 0.5, i0, i1), 0, sums.size - 1)
        np.add.at(sums, idx, w_vals)
        for j in np.unique(idx):
            touched.add(int(j))
    else:
        if np.any(in0):
            np.add.at(sums, i0[in0], w_vals[in0] * (1.0 - frac[in0]))
            for j in np.unique(i0[in0]):
                touched.add(int(j))
        if np.any(in1):
            np.add.at(sums, i1[in1], w_vals[in1] * frac[in1])
            for j in np.unique(i1[in1]):
                touched.add(int(j))


def _to_density(sums: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Convert per-bin mass to a PDF per unit-u using widths, and renormalize."""
    total = float(np.sum(sums))
    if total <= 0:
        return np.zeros_like(sums)
    dens = (sums / total) / widths
    # numerical polish: re-normalize so sum(dens * widths) == 1
    area = float(np.sum(dens * widths))
    if area > 0:
        dens = dens / area
    return dens


# ----------------------------- Helpers -----------------------------

def _extract_arrays_from_row(row: Dict[str, Any]) -> Optional[Tuple[float, np.ndarray, np.ndarray, float, float]]:
    """
    Pull rg_area (R), r-array, count-array, dr (bin width in r), Nb (total counts).
    Permissive on column names (bd_* or rp_*; rg_area or Rg_area).
    Returns (R, r, c, dr, Nb) or None.
    """
    # R (rg_area)
    R = row.get("rg_area", row.get("Rg_area", None))
    try:
        R = float(R)
    except Exception:
        return None
    if not (np.isfinite(R) and R > 0):
        return None

    # r, c
    r = row.get("bd_r", row.get("rp_r", None))
    c = row.get("bd_counts", row.get("rp_counts", None))
    if r is None or c is None:
        return None
    r_arr = np.asarray(r, dtype=np.float64).ravel()
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    L = min(r_arr.size, c_arr.size)
    if L == 0:
        return None
    r_arr = r_arr[:L]; c_arr = c_arr[:L]

    # dr, Nb (fallbacks if missing)
    dr = row.get("bd_bin_width", None)
    Nb = row.get("bd_n", None)
    try:
        dr = float(dr) if dr is not None else float(np.median(np.diff(r_arr))) if r_arr.size >= 2 else np.nan
    except Exception:
        dr = np.nan
    try:
        Nb = float(Nb) if Nb is not None else float(np.sum(c_arr))
    except Exception:
        Nb = np.nan

    if not (np.isfinite(dr) and dr > 0 and np.isfinite(Nb) and Nb > 0):
        return None
    return float(R), r_arr, c_arr, float(dr), float(Nb)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="BDD aggregator (width-aware PDF output)")
    ap.add_argument("-c", "--config", required=True, help="Path to per-run config (YAML/JSON)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = validate_config(load_yaml(args.config))
    tag = build_tag(cfg)
    arts = expand_artifacts(cfg, tag)

    out_dir = Path(cfg["paths"]["output_root"])
    ensure_dir(str(out_dir))
    write_resolved_config(cfg, out_dir)

    bdd_cfg = cfg["bdd"]
    weighting = str(bdd_cfg.get("weighting", "equal")).lower()           # 'equal' | 'by_counts'
    splat_mode = str(bdd_cfg.get("SPLAT", "linear")).lower()             # 'linear' | 'nearest'
    spacing = str(bdd_cfg.get("U_SPACING", "log")).lower()

    # ---------------- Pass 1: determine global u-range ----------------
    umins: List[float] = []
    umaxs: List[float] = []
    n_scanned = 0
    n_pass_filters = 0

    source = cfg["source"]
    iter_rows = _iter_png_rows if source == "png" else _iter_siteperc_rows

    for row in iter_rows(cfg):
        n_scanned += 1
        # apply area/perim filters already handled in iterators; count as pass
        n_pass_filters += 1
        ext = _extract_arrays_from_row(row)
        if ext is None:
            continue
        R, r_arr, c_arr, dr, Nb = ext
        if r_arr.size == 0 or c_arr.size == 0:
            continue
        u = r_arr / R
        mask = np.isfinite(u) & (u > 0) & np.isfinite(c_arr) & (c_arr > 0)
        if not np.any(mask):
            continue
        umins.append(float(np.min(u[mask])))
        umaxs.append(float(np.max(u[mask])))

    if not umins:
        # Nothing to do
        diag = {
            "metric": "bdd",
            "tag": tag,
            "n_scanned": int(n_scanned),
            "n_pass_filters": int(n_pass_filters),
            "n_used": 0,
            "warnings": ["No valid BDD inputs after filters."],
            "provenance": {"resolved_config_path": str(out_dir / "resolved_config.yaml")},
        }
        with open(out_dir / arts["diagnostics_json"], "w") as f:
            json.dump(diag, f, indent=2, sort_keys=True)

        t1 = time.time()
        summary = {
            "n_scanned": int(n_scanned),
            "n_pass_filters": int(n_pass_filters),
            "n_used": 0,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
            "duration_sec": float(t1 - t0),
        }
        write_run_info(cfg, tag, {
            "csv": arts.get("csv"),
            "diagnostics_json": arts.get("diagnostics_json"),
            "run_info_json": arts.get("run_info_json"),
        }, out_dir, status="failed", summary=summary, diagnostics_path=str(out_dir / arts["diagnostics_json"]))
        return

    umin = max(1e-12, float(np.min(umins)))
    umax = float(np.max(umaxs))
    edges, centers, widths = _make_u_grid_with_widths(bdd_cfg, umin, umax)

    # ---------------- Pass 2: accumulate per-bin mass ----------------
    nb = widths.size
    sums_len = np.zeros(nb, dtype=float)  # "by_counts" path (length/counts-weighted)
    sums_eq  = np.zeros(nb, dtype=float)  # "equal" path (equal-cloud mixture)
    cov_len  = np.zeros(nb, dtype=int)
    cov_eq   = np.zeros(nb, dtype=int)
    n_used = 0

    for row in iter_rows(cfg):
        ext = _extract_arrays_from_row(row)
        if ext is None:
            continue
        R, r_arr, c_arr, dr, Nb = ext

        u = r_arr / R
        mask = np.isfinite(u) & (u > 0) & np.isfinite(c_arr) & (c_arr > 0)
        if not np.any(mask):
            continue
        u = u[mask]; w = c_arr[mask]

        # Convert raw counts to per-u mass (both paths), like the second script:
        factor = (R / dr)            # dr is bin width in r; R rescales to u=r/R
        w_len = w * factor           # length/“counts”-weighted (pool then normalize)
        w_eq  = (w / Nb) * factor    # equal-cloud: normalize each cloud first, then mix

        touched = set()
        _splat_with_edges(u, w_len, edges, spacing, splat_mode, sums_len, touched)
        for j in touched:
            cov_len[j] += 1

        touched2 = set()
        _splat_with_edges(u, w_eq, edges, spacing, splat_mode, sums_eq, touched2)
        for j in touched2:
            cov_eq[j] += 1

        n_used += 1

    # ---------------- Convert to PDF and write single CSV ----------------
    dens_len = _to_density(sums_len, widths)
    dens_eq  = _to_density(sums_eq,  widths)

    if weighting == "by_counts":
        Y_pdf = dens_len
        coverage = cov_len
        density_mode = "by_counts"
    else:
        Y_pdf = dens_eq
        coverage = cov_eq
        density_mode = "equal"

    # Write CSV with preserved schema
    out_csv_path = out_dir / arts["csv"]
    pd.DataFrame({"r_norm": centers, "density": Y_pdf}).to_csv(out_csv_path, index=False)

    # ---------------- Diagnostics and run-info ----------------
    min_cov = int(bdd_cfg.get("MIN_CLOUDS_PER_BIN", 10))
    low_cov_bins = int(np.sum(coverage < min_cov))

    diag = {
        "metric": "bdd",
        "tag": tag,
        "n_scanned": int(n_scanned),
        "n_pass_filters": int(n_pass_filters),
        "n_used": int(n_used),
        "r_norm_min_realized": float(centers[0]),
        "r_norm_max_realized": float(centers[-1]),
        "U_SPACING": bdd_cfg.get("U_SPACING", "log"),
        "U_NBINS": int(bdd_cfg.get("U_NBINS", 100)),
        "U_MARGIN_FRAC": float(bdd_cfg.get("U_MARGIN_FRAC", 0.05)),
        "SPLAT": bdd_cfg.get("SPLAT", "linear"),
        "weighting": density_mode,
        "density_normalization": "pdf_width_aware",
        "coverage": {
            "MIN_CLOUDS_PER_BIN": min_cov,
            "bins_below_min": low_cov_bins
        },
        "widths": {
            "min": float(np.min(widths)),
            "max": float(np.max(widths)),
            "median": float(np.median(widths))
        },
        "provenance": {"resolved_config_path": str(out_dir / "resolved_config.yaml")},
        "warnings": [],
    }
    with open(out_dir / arts["diagnostics_json"], "w") as f:
        json.dump(diag, f, indent=2, sort_keys=True)

    t1 = time.time()
    summary = {
        "n_scanned": int(n_scanned),
        "n_pass_filters": int(n_pass_filters),
        "n_used": int(n_used),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
        "duration_sec": float(t1 - t0),
    }
    write_run_info(cfg, tag, {
        "csv": arts.get("csv"),
        "diagnostics_json": arts.get("diagnostics_json"),
        "run_info_json": arts.get("run_info_json"),
    }, out_dir, status="success", summary=summary, diagnostics_path=str(out_dir / arts["diagnostics_json"]))


if __name__ == "__main__":
    main()
