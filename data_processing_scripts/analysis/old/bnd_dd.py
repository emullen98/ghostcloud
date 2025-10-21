#!/usr/bin/env python3
"""
Boundary Distance Distribution (BDD) — scale-free aggregate across clouds.

We build one global curve of boundary-distance *density per unit-u* vs
    u = r / Rg_area
aggregated across all clouds that pass filters, with two weighting modes:
  (1) length-weighted  (raw boundary length pooling) and
  (2) equal-cloud      (each cloud contributes equally).

- Strict source separation: SOURCE ∈ {"png","siteperc"} (never both).
- PNG filters: time_windows (list), threshold_policy ("all" | "argmax_clouds"),
               thresholds_whitelist (optional with "all"),
               on_missing_metadata ("skip_run" | "keep_all_thresholds").
- Site-perc filters: p_vals (list or None).
- Geometry filters (common): MIN_AREA ≤ area ≤ MAX_AREA.
- Required per-cloud columns:
    rg_area: float             # R_eff
    bd_r: np.ndarray           # r-bin centers (px, from COM)  [falls back to rp_r]
    bd_counts: np.ndarray      # boundary pixels per bin (raw) [falls back to rp_counts]
    bd_bin_width: float        # Δr in pixels used for bd histogram (inferred if absent)
    bd_n: int                  # total boundary pixels (sum bd_counts; inferred if absent)

Change of variables:
    u = r / Rg_area, Δu = Δr / Rg_area
Contributions per sample i:
    length-weighted:     w_i = bd_counts[i] / Δu = bd_counts[i] * (Rg_area / Δr)
    equal-cloud weighted: w_i = (bd_counts[i] / bd_n) / Δu = (bd_counts[i] / bd_n) * (Rg_area / Δr)

Alignment (minimal binning):
    Single shared u-grid discovered from data and padded slightly.
    Default: log-spaced grid with NBINS bins; linear 2-bin "splat" to avoid aliasing.
    Final densities are normalized by total mass *and* by the bin widths in u.

Outputs (under OUTPUT_DIR, filenames prefixed by SUFFIX_TAG):
  - <SUFFIX>__bdd_lenweighted.png               (log–log)
  - <SUFFIX>__bdd_equalcloud.png                (log–log)
  - <SUFFIX>__bdd_lenweighted_linear.png        (linear–linear)
  - <SUFFIX>__bdd_equalcloud_linear.png         (linear–linear)
  - <SUFFIX>__bdd_len.csv                       (plot-ready data; u,y,coverage,width)
  - <SUFFIX>__bdd_eq.csv                        (plot-ready data; u,y,coverage,width)
  - <SUFFIX>__bdd_metrics.json

Requires: pandas, numpy, pyarrow (for parquet), matplotlib
"""

from __future__ import annotations
import os, re, glob, json, math, textwrap
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Path helpers (absolute) ----------------
def _abs(p: str) -> str:
    return str(Path(p).expanduser().resolve())

# ===================== CONFIG (EDITABLE DEFAULTS) =====================

# Choose exactly one
SOURCE = "siteperc"        # "png" or "siteperc" (NEVER "both")

# Suffix tag to distinguish runs in filenames
SUFFIX_TAG = "SITEPERC_p04074"

# Common geometry filters (match FD script defaults)
MIN_AREA = 500
MAX_AREA = 7_500_000
# (Perimeter filters can be added similarly if needed.)

# PNG filters
TIME_WINDOWS: List[Tuple[str, str]] = []      # list of ("HH:MM","HH:MM"); [] => all
THRESHOLD_POLICY = "argmax_clouds"            # "all" | "argmax_clouds"
THRESHOLDS_WHITELIST: Optional[List[float]] = None
ON_MISSING_METADATA = "skip_run"              # if argmax_clouds

# Site-perc filters (None => accept all present)
P_VALS: Optional[List[float]] = None

# Roots (mirror your FD script)
PNG_PER_CLOUD_ROOT = "scratch/all_clouds_data/threshold_autocorr_bd/per_cloud"
PNG_META_ROOT      = "scratch/all_clouds_data/threshold_autocorr_bd"
SP_PER_CLOUD_ROOT  = "scratch/all_clouds_data/sp_autocorr_bd/per_cloud"

# Output directory
OUTPUT_DIR = "scratch/all_clouds_data/analysis/bd_distribution"

# U-grid knobs (kept lean)
U_SPACING = "log"            # "log" or "linear"
U_NBINS = 100
U_MARGIN_FRAC = 0.05         # 5% padding in axis space
SPLAT = "linear"             # "linear" (2-bin spread) or "nearest"
MIN_CLOUDS_PER_BIN = 10      # coverage cue only (doesn't change sums)

# Plot look
FIGSIZE = (9.0, 7.0)
DPI = 300

# --- Resolve to absolute paths so cwd doesn't matter ---
PNG_PER_CLOUD_ROOT = _abs(PNG_PER_CLOUD_ROOT)
PNG_META_ROOT      = _abs(PNG_META_ROOT)
SP_PER_CLOUD_ROOT  = _abs(SP_PER_CLOUD_ROOT)
OUTPUT_DIR         = _abs(OUTPUT_DIR)

# =====================================================================

# ---------------- Utilities ----------------
def _assert_source():
    if SOURCE not in {"png", "siteperc"}:
        raise ValueError('SOURCE must be "png" or "siteperc".')
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
        return False
    for start, end in windows:
        s = _hm_to_minutes(start); e = _hm_to_minutes(end)
        if s <= e:
            if s <= minutes < e: return True
        else:
            if minutes >= s or minutes < e: return True
    return False

def _discover_png_runs() -> List[str]:
    pattern = os.path.join(PNG_PER_CLOUD_ROOT, "*")
    runs = []
    for d in glob.iglob(pattern):
        if os.path.isdir(d):
            runs.append(os.path.basename(d))
    return sorted(runs)

def _discover_sp_parquets() -> List[str]:
    pattern = os.path.join(SP_PER_CLOUD_ROOT, "*", "cloud_metrics.part*.parquet")
    return sorted(glob.iglob(pattern))

def _png_meta_path(run_tag: str) -> str:
    return os.path.join(PNG_META_ROOT, f"{run_tag}_meta.json")

def _choose_argmax_threshold(run_tag: str) -> Optional[float]:
    meta_path = _png_meta_path(run_tag)
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None
    d = meta.get("num_clouds_by_threshold")
    if not isinstance(d, dict) or not d:
        return None
    best_thr = None; best_cnt = None
    for k, v in d.items():
        try:
            thr = float(k); cnt = int(v)
        except Exception:
            continue
        if best_cnt is None or cnt > best_cnt or (cnt == best_cnt and (best_thr is None or thr < best_thr)):
            best_cnt = cnt; best_thr = thr
    return best_thr

def _iter_png_cloud_rows():
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
                continue
        for pq in shard_paths:
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

def _iter_sp_cloud_rows():
    for pq in _discover_sp_parquets():
        df = pd.read_parquet(pq)
        if P_VALS is not None and "p_val" in df.columns:
            df = df[df["p_val"].isin(P_VALS)]
        for _, row in df.iterrows():
            yield row

# ---------- Area filter ----------
def _area_ok(row: pd.Series) -> bool:
    if "area" not in row or pd.isna(row["area"]):
        return False
    try:
        a = int(row["area"])
    except Exception:
        try:
            a = int(float(row["area"]))
        except Exception:
            return False
    return (a >= MIN_AREA) and (a <= MAX_AREA)

# ---------------- BDD helpers ----------------
def _extract_bd_fields(row: pd.Series):
    """Return (rg_area, bd_r, bd_counts, bd_bin_width, bd_n) or None if invalid. Accepts legacy rp_* names."""
    if not _area_ok(row):
        return None

    # rg_area (allow 'Rg_area' fallback)
    if "rg_area" in row and pd.notna(row["rg_area"]):
        R = float(row["rg_area"])
    elif "Rg_area" in row and pd.notna(row["Rg_area"]):
        R = float(row["Rg_area"])
    else:
        return None

    # arrays
    bd_r = None
    bd_counts = None
    if "bd_r" in row and isinstance(row["bd_r"], (list, np.ndarray)):
        bd_r = np.asarray(row["bd_r"], dtype=np.float64)
    elif "rp_r" in row and isinstance(row["rp_r"], (list, np.ndarray)):
        bd_r = np.asarray(row["rp_r"], dtype=np.float64)

    if "bd_counts" in row and isinstance(row["bd_counts"], (list, np.ndarray)):
        bd_counts = np.asarray(row["bd_counts"], dtype=np.float64)
    elif "rp_counts" in row and isinstance(row["rp_counts"], (list, np.ndarray)):
        bd_counts = np.asarray(row["rp_counts"], dtype=np.float64)

    # widths and totals
    if "bd_bin_width" in row and pd.notna(row["bd_bin_width"]):
        dr = float(row["bd_bin_width"])
    else:
        if isinstance(bd_r, np.ndarray) and bd_r.size >= 2:
            diffs = np.diff(bd_r)
            dr = float(np.median(diffs))
        else:
            return None

    if "bd_n" in row and pd.notna(row["bd_n"]):
        Nb = float(row["bd_n"])
    else:
        Nb = float(np.sum(bd_counts)) if isinstance(bd_counts, np.ndarray) else 0.0

    # validate
    if not isinstance(bd_r, np.ndarray) or not isinstance(bd_counts, np.ndarray):
        return None
    if bd_r.size == 0 or bd_counts.size == 0 or bd_r.size != bd_counts.size:
        return None
    if not np.isfinite(R) or R <= 0: return None
    if not np.isfinite(dr) or dr <= 0: return None
    if not np.isfinite(Nb) or Nb <= 0: return None

    return R, bd_r, bd_counts, dr, Nb

def _discover_u_bounds(rows_iter):
    """First pass: discover u_min/u_max over all valid rows. Returns (umin, umax, counts)."""
    umin = np.inf; umax = -np.inf
    counts = {"files_considered": 0, "runs_considered": 0, "scanned_rows": 0, "kept_rows": 0, "skipped_bad_rows": 0}
    for row in rows_iter:
        counts["scanned_rows"] += 1
        ext = _extract_bd_fields(row)
        if ext is None:
            counts["skipped_bad_rows"] += 1
            continue
        R, r, c, dr, Nb = ext
        u = r / R
        mask = np.isfinite(u) & (u > 0) & np.isfinite(c) & (c > 0)
        if not np.any(mask):
            counts["skipped_bad_rows"] += 1
            continue
        u_valid = u[mask]
        umin = min(umin, float(np.min(u_valid)))
        umax = max(umax, float(np.max(u_valid)))
        counts["kept_rows"] += 1
    return umin, umax, counts

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
        centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    else:
        span = umax_obs - umin_obs
        pad = margin_frac * (span if span > 0 else 1.0)
        umin = max(umin_obs - pad, eps)
        umax = umax_obs + pad
        edges = np.linspace(umin, umax, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return edges, centers, widths

def _splat_into_bins(u_vals: np.ndarray, w_vals: np.ndarray,
                     edges: np.ndarray, spacing: str, mode: str,
                     sums: np.ndarray, covered_bins: set):
    """Add contributions to 'sums' in place; record bins touched into covered_bins (per cloud)."""
    eps = 1e-12
    if u_vals.size == 0:
        return
    if spacing == "log":
        logu = np.log(np.clip(u_vals, eps, None))
        log_edges = np.log(np.clip(edges, eps, None))
        d = log_edges[1] - log_edges[0]  # uniform in log-space
        f = (logu - log_edges[0]) / d
    else:
        d = edges[1] - edges[0]
        f = (u_vals - edges[0]) / d

    i0 = np.floor(f).astype(int)
    frac = f - i0
    i1 = i0 + 1

    # keep in-range indices
    in0 = (i0 >= 0) & (i0 < sums.size)
    in1 = (i1 >= 0) & (i1 < sums.size)

    if mode == "nearest":
        # nearest neighbor
        idx = np.clip(np.where(frac <= 0.5, i0, i1), 0, sums.size - 1)
        np.add.at(sums, idx, w_vals)
        for j in np.unique(idx):
            covered_bins.add(int(j))
    else:
        # linear 2-bin split
        if np.any(in0):
            np.add.at(sums, i0[in0], w_vals[in0] * (1.0 - frac[in0]))
            for j in np.unique(i0[in0]): covered_bins.add(int(j))
        if np.any(in1):
            np.add.at(sums, i1[in1], w_vals[in1] * frac[in1])
            for j in np.unique(i1[in1]): covered_bins.add(int(j))

def _accumulate(rows_iter, edges: np.ndarray, widths: np.ndarray):
    """Return (density_len, density_eq, coverage_len, coverage_eq, n_clouds, total_len, total_eq)."""
    nbins = widths.size
    sums_len = np.zeros(nbins, dtype=np.float64)
    sums_eq  = np.zeros(nbins, dtype=np.float64)
    coverage_len = np.zeros(nbins, dtype=np.int64)
    coverage_eq  = np.zeros(nbins, dtype=np.int64)
    n_clouds = 0

    for row in rows_iter:
        ext = _extract_bd_fields(row)
        if ext is None:
            continue
        R, r, c, dr, Nb = ext

        # per-sample contributions
        u = r / R
        mask = np.isfinite(u) & (u > 0) & np.isfinite(c) & (c > 0)
        if not np.any(mask):
            continue
        u = u[mask]
        w = c[mask]

        # weights per unit-u
        factor = (R / dr)
        w_len = w * factor
        w_eq  = (w / Nb) * factor

        # accumulate (length-weighted)
        covered_bins = set()
        _splat_into_bins(u, w_len, edges, U_SPACING, SPLAT, sums_len, covered_bins)
        if covered_bins:
            coverage_len[list(covered_bins)] += 1

        # accumulate (equal-cloud)
        covered_bins_eq = set()
        _splat_into_bins(u, w_eq, edges, U_SPACING, SPLAT, sums_eq, covered_bins_eq)
        if covered_bins_eq:
            coverage_eq[list(covered_bins_eq)] += 1

        n_clouds += 1

    # normalize to densities
    # density(u_bin) = (sums / total_mass) / bin_width
    dens_len = np.zeros_like(sums_len)
    dens_eq  = np.zeros_like(sums_eq)
    total_len = float(np.sum(sums_len))
    total_eq  = float(np.sum(sums_eq))
    if total_len > 0:
        dens_len = (sums_len / total_len) / widths
    if total_eq > 0:
        dens_eq = (sums_eq / total_eq) / widths
    return dens_len, dens_eq, coverage_len, coverage_eq, n_clouds, total_len, total_eq

# ---------------- Plotting ----------------
def _plot_u_density(centers: np.ndarray, density: np.ndarray, coverage: np.ndarray,
                    n_clouds: int, title: str, subtitle: str, out_png: str,
                    xscale: str, yscale: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(centers, density, lw=2.0)
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    ax.set_xlabel("u = r / Rg_area"); ax.set_ylabel("Density (per unit-u)")
    ax.set_title(title, pad=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    # Coverage cue: mark bins with low contributors
    if MIN_CLOUDS_PER_BIN > 0:
        low = (coverage < MIN_CLOUDS_PER_BIN) & (density > 0)
        if np.any(low):
            ax.scatter(centers[low], density[low], s=8, alpha=0.6)

    # Footer
    note = f"N_clouds={n_clouds} | nbins={len(centers)} | spacing={U_SPACING} | splat={SPLAT}"
    wrapped = "\n".join(textwrap.wrap(subtitle + "  |  " + note, width=110))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.text(
        0.5, 0.045, wrapped,
        ha="center", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95, linewidth=0.5),
    )
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

# ---------------- Main ----------------
def main():
    _assert_source()
    _ensure_outdir()

    # Iterators and discovery
    if SOURCE == "png":
        umin, umax, counts = _discover_u_bounds(_iter_png_cloud_rows())
        runs_considered = set()
        for run_tag in _discover_png_runs():
            if not _within_any_window(run_tag, TIME_WINDOWS):
                continue
            shard_paths = sorted(glob.iglob(os.path.join(PNG_PER_CLOUD_ROOT, run_tag, "cloud_metrics.part*.parquet")))
            if not shard_paths:
                continue
            chosen_thr = None
            if THRESHOLD_POLICY == "argmax_clouds":
                chosen_thr = _choose_argmax_threshold(run_tag)
                if chosen_thr is None and ON_MISSING_METADATA == "skip_run":
                    continue
            kept_any = False
            for pq in shard_paths:
                df = pd.read_parquet(pq)
                if "threshold" in df.columns:
                    if THRESHOLD_POLICY == "argmax_clouds":
                        if chosen_thr is not None:
                            df = df[np.isclose(df["threshold"].astype(float), float(chosen_thr), rtol=1e-10, atol=1e-12)]
                    elif THRESHOLD_POLICY == "all":
                        if THRESHOLDS_WHITELIST is not None:
                            df = df[df["threshold"].isin(THRESHOLDS_WHITELIST)]
                for _, row in df.iterrows():
                    if _extract_bd_fields(row) is not None:
                        kept_any = True
                        break
                if kept_any:
                    break
            if kept_any:
                runs_considered.add(run_tag)
        counts["runs_considered"] = len(runs_considered)
        mode_descr = f'png[{THRESHOLD_POLICY}]'
        rows_iter_for_accum = _iter_png_cloud_rows()
    else:
        umin, umax, counts = _discover_u_bounds(_iter_sp_cloud_rows())
        counts["files_considered"] = len(_discover_sp_parquets())
        mode_descr = 'siteperc'
        rows_iter_for_accum = _iter_sp_cloud_rows()

    # Build u-grid
    edges, centers, widths = _build_u_grid(umin, umax, U_SPACING, U_NBINS, U_MARGIN_FRAC)

    # Accumulate
    dens_len, dens_eq, coverage_len, coverage_eq, n_clouds, total_len, total_eq = _accumulate(
        rows_iter_for_accum, edges, widths
    )

    if n_clouds == 0 or (dens_len.sum() == 0 and dens_eq.sum() == 0):
        print("[WARN] No valid clouds after filtering; nothing to plot.")
        print("Counts:", counts)
        return

    # ---------- NEW: Save plot-ready CSVs by default ----------
    bdd_len_csv = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__bdd_len.csv")
    bdd_eq_csv  = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__bdd_eq.csv")

    pd.DataFrame({
        "u": centers,
        "y": dens_len,
        "coverage": coverage_len,
        "width": widths,
    }).to_csv(bdd_len_csv, index=False)

    pd.DataFrame({
        "u": centers,
        "y": dens_eq,
        "coverage": coverage_eq,
        "width": widths,
    }).to_csv(bdd_eq_csv, index=False)
    # ----------------------------------------------------------

    # Compose labels
    filt_bits = [f"{MIN_AREA} ≤ area ≤ {MAX_AREA}"]
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
    subtitle = f"Mode: {mode_descr}  |  Filters: " + "; ".join(filt_bits)

    # File paths
    out_len_log   = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__bdd_lenweighted.png")
    out_eq_log    = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__bdd_equalcloud.png")
    out_len_lin   = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__bdd_lenweighted_linear.png")
    out_eq_lin    = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__bdd_equalcloud_linear.png")
    metrics_json  = os.path.join(OUTPUT_DIR, f"{SUFFIX_TAG}__bdd_metrics.json")

    # Plots: log–log
    title = f"Boundary Distance Distribution — {SOURCE.upper()} (length-weighted)"
    _plot_u_density(centers, dens_len, coverage_len, n_clouds, title, subtitle, out_len_log, "log", "log")

    title2 = f"Boundary Distance Distribution — {SOURCE.upper()} (equal-cloud)"
    _plot_u_density(centers, dens_eq, coverage_eq, n_clouds, title2, subtitle, out_eq_log, "log", "log")

    # Plots: linear–linear (as requested)
    title3 = f"Boundary Distance Distribution — {SOURCE.upper()} (length-weighted, linear axes)"
    _plot_u_density(centers, dens_len, coverage_len, n_clouds, title3, subtitle, out_len_lin, "linear", "linear")

    title4 = f"Boundary Distance Distribution — {SOURCE.upper()} (equal-cloud, linear axes)"
    _plot_u_density(centers, dens_eq, coverage_eq, n_clouds, title4, subtitle, out_eq_lin, "linear", "linear")

    # Metrics
    metrics = {
        "source": SOURCE,
        "suffix_tag": SUFFIX_TAG,
        "mode_descr": mode_descr,
        "filters": {
            "min_area": MIN_AREA, "max_area": MAX_AREA,
            "time_windows": TIME_WINDOWS if SOURCE == "png" else None,
            "threshold_policy": THRESHOLD_POLICY if SOURCE == "png" else None,
            "thresholds_whitelist": THRESHOLDS_WHITELIST if (SOURCE == "png" and THRESHOLD_POLICY=="all") else None,
            "on_missing_metadata": ON_MISSING_METADATA if (SOURCE == "png" and THRESHOLD_POLICY=="argmax_clouds") else None,
            "p_vals": P_VALS if SOURCE == "siteperc" else None,
        },
        "u_grid": {
            "spacing": U_SPACING,
            "nbins": int(U_NBINS),
            "margin_frac": float(U_MARGIN_FRAC),
            "u_min_obs": float(umin),
            "u_max_obs": float(umax),
            "edges_len": int(len(edges)),
        },
        "splat": SPLAT,
        "min_clouds_per_bin": int(MIN_CLOUDS_PER_BIN),
        "norm_method": "rg_area",
        "bin_width_r": "per-row: bd_bin_width",
        "weighting_modes": {
            "length": {
                "total_contribution": float(total_len),
                "n_clouds": int(n_clouds),
                "plots": {"loglog": out_len_log, "linear": out_len_lin}
            },
            "equal_cloud": {
                "total_contribution": float(total_eq),
                "n_clouds": int(n_clouds),
                "plots": {"loglog": out_eq_log, "linear": out_eq_lin}
            }
        },
        "counts": counts,
        "notes": "Area filter enforced; no recentering; no ring correction; densities per unit-u; aggregated across all clouds passing filters."
    }
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary
    print("-" * 60)
    print(" BDD (u = r/Rg_area) — global aggregate")
    print("-" * 60)
    print(f"Saved plot-data CSVs:         {bdd_len_csv} , {bdd_eq_csv}")
    print(f"Saved plots (length-weighted): {out_len_log}  |  {out_len_lin}")
    print(f"Saved plots (equal-cloud):     {out_eq_log}   |  {out_eq_lin}")
    print(f"Saved metrics:                 {metrics_json}")
    print("Counts:", metrics["counts"])

# -------------------- CLI overrides + entrypoint --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate Boundary Distance Distribution across clouds.")
    parser.add_argument("--source", choices=["png", "siteperc"], help="Data source: png or siteperc.")
    parser.add_argument("--threshold_policy", choices=["all", "argmax_clouds"], help="PNG only.")
    parser.add_argument("--time_window", help="Single 'HH:MM-HH:MM' or 'all' (for all times).")
    parser.add_argument("--suffix_tag", help="Suffix tag for output filenames (required).")
    parser.add_argument("--p_vals", help="Site-perc p-values, comma-separated or 'all'.")
    # grid knobs
    parser.add_argument("--u_spacing", choices=["log","linear"])
    parser.add_argument("--u_nbins", type=int)
    parser.add_argument("--u_margin_frac", type=float)
    parser.add_argument("--splat", choices=["linear","nearest"])
    parser.add_argument("--min_clouds_per_bin", type=int)
    # area filter overrides
    parser.add_argument("--min_area", type=int)
    parser.add_argument("--max_area", type=int)
    args = parser.parse_args()

    # Apply overrides
    if args.source: SOURCE = args.source
    if args.threshold_policy: THRESHOLD_POLICY = args.threshold_policy
    if args.time_window:
        if args.time_window.lower() == "all":
            TIME_WINDOWS = []
        else:
            start, end = args.time_window.split("-")
            TIME_WINDOWS = [(start.strip(), end.strip())]
    if args.suffix_tag: SUFFIX_TAG = args.suffix_tag
    if args.p_vals:
        if args.p_vals.lower() == "all":
            P_VALS = None
        else:
            P_VALS = [float(x) for x in args.p_vals.split(",")]
    if args.u_spacing: U_SPACING = args.u_spacing
    if args.u_nbins: U_NBINS = int(args.u_nbins)
    if args.u_margin_frac is not None: U_MARGIN_FRAC = float(args.u_margin_frac)
    if args.splat: SPLAT = args.splat
    if args.min_clouds_per_bin is not None: MIN_CLOUDS_PER_BIN = int(args.min_clouds_per_bin)
    if args.min_area is not None: MIN_AREA = int(args.min_area)
    if args.max_area is not None: MAX_AREA = int(args.max_area)

    main()
