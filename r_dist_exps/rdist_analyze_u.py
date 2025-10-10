#!/usr/bin/env python3
"""
rdist_analyze_u.py

Aggregate per-cloud boundary radial profiles into a scale-invariant view with
configurable size normalizer R (R_eff) used in u = ln(r / R).

Now includes absolute-u analyses in addition to Δu:
  - Global u survival curves (right: P(U>=t), left: P(U<=t)) with tail fits.
  - Per-cloud u_peak distribution (uses the same 'center-recenter' statistic).
  - Optional size-tercile overlays for u survivals (mirrors u hist overlays).

Supported R choices (per-cloud):
  - rg_area      : radius of gyration of area (preferred). Uses column 'Rg_area'
                   if present; otherwise falls back to rg_boundary surrogate.
  - rg_boundary  : radius of gyration computed from boundary histogram (rp_r, rp_counts).
  - req          : equivalent disk radius = sqrt(area / pi).
  - rlog         : geometric mean of boundary radii (from rp_r, rp_counts).
  - rlog_trim    : trimmed geometric mean (trim q% from each tail in r).

Other features preserved:
  - per-cloud recentering in u using weighted median (or log-mean)
  - Δu survival curves (two-sided), tail slope fits
  - optional inner-radius pixel cutoff
  - equal-cloud vs length (count) weighting
  - size-tercile overlays
  - α-grid sanity overlays in r-space (raw, /r^0.5, /r^1)

Usage example:
python -m clouds.r_dist_exps.rdist_analyze_u \
  --per-cloud-dir scratch/expC/per_cloud/expC_internal_W10000_H10000_p0.407400_seed987 \
  --outdir scratch/expC/analysis_u_rgBoundary/expC_internal_W10000_H10000_p0.407400_seed987 \
  --du 0.04 \
  --norm-method rg_boundary \
  --size-terciles \
  --plots
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt

PI = math.pi

# ---------------------------
# Helpers
# ---------------------------

def _iter_rows(dataset: ds.Dataset, columns: List[str]) -> Iterable[Dict]:
    """Stream rows from a pyarrow Dataset as Python dicts (selected columns only)."""
    scanner = dataset.scanner(columns=columns)
    for batch in scanner.to_batches():
        tbl = pa.Table.from_batches([batch])
        cols = [tbl[c].to_pylist() for c in columns]
        for values in zip(*cols):
            yield {k: v for k, v in zip(columns, values)}

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order].astype(np.float64)
    cdf = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cdf, 0.5)
    idx = min(idx, len(v) - 1)
    return float(v[idx])

def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))

def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares fit y ~ a*x + b; returns (a, b, r2)."""
    if x.size < 2:
        return np.nan, np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    return float(a), float(b), float(r2)

# ---------------------------
# Binners
# ---------------------------

class LogRadiusBinner:
    """Histogram in u = ln(r/R) with linear y (density per log-radius)."""

    def __init__(self, u_min: float, u_max: float, du: float):
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.du = float(du)
        nbins = int(np.floor((self.u_max - self.u_min) / self.du)) + 1
        self.edges = self.u_min + np.arange(nbins + 1) * self.du
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.counts = np.zeros(nbins, dtype=np.float64)

    def splat(self, u: np.ndarray, w: np.ndarray) -> None:
        """Linear 'subpixel' splat to adjacent bins."""
        mask = (u >= self.edges[0]) & (u < self.edges[-1])
        if not np.any(mask):
            return
        u = u[mask]
        w = w[mask].astype(np.float64)

        f = (u - self.u_min) / self.du
        i0 = np.floor(f).astype(int)
        frac = f - i0
        i1 = i0 + 1

        np.add.at(self.counts, i0, w * (1.0 - frac))
        in_bounds = i1 < self.counts.size
        np.add.at(self.counts, i1[in_bounds], w[in_bounds] * frac[in_bounds])

    def density(self) -> np.ndarray:
        total = self.counts.sum()
        if total <= 0:
            return np.zeros_like(self.counts)
        return self.counts / (total * self.du)

class DeltaUBinner(LogRadiusBinner):
    pass

# ---------------------------
# R_eff calculators (per cloud)
# ---------------------------

def _rg_boundary_from_hist(r: np.ndarray, counts: np.ndarray) -> float:
    """Boundary Rg from radial boundary histogram."""
    w = counts.astype(np.float64)
    if w.sum() <= 0:
        return np.nan
    m2 = np.sum(w * (r ** 2)) / np.sum(w)
    return float(np.sqrt(max(m2, 0.0)))

def _rlog_from_hist(r: np.ndarray, counts: np.ndarray, qtrim: float | None = None) -> float:
    """Geometric mean radius from boundary histogram; optional symmetric trimming in r."""
    w = counts.astype(np.float64)
    m = (r > 0) & (w > 0)
    if not np.any(m):
        return np.nan
    r, w = r[m], w[m]
    order = np.argsort(r)
    r, w = r[order], w[order]
    cw = np.cumsum(w)
    tot = cw[-1]
    if qtrim:
        lo = qtrim * tot
        hi = (1.0 - qtrim) * tot
        keep = (cw >= lo) & (cw <= hi)
        if not np.any(keep):
            keep = slice(None)
        r, w = r[keep], w[keep]
    ln_r = np.log(r)
    Rlog = np.exp(np.sum(w * ln_r) / np.sum(w))
    return float(Rlog)

def _req_from_area(area: int) -> float:
    return math.sqrt(max(area, 0) / PI)

def _choose_Reff(norm_method: str,
                 area: int,
                 rp_r: np.ndarray,
                 rp_counts: np.ndarray,
                 maybe_rg_area: float | None = None) -> float:
    """
    Decide R_eff per cloud based on norm_method.
    - 'rg_area' uses maybe_rg_area if provided; otherwise falls back to 'rg_boundary'.
    - 'rg_boundary' uses boundary histogram.
    - 'req' uses sqrt(A/pi).
    - 'rlog' geometric mean of boundary radii.
    - 'rlog_trim' trimmed geometric mean (5% default).
    """
    nm = norm_method.lower()
    if nm == "rg_area":
        if (maybe_rg_area is not None) and np.isfinite(maybe_rg_area) and (maybe_rg_area > 0):
            return float(maybe_rg_area)
        rg_b = _rg_boundary_from_hist(rp_r, rp_counts)
        return rg_b if np.isfinite(rg_b) and rg_b > 0 else _req_from_area(area)
    elif nm == "rg_boundary":
        rg_b = _rg_boundary_from_hist(rp_r, rp_counts)
        return rg_b if np.isfinite(rg_b) and rg_b > 0 else _req_from_area(area)
    elif nm == "req":
        return _req_from_area(area)
    elif nm == "rlog":
        rlog = _rlog_from_hist(rp_r, rp_counts, qtrim=None)
        return rlog if np.isfinite(rlog) and rlog > 0 else _req_from_area(area)
    elif nm == "rlog_trim":
        rlogt = _rlog_from_hist(rp_r, rp_counts, qtrim=0.05)
        return rlogt if np.isfinite(rlogt) and rlogt > 0 else _req_from_area(area)
    else:
        return _req_from_area(area)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate per-cloud r-profiles into u-space with metrics.")
    ap.add_argument("--per-cloud-dir", required=True, type=Path,
                    help="Directory containing per_cloud Parquet shards (basename 'rprof').")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Output directory for agg artifacts and plots.")
    ap.add_argument("--du", type=float, default=0.04, help="Bin width in u = ln(r/R).")
    ap.add_argument("--u-margin", type=float, default=0.05,
                    help="Extra margin around observed [u_min,u_max].")
    ap.add_argument("--u0", type=float, default=0.30,
                    help="Peakiness window: use |u| <= u0.")
    ap.add_argument("--delta-fit-min", type=float, default=0.30,
                    help="Tail fit window lower bound for Δu.")
    ap.add_argument("--delta-fit-max", type=float, default=1.00,
                    help="Tail fit window upper bound for Δu.")
    # NEW: u-tail fit windows (absolute u)
    ap.add_argument("--u-fit-right-min", type=float, default=0.30,
                    help="Right-tail fit lower bound for u (t where U>=t).")
    ap.add_argument("--u-fit-right-max", type=float, default=1.20,
                    help="Right-tail fit upper bound for u.")
    ap.add_argument("--u-fit-left-max", type=float, default=-0.30,
                    help="Left-tail fit upper bound for u (most negative t to include).")
    ap.add_argument("--u-fit-left-min", type=float, default=-1.20,
                    help="Left-tail fit lower bound for u (more negative).")
    ap.add_argument("--alpha-grid", type=str, default="0,0.5,1",
                    help="Comma-separated α values for H(r)/r^α overlays.")
    ap.add_argument("--norm-method", choices=["rg_area","rg_boundary","req","rlog","rlog_trim"],
                    default="rg_area",
                    help="Choice of R_eff used in u = ln(r/R_eff).")
    ap.add_argument("--center-recenter", choices=["median","logmean"], default="median",
                    help="Statistic used both as Δu center and as per-cloud u_peak.")
    ap.add_argument("--inner-radius-px", type=float, default=0.0,
                    help="Ignore r < this many pixels when forming u.")
    ap.add_argument("--cloud-weighting", choices=["length", "equal"], default="length",
                    help="Aggregate H(u) by boundary length (sum counts) or equally weight each cloud.")
    ap.add_argument("--size-terciles", action="store_true", help="Compute size-split aggregates (by area).")
    ap.add_argument("--plots", action="store_true", help="Save plots.")
    args = ap.parse_args()

    per_cloud_dir: Path = args.per_cloud_dir
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    alpha_vals = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    # Try to include optional columns if present
    dset = ds.dataset(per_cloud_dir, format="parquet")
    base_cols = set(["cloud_idx", "area", "rp_r", "rp_counts"])
    extra_cols = []
    try:
        any_file = next(iter(dset.files))
        sch = pa.parquet.ParquetFile(any_file).schema_arrow
        names = set(sch.names)
        if "Rg_area" in names:
            extra_cols.append("Rg_area")
    except Exception:
        pass
    need_cols = list(base_cols.union(extra_cols))

    # ---------- PASS 1: determine u-range and size thresholds ----------
    u_min_obs, u_max_obs = +np.inf, -np.inf
    areas = []

    for row in _iter_rows(dset, need_cols):
        area = int(row["area"])
        rp_r = np.asarray(row["rp_r"], dtype=np.float64)
        counts = np.asarray(row["rp_counts"], dtype=np.float64)
        if rp_r.size == 0 or counts.sum() == 0:
            continue
        if args.inner_radius_px > 0:
            keep = rp_r >= args.inner_radius_px
            if not np.any(keep):
                continue
            rp_r, counts = rp_r[keep], counts[keep]

        maybe_rg_area = row.get("Rg_area", None)
        R = _choose_Reff(args.norm_method, area, rp_r, counts, maybe_rg_area)

        u = _safe_log(rp_r / R)
        nz = counts > 0
        if np.any(nz):
            u_min_obs = min(u_min_obs, float(u[nz].min()))
            u_max_obs = max(u_max_obs, float(u[nz].max()))
        areas.append(area)

    if not areas:
        print("[WARN] No valid rows found.")
        return

    u_pad = args.u_margin * (u_max_obs - u_min_obs if u_max_obs > u_min_obs else 1.0)
    u_min = u_min_obs - u_pad
    u_max = u_max_obs + u_pad

    # Size splits (terciles by area)
    terc_cuts = None
    if args.size_terciles:
        q1, q2 = np.quantile(areas, [1/3, 2/3])
        terc_cuts = (q1, q2)

    # ---------- Initialize binners ----------
    H_u_all = LogRadiusBinner(u_min, u_max, args.du)
    H_delu_all = DeltaUBinner(-(u_max - u_min), (u_max - u_min), args.du)

    H_u_small = LogRadiusBinner(u_min, u_max, args.du) if terc_cuts else None
    H_u_mid   = LogRadiusBinner(u_min, u_max, args.du) if terc_cuts else None
    H_u_large = LogRadiusBinner(u_min, u_max, args.du) if terc_cuts else None

    # α-grid (in r-space) for sanity overlays
    H_r_alpha = {alpha: np.zeros(0, dtype=np.float64) for alpha in alpha_vals}
    r_bin_width = None  # infer from first row

    # NEW: collect per-cloud u_peak values (median/logmean)
    percloud_u_peaks: List[float] = []
    percloud_areas: List[int] = []

    # ---------- PASS 2: accumulate ----------
    total_boundary_points = 0

    for row in _iter_rows(dset, need_cols):
        area = int(row["area"])
        rp_r = np.asarray(row["rp_r"], dtype=np.float64)
        counts = np.asarray(row["rp_counts"], dtype=np.float64)

        if rp_r.size == 0 or counts.sum() == 0:
            continue
        if args.inner_radius_px > 0:
            keep = rp_r >= args.inner_radius_px
            if not np.any(keep):
                continue
            rp_r, counts = rp_r[keep], counts[keep]

        if r_bin_width is None and rp_r.size >= 2:
            r_bin_width = float(rp_r[1] - rp_r[0])

        maybe_rg_area = row.get("Rg_area", None)
        R = _choose_Reff(args.norm_method, area, rp_r, counts, maybe_rg_area)

        # ---- u space accumulation ----
        u = _safe_log(rp_r / R)

        # per-cloud center (used both as Δu center and as u_peak)
        if args.center_recenter == "median":
            u_c = _weighted_median(u, counts)
        else:
            w = counts.astype(np.float64)
            u_c = float(np.sum(w * u) / np.sum(w))

        percloud_u_peaks.append(float(u_c))
        percloud_areas.append(area)

        # Global H_u(u)
        if args.cloud_weighting == "length":
            H_u_all.splat(u, counts)
        else:
            w = counts / max(counts.sum(), 1.0)
            H_u_all.splat(u, w)

        # Δu per-cloud recenter for two-sided tails (always length-weighted shape)
        delu = u - u_c
        H_delu_all.splat(delu, counts)

        # Size splits
        if terc_cuts:
            target = H_u_small if area <= terc_cuts[0] else H_u_mid if area <= terc_cuts[1] else H_u_large
            if args.cloud_weighting == "length":
                target.splat(u, counts)
            else:
                target.splat(u, counts / max(counts.sum(), 1.0))

        # ---- α-grid in r-space (sanity overlays) ----
        for alpha in alpha_vals:
            w = counts / np.power(np.maximum(rp_r, 1e-12), alpha)
            if H_r_alpha[alpha].size < w.size:
                grown = np.zeros(w.size, dtype=np.float64)
                if H_r_alpha[alpha].size:
                    grown[: H_r_alpha[alpha].size] = H_r_alpha[alpha]
                H_r_alpha[alpha] = grown
            H_r_alpha[alpha][: w.size] += w
        total_boundary_points += int(counts.sum())

    # ---------- Metrics ----------
    density_u = H_u_all.density()
    u_centers = H_u_all.centers
    win = np.abs(u_centers) <= args.u0
    if np.any(win):
        peak = float(density_u[win].max())
        med = float(np.median(density_u[win]))
        P = (peak / med) if med > 0 else np.inf
    else:
        P = np.nan

    # Δu survival (two-sided)
    delu_centers = H_delu_all.centers
    delu_counts = H_delu_all.counts
    total_delu = delu_counts.sum()
    right_mask = delu_centers >= 0
    x_right = delu_centers[right_mask]
    y_right = delu_counts[right_mask]
    S_right = np.cumsum(y_right[::-1])[::-1] / max(total_delu, 1)
    left_mask = delu_centers <= 0
    x_left = -delu_centers[left_mask]
    y_left = delu_counts[left_mask]
    S_left = np.cumsum(y_left) / max(total_delu, 1)

    def fit_tail(x, S):
        mask = (S > 0)
        if not np.any(mask):
            return dict(slope=np.nan, intercept=np.nan, r2=np.nan, n=0)
        a, b, r2 = _fit_line(x[mask], np.log(S[mask]))
        return dict(slope=a, intercept=b, r2=r2, n=int(mask.sum()))

    # Δu tail fits within window
    mask_r = (x_right >= args.delta_fit_min) & (x_right <= args.delta_fit_max) & (S_right > 0)
    fit_right = fit_tail(x_right[mask_r], S_right[mask_r])
    mask_l = (x_left >= args.delta_fit_min) & (x_left <= args.delta_fit_max) & (S_left > 0)
    fit_left  = fit_tail(x_left[mask_l],  S_left[mask_l])

    # -------- NEW: absolute-u survivals --------
    u_counts = H_u_all.counts
    total_u = u_counts.sum()
    # Right tail S_u_plus(t) = P(U >= t) at bin centers to the right
    S_u_plus = np.cumsum(u_counts[::-1])[::-1] / max(total_u, 1.0)
    # Left tail S_u_minus(t) = P(U <= t) at bin centers to the left
    S_u_minus = np.cumsum(u_counts) / max(total_u, 1.0)

    # u-tail fits (independent windows for left/right)
    # Right: use [u_fit_right_min, u_fit_right_max]
    mask_ur = (u_centers >= args.u_fit_right_min) & (u_centers <= args.u_fit_right_max) & (S_u_plus > 0)
    fit_u_right = fit_tail(u_centers[mask_ur], S_u_plus[mask_ur])
    # Left: use [u_fit_left_min, u_fit_left_max] (note: left_min is more negative)
    # Ensure left_min < left_max < 0
    u_lmin = min(args.u_fit_left_min, args.u_fit_left_max)
    u_lmax = max(args.u_fit_left_min, args.u_fit_left_max)
    mask_ul = (u_centers >= u_lmin) & (u_centers <= u_lmax) & (S_u_minus > 0)
    fit_u_left = fit_tail(u_centers[mask_ul], S_u_minus[mask_ul])

    # ---------- Save artifacts ----------
    agg_dir = outdir / "agg"
    plot_dir = outdir / "plots"
    agg_dir.mkdir(parents=True, exist_ok=True)
    if args.plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "u_center": u_centers,
        "counts": H_u_all.counts,
        "density": density_u,
        "S_u_plus": S_u_plus,   # P(U >= t)
        "S_u_minus": S_u_minus, # P(U <= t)
    }).to_parquet(agg_dir / "u_hist_and_survivals.parquet")

    pd.DataFrame({
        "delu_center": delu_centers,
        "counts": H_delu_all.counts,
    }).to_parquet(agg_dir / "delu_hist.parquet")

    # Size splits (u hists)
    if terc_cuts:
        for name, H in [("small", H_u_small), ("mid", H_u_mid), ("large", H_u_large)]:
            tot = H.counts.sum()
            den = (H.counts / (tot * H.du)) if tot > 0 else np.zeros_like(H.counts)
            pd.DataFrame({
                "u_center": H.centers,
                "counts": H.counts,
                "density": den
            }).to_parquet(agg_dir / f"u_hist_{name}.parquet")

    # α-grid overlays
    for alpha, series in H_r_alpha.items():
        pd.DataFrame({
            "r_index": np.arange(series.size),
            "value": series
        }).to_parquet(agg_dir / f"r_alpha_{alpha:g}.parquet")

    # Per-cloud u_peak values
    pd.DataFrame({
        "u_peak": percloud_u_peaks,
        "area": percloud_areas
    }).to_parquet(agg_dir / "percloud_u_peaks.parquet")

    # Metrics JSON
    metrics = {
        "total_boundary_points": int(total_boundary_points),
        "u_bin_width": args.du,
        "u_min": float(H_u_all.edges[0]),
        "u_max": float(H_u_all.edges[-1]),
        "peakiness_P_absu_le_u0": float(P),
        "u0": args.u0,
        "tail_fit_window_delu": [args.delta_fit_min, args.delta_fit_max],
        "right_tail_fit_delu": fit_right,
        "left_tail_fit_delu":  fit_left,
        "tail_fit_window_u_right": [args.u_fit_right_min, args.u_fit_right_max],
        "tail_fit_window_u_left":  [u_lmin, u_lmax],
        "right_tail_fit_u": fit_u_right,
        "left_tail_fit_u":  fit_u_left,
        "alpha_grid": [float(a) for a in alpha_vals],
        "size_terciles": (terc_cuts if terc_cuts else None),
        "norm_method": args.norm_method,
        "inner_radius_px": args.inner_radius_px,
        "cloud_weighting": args.cloud_weighting,
        "center_recenter": args.center_recenter,
    }
    (agg_dir / "metrics.json").write_text(pd.Series(metrics).to_json(indent=2))

    # ---------- Plots ----------
    if args.plots:
        # Main: H(u)
        plt.figure()
        plt.plot(u_centers, density_u, lw=1.5)
        plt.axvspan(-args.u0, args.u0, alpha=0.1)
        plt.xlabel(f"u = ln(r / R_eff), R_eff={args.norm_method}")
        plt.ylabel("Density (per log-radius)")
        plt.title("Aggregate H(u)")
        plt.tight_layout()
        plt.savefig(plot_dir / "agg_u_hist.png", dpi=160)

        # Size overlay (u)
        if terc_cuts:
            plt.figure()
            for name, H in [("small", H_u_small), ("mid", H_u_mid), ("large", H_u_large)]:
                tot = H.counts.sum()
                den = (H.counts / (tot * H.du)) if tot > 0 else np.zeros_like(H.counts)
                plt.plot(H.centers, den, label=name, lw=1.4)
            plt.axvspan(-args.u0, args.u0, alpha=0.1)
            plt.xlabel(f"u = ln(r / R_eff), R_eff={args.norm_method}")
            plt.ylabel("Density")
            plt.legend()
            plt.title("H(u) by size tercile")
            plt.tight_layout()
            plt.savefig(plot_dir / "size_split_u_overlays.png", dpi=160)

        # NEW: u survivals (semi-log y)
        plt.figure()
        plt.semilogy(u_centers, S_u_plus, label="right: S_u^+(t)=P(U≥t)")
        plt.semilogy(u_centers, S_u_minus, label="left : S_u^-(t)=P(U≤t)")
        # Right-tail fit overlay
        if fit_u_right["n"] > 1 and np.isfinite(fit_u_right["slope"]):
            xr = np.linspace(args.u_fit_right_min, args.u_fit_right_max, 80)
            yr = np.exp(fit_u_right["slope"] * xr + fit_u_right["intercept"])
            plt.semilogy(xr, yr, "--", label=f"right fit (slope={fit_u_right['slope']:.3f})")
        # Left-tail fit overlay
        if fit_u_left["n"] > 1 and np.isfinite(fit_u_left["slope"]):
            xl = np.linspace(u_lmin, u_lmax, 80)
            yl = np.exp(fit_u_left["slope"] * xl + fit_u_left["intercept"])
            plt.semilogy(xl, yl, "--", label=f"left fit (slope={fit_u_left['slope']:.3f})")
        plt.xlabel("u = ln(r / R_eff)")
        plt.ylabel("Survival")
        plt.title("Absolute-u survival curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "u_survivals.png", dpi=160)

        # NEW: per-cloud u_peak distribution
        plt.figure()
        plt.hist(percloud_u_peaks, bins=max(20, int(np.sqrt(len(percloud_u_peaks)))), density=True)
        plt.xlabel(f"u_peak per cloud ({args.center_recenter})")
        plt.ylabel("Density")
        plt.title("Distribution of per-cloud u_peak")
        plt.tight_layout()
        plt.savefig(plot_dir / "percloud_u_peak_hist.png", dpi=160)

        # Δu survival curves (semi-log y), unchanged
        plt.figure()
        plt.semilogy(x_right, S_right, label="right (Δu ≥ δ)")
        plt.semilogy(x_left,  S_left,  label="left  (Δu ≤ -δ)")
        # Fits
        if fit_right["n"] > 1 and np.isfinite(fit_right["slope"]):
            xw = np.linspace(args.delta_fit_min, args.delta_fit_max, 50)
            yw = np.exp(fit_right["slope"] * xw + fit_right["intercept"])
            plt.semilogy(xw, yw, "--", label=f"right fit (slope={fit_right['slope']:.3f})")
        if fit_left["n"] > 1 and np.isfinite(fit_left["slope"]):
            xw = np.linspace(args.delta_fit_min, args.delta_fit_max, 50)
            yw = np.exp(fit_left["slope"] * xw + fit_left["intercept"])
            plt.semilogy(xw, yw, "--", label=f"left fit (slope={fit_left['slope']:.3f})")
        plt.xlabel("δ (|Δu|)")
        plt.ylabel("Survival S(δ)")
        plt.title("Two-sided tail survival (Δu)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "survival_two_sided.png", dpi=160)

        # α-grid overlays (r-space)
        plt.figure()
        for alpha, series in H_r_alpha.items():
            if r_bin_width is not None:
                r_centers = (np.arange(series.size) + 0.5) * r_bin_width
                plt.plot(r_centers, series, label=f"α={alpha:g}", lw=1.3)
                plt.xlabel("r (px)")
            else:
                plt.plot(series, label=f"α={alpha:g}", lw=1.3)
                plt.xlabel("r-bin index")
            plt.ylabel("Aggregated value")
        plt.title("H(r)/r^α overlays")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "r_alpha_overlays.png", dpi=160)

    print("[OK] Finished. Artifacts written to:", str(outdir))


if __name__ == "__main__":
    main()
