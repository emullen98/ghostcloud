#!/usr/bin/env python3
"""
rdist_analyze_rinv.py

Aggregate per-cloud boundary radial profiles in linear radius with the 1/r correction.

Inputs: per-cloud Parquet shards produced by siteperc_rprof_from_model.py
        (expects columns: cloud_idx, area, rp_r, rp_counts)

What it does (per cloud):
  - size normalizer R = sqrt(area/pi)  (r_eq)
  - scaled radius:   r_tilde = r / R
  - apply crowding correction: weights = rp_counts / r
      (optional: also divide by (2*pi) and bin-width to match "ring-corrected" units)
  - accumulate:
      A) H_corr(r_tilde): histogram in r_tilde with 1/r weights (global; and by size tercile)
      B) H_deltar(r_tilde - median_r_tilde): re-centered histogram for two-sided tails

Outputs:
  - agg/u_histograms (here: r_tilde_hist.parquet, delta_rt_hist.parquet)
  - metrics.json with peakiness and left/right survival slopes (linear Δr_tilde window)
  - plots (main corrected density vs r_tilde, two-sided survival, size-tercile overlays)

Sample usage:
python -m clouds.r_dist_exps.rdist_analyze_rinv \
  --per-cloud-dir scratch/expC/per_cloud/expC_internal_W10000_H10000_p0.407400_seed987 \
  --outdir scratch/expC/analysis_rinv/expC_internal_W10000_H10000_p0.407400_seed987 \
  --drt 0.02 \
  --size-terciles \
  --plots \
  --include_2pi \
  --inner-radius-px 0 \
  --cloud-weighting length
  
"""

from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import Iterable, Dict, List, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt

PI = math.pi

# ---------------------------
# helpers
# ---------------------------

def _iter_rows(dataset: ds.Dataset, cols: List[str]) -> Iterable[Dict]:
    scanner = dataset.scanner(columns=cols)
    for batch in scanner.to_batches():
        tbl = pa.Table.from_batches([batch])
        col_lists = [tbl[c].to_pylist() for c in cols]
        for vals in zip(*col_lists):
            yield {k: v for k, v in zip(cols, vals)}

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order].astype(np.float64)
    c = np.cumsum(w) / np.sum(w)
    i = np.searchsorted(c, 0.5)
    i = min(i, len(v) - 1)
    return float(v[i])

def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
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
# binners
# ---------------------------

class LinearBinner:
    """Histogram on a fixed linear grid [x_min, x_max] with step dx; supports 'splat' (linear) add."""
    def __init__(self, x_min: float, x_max: float, dx: float):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.dx = float(dx)
        nbins = int(np.floor((self.x_max - self.x_min) / self.dx)) + 1
        self.edges = self.x_min + np.arange(nbins + 1) * self.dx
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.counts = np.zeros(nbins, dtype=np.float64)

    def splat(self, x: np.ndarray, w: np.ndarray) -> None:
        m = (x >= self.edges[0]) & (x < self.edges[-1])
        if not np.any(m): return
        x = x[m]
        w = w[m].astype(np.float64)
        f = (x - self.x_min) / self.dx
        i0 = np.floor(f).astype(int)
        frac = f - i0
        i1 = i0 + 1
        np.add.at(self.counts, i0, w * (1.0 - frac))
        inb = i1 < self.counts.size
        np.add.at(self.counts, i1[inb], w[inb] * frac[inb])

    def density(self) -> np.ndarray:
        total = self.counts.sum()
        if total <= 0:
            return np.zeros_like(self.counts)
        return self.counts / (total * self.dx)

# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="1/r–corrected aggregate in linear r-space.")
    ap.add_argument("--per-cloud-dir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--drt", type=float, default=0.02, help="Bin width in r_tilde = r / R_eq.")
    ap.add_argument("--rt_margin", type=float, default=0.02, help="Pad around observed r_tilde range.")
    ap.add_argument("--include_2pi", action="store_true",
                    help="Also divide by 2π (pure scale factor).")
    ap.add_argument("--include_binwidth", action="store_true",
                    help="Also divide by Δr (pure scale factor given fixed Δr).")
    ap.add_argument("--inner-radius-px", type=float, default=0.0,
                    help="Ignore r < this many pixels before correction.")
    ap.add_argument("--cloud-weighting", choices=["length","equal"], default="length",
                    help="Sum raw weights (length) or normalize each cloud to unit mass (equal).")
    ap.add_argument("--size-terciles", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--tail-dmin", type=float, default=0.05,
                    help="Lower bound for Δr_tilde tail fits (linear units).")
    ap.add_argument("--tail-dmax", type=float, default=0.40,
                    help="Upper bound for Δr_tilde tail fits (linear units).")
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "plots"
    agg_dir = outdir / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    if args.plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    dset = ds.dataset(args.per_cloud_dir, format="parquet")
    cols = ["cloud_idx","area","rp_r","rp_counts"]

    # -------- PASS 1: find r_tilde range and size splits --------
    rt_min_obs, rt_max_obs = +np.inf, -np.inf
    areas = []
    r_bin_width_px = None

    for row in _iter_rows(dset, cols):
        area = int(row["area"])
        r = np.asarray(row["rp_r"], dtype=np.float64)
        c = np.asarray(row["rp_counts"], dtype=np.float64)
        if r.size == 0 or c.sum() == 0: continue
        if r_bin_width_px is None and r.size >= 2:
            r_bin_width_px = float(r[1] - r[0])
        if args.inner_radius_px > 0:
            keep = r >= args.inner_radius_px
            if not np.any(keep): continue
            r, c = r[keep], c[keep]
        R = math.sqrt(area / PI)
        rt = r / R
        nz = c > 0
        if np.any(nz):
            rt_min_obs = min(rt_min_obs, float(rt[nz].min()))
            rt_max_obs = max(rt_max_obs, float(rt[nz].max()))
        areas.append(area)

    if not areas:
        print("[WARN] no valid clouds")
        return

    pad = args.rt_margin * (rt_max_obs - rt_min_obs if rt_max_obs > rt_min_obs else 1.0)
    rt_min = max(0.0, rt_min_obs - pad)
    rt_max = rt_max_obs + pad

    terc_cuts = None
    if args.size_terciles:
        q1, q2 = np.quantile(areas, [1/3, 2/3])
        terc_cuts = (q1, q2)

    # -------- initialize binners --------
    H_rt_all   = LinearBinner(rt_min, rt_max, args.drt)
    H_drt_all  = LinearBinner(- (rt_max-rt_min), (rt_max-rt_min), args.drt)  # roomy Δr_tilde
    H_rt_small = LinearBinner(rt_min, rt_max, args.drt) if terc_cuts else None
    H_rt_mid   = LinearBinner(rt_min, rt_max, args.drt) if terc_cuts else None
    H_rt_large = LinearBinner(rt_min, rt_max, args.drt) if terc_cuts else None

    total_mass = 0.0

    # -------- PASS 2: accumulate --------
    for row in _iter_rows(dset, cols):
        area = int(row["area"])
        r = np.asarray(row["rp_r"], dtype=np.float64)
        c = np.asarray(row["rp_counts"], dtype=np.float64)
        if r.size == 0 or c.sum() == 0: continue

        if args.inner_radius_px > 0:
            keep = r >= args.inner_radius_px
            if not np.any(keep): continue
            r, c = r[keep], c[keep]

        R = math.sqrt(area / PI)
        rt = r / R

        # --- 1/r correction weights ---
        w = c / np.maximum(r, 1e-12)
        if args.include_2pi:
            w = w / (2.0 * PI)
        if args.include_binwidth and r_bin_width_px is not None:
            w = w / r_bin_width_px  # only a global scale if Δr is fixed across clouds

        # per-cloud normalization (equal weight per cloud) if requested
        if args.cloud_weighting == "equal":
            w = w / max(w.sum(), 1.0)

        # re-centering in linear r_tilde
        rt_med = _weighted_median(rt, w if args.cloud_weighting=="equal" else c)
        drt = rt - rt_med

        # accumulate
        H_rt_all.splat(rt, w)
        H_drt_all.splat(drt, w if args.cloud_weighting=="equal" else c)

        if terc_cuts:
            target = H_rt_small if area <= terc_cuts[0] else H_rt_mid if area <= terc_cuts[1] else H_rt_large
            target.splat(rt, w)

        total_mass += float(w.sum())

    # -------- densities & metrics --------
    rt_centers = H_rt_all.centers
    den_rt = H_rt_all.density()

    # peakiness near center window (say |rt - 1| <= 0.2 roughly ~ ln case)
    # we'll center in linear by median of the aggregate distribution
    # (for simplicity: take median bin index weighted by counts)
    agg_med_idx = np.searchsorted(np.cumsum(H_rt_all.counts)/max(H_rt_all.counts.sum(),1), 0.5)
    rt_med_agg = rt_centers[min(max(agg_med_idx,0), len(rt_centers)-1)]
    win = np.abs(rt_centers - rt_med_agg) <= 0.2
    if np.any(win):
        peak = float(den_rt[win].max())
        med  = float(np.median(den_rt[win]))
        P = peak/med if med>0 else np.inf
    else:
        P = np.nan

    # survival from Δr_tilde (linear window)
    drt_centers = H_drt_all.centers
    drt_counts  = H_drt_all.counts
    tot = drt_counts.sum()
    # right: Δr_tilde >= δ
    mask_r = drt_centers >= 0
    x_r = drt_centers[mask_r]
    y_r = drt_counts[mask_r]
    S_r = np.cumsum(y_r[::-1])[::-1] / max(tot,1)
    # left: Δr_tilde <= -δ
    mask_l = drt_centers <= 0
    x_l = -drt_centers[mask_l]
    y_l =  drt_counts[mask_l]
    S_l = np.cumsum(y_l) / max(tot,1)

    # fit in [tail-dmin, tail-dmax]
    def fit_tail(x, S, dmin, dmax):
        m = (x >= dmin) & (x <= dmax) & (S > 0)
        if not np.any(m):
            return dict(slope=np.nan, intercept=np.nan, r2=np.nan, n=0, window=[dmin,dmax])
        a,b,r2 = _fit_line(x[m], np.log(S[m]))
        return dict(slope=a, intercept=b, r2=r2, n=int(m.sum()), window=[dmin,dmax])

    fit_right = fit_tail(x_r, S_r, args.tail_dmin, args.tail_dmax)
    fit_left  = fit_tail(x_l, S_l, args.tail_dmin, args.tail_dmax)

    # -------- save artifacts --------
    agg_dir.joinpath("r_tilde_hist.parquet").write_bytes(
        pd.DataFrame({"r_tilde": rt_centers, "counts": H_rt_all.counts, "density": den_rt}).to_parquet()
    )
    agg_dir.joinpath("delta_rt_hist.parquet").write_bytes(
        pd.DataFrame({"delta_r_tilde": drt_centers, "counts": H_drt_all.counts}).to_parquet()
    )
    if terc_cuts:
        for name, H in [("small", H_rt_small), ("mid", H_rt_mid), ("large", H_rt_large)]:
            agg_dir.joinpath(f"r_tilde_hist_{name}.parquet").write_bytes(
                pd.DataFrame({"r_tilde": H.centers, "counts": H.counts}).to_parquet()
            )

    metrics = {
        "r_tilde_min": float(rt_centers.min()) if len(rt_centers) else None,
        "r_tilde_max": float(rt_centers.max()) if len(rt_centers) else None,
        "bin_width_r_tilde": args.drt,
        "peakiness_P": float(P),
        "right_tail_fit": fit_right,
        "left_tail_fit":  fit_left,
        "cloud_weighting": args.cloud_weighting,
        "inner_radius_px": args.inner_radius_px,
        "include_2pi": bool(args.include_2pi),
        "include_binwidth": bool(args.include_binwidth),
        "size_terciles": (terc_cuts if terc_cuts else None),
    }
    (outdir / "metrics.json").write_text(pd.Series(metrics).to_json(indent=2))

    # -------- plots --------
    if args.plots:
        # main
        plt.figure()
        plt.plot(rt_centers, den_rt, lw=1.6)
        plt.xlabel(r"$\tilde r = r / R_{\rm eq}$")
        plt.ylabel("Density (1/r corrected)")
        plt.title("Aggregate H_corr( r~ )")
        plt.tight_layout()
        plt.savefig(plot_dir / "agg_rtilde_density.png", dpi=160)

        # size-split overlays
        if terc_cuts:
            plt.figure()
            for name, H in [("small", H_rt_small), ("mid", H_rt_mid), ("large", H_rt_large)]:
                den = (H.counts / (H.counts.sum() * H.dx)) if H.counts.sum()>0 else np.zeros_like(H.counts)
                plt.plot(H.centers, den, label=name, lw=1.4)
            plt.xlabel(r"$\tilde r$")
            plt.ylabel("Density")
            plt.title("H_corr( r~ ) by size tercile")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "size_split_rtilde.png", dpi=160)

        # survival curves from Δr_tilde
        plt.figure()
        plt.semilogy(x_r, S_r, label="right (Δr~ ≥ δ)")
        plt.semilogy(x_l, S_l, label="left  (Δr~ ≤ -δ)")
        # fitted lines
        for lbl, x, fit in [("right", x_r, fit_right), ("left", x_l, fit_left)]:
            if fit["n"] > 1 and np.isfinite(fit["slope"]):
                xx = np.linspace(args.tail_dmin, args.tail_dmax, 50)
                yy = np.exp(fit["slope"]*xx + fit["intercept"])
                plt.semilogy(xx, yy, "--", label=f"{lbl} fit (slope={fit['slope']:.3f})")
        plt.xlabel(r"$\delta$ (|Δr~|)")
        plt.ylabel("Survival S(δ)")
        plt.title("Two-sided tail survival (Δr~) with 1/r correction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "survival_delta_rtilde.png", dpi=160)

    print("[OK] Finished:", outdir)

if __name__ == "__main__":
    main()
