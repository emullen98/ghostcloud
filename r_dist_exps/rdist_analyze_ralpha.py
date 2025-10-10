#!/usr/bin/env python3
"""
rdist_analyze_ralpha.py

Like the 1/r analysis, but generalized to r^(-alpha).
You can pass a grid of alpha values (e.g., "1,1.3333,1.5") and compare overlays.

Inputs: per-cloud Parquet shards from siteperc_rprof_from_model.py
Columns used: cloud_idx, area, rp_r, rp_counts

For each alpha:
  - r_tilde = r / R_eq  with R_eq = sqrt(area/pi)
  - weights w_alpha = rp_counts / r^alpha  (optional: / (2*pi) / Δr for pure scaling)
  - aggregate H_alpha(r_tilde), plus Δr_tilde survival and tail fits

Outputs:
  - agg/rtilde_hist_alpha_<alpha>.parquet  (counts & density)
  - agg/delta_rt_hist_alpha_<alpha>.parquet
  - metrics.json (per-alpha peakiness & left/right tail fit)
  - plots: overlays across alpha, survival across alpha, size-terciles for a chosen alpha

Usage example:
python -m clouds.r_dist_exps.rdist_analyze_ralpha \
  --per-cloud-dir scratch/expC/per_cloud/expC_internal_W10000_H10000_p0.407400_seed987 \
  --outdir scratch/expC/analysis_ralpha/expC_internal_W10000_H10000_p0.407400_seed987 \
  --alpha-grid 1,1.35,1.5,1.7 \
  --drt 0.02 \
  --plots \
  --cloud-weighting length \
  --inner-radius-px 0
  
"""

from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt

PI = math.pi

# ---------------- Helpers ----------------

def _iter_rows(dataset: ds.Dataset, cols: List[str]) -> Iterable[Dict]:
    sc = dataset.scanner(columns=cols)
    for batch in sc.to_batches():
        tbl = pa.Table.from_batches([batch])
        lists = [tbl[c].to_pylist() for c in cols]
        for vals in zip(*lists):
            yield {k: v for k, v in zip(cols, vals)}

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order].astype(np.float64)
    cdf = np.cumsum(w) / np.sum(w)
    i = np.searchsorted(cdf, 0.5)
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

class LinearBinner:
    def __init__(self, x_min: float, x_max: float, dx: float):
        self.x_min = float(x_min); self.x_max = float(x_max); self.dx = float(dx)
        nb = int(np.floor((self.x_max - self.x_min) / self.dx)) + 1
        self.edges = self.x_min + np.arange(nb + 1) * self.dx
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.counts = np.zeros(nb, dtype=np.float64)
    def splat(self, x: np.ndarray, w: np.ndarray) -> None:
        m = (x >= self.edges[0]) & (x < self.edges[-1])
        if not np.any(m): return
        x = x[m]; w = w[m].astype(np.float64)
        f = (x - self.x_min) / self.dx
        i0 = np.floor(f).astype(int); frac = f - i0; i1 = i0 + 1
        np.add.at(self.counts, i0, w * (1.0 - frac))
        inb = i1 < self.counts.size
        np.add.at(self.counts, i1[inb], w[inb] * frac[inb])
    def density(self) -> np.ndarray:
        tot = self.counts.sum()
        return self.counts / (tot * self.dx) if tot > 0 else np.zeros_like(self.counts)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate with r^(-alpha) correction (alpha sweep).")
    ap.add_argument("--per-cloud-dir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--alpha-grid", type=str, default="1",
                    help='Comma list, e.g. "1,1.3333,1.5,1.7"')
    ap.add_argument("--drt", type=float, default=0.02, help="Bin width for r_tilde.")
    ap.add_argument("--rt-margin", type=float, default=0.02)
    ap.add_argument("--inner-radius-px", type=float, default=0.0,
                    help="Ignore r < this many px before correction.")
    ap.add_argument("--cloud-weighting", choices=["length","equal"], default="length",
                    help="Sum raw weights or normalize each cloud to unit mass.")
    ap.add_argument("--size-terciles", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--include-2pi", action="store_true", help="Divide weights by 2π (pure scale).")
    ap.add_argument("--include-binwidth", action="store_true", help="Divide by Δr (pure scale).")
    ap.add_argument("--tail-dmin", type=float, default=0.05)
    ap.add_argument("--tail-dmax", type=float, default=0.40)
    ap.add_argument("--tercile-alpha", type=float, default=None,
                    help="If set, also emit size-tercile overlays for this α.")
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "plots"
    agg_dir = outdir / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    if args.plots: plot_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in args.alpha_grid.split(",") if a.strip()]

    dset = ds.dataset(args.per_cloud_dir, format="parquet")
    cols = ["cloud_idx","area","rp_r","rp_counts"]

    # PASS 1: range + terciles
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
            m = r >= args.inner_radius_px
            if not np.any(m): continue
            r, c = r[m], c[m]
        R = math.sqrt(area / PI)
        rt = r / R
        nz = c > 0
        if np.any(nz):
            rt_min_obs = min(rt_min_obs, float(rt[nz].min()))
            rt_max_obs = max(rt_max_obs, float(rt[nz].max()))
        areas.append(area)
    if not areas:
        print("[WARN] No valid clouds."); return
    pad = args.rt_margin * (rt_max_obs - rt_min_obs if rt_max_obs > rt_min_obs else 1.0)
    rt_min = max(0.0, rt_min_obs - pad); rt_max = rt_max_obs + pad

    terc_cuts = None
    if args.size_terciles:
        q1, q2 = np.quantile(areas, [1/3, 2/3])
        terc_cuts = (q1, q2)

    # Prepare per-alpha binners
    H_rt = {alpha: LinearBinner(rt_min, rt_max, args.drt) for alpha in alphas}
    H_drt = {alpha: LinearBinner(-(rt_max-rt_min), (rt_max-rt_min), args.drt) for alpha in alphas}
    size_sets = {}
    if terc_cuts and args.tercile_alpha is not None and args.tercile_alpha in alphas:
        a = args.tercile_alpha
        size_sets[a] = {
            "small": LinearBinner(rt_min, rt_max, args.drt),
            "mid":   LinearBinner(rt_min, rt_max, args.drt),
            "large": LinearBinner(rt_min, rt_max, args.drt),
        }

    # PASS 2: accumulate
    for row in _iter_rows(dset, cols):
        area = int(row["area"])
        r = np.asarray(row["rp_r"], dtype=np.float64)
        c = np.asarray(row["rp_counts"], dtype=np.float64)
        if r.size == 0 or c.sum() == 0: continue
        if args.inner_radius_px > 0:
            m = r >= args.inner_radius_px
            if not np.any(m): continue
            r, c = r[m], c[m]
        R = math.sqrt(area / PI)
        rt = r / R

        # base weights (we will reuse for each alpha)
        base = c.astype(np.float64)
        for alpha in alphas:
            w = base / np.power(np.maximum(r, 1e-12), alpha)
            if args.include_2pi: w = w / (2.0 * PI)   # pure scaling
            if args.include_binwidth and r_bin_width_px is not None:
                w = w / r_bin_width_px             # pure scaling if Δr is shared
            if args.cloud_weighting == "equal":
                w = w / max(w.sum(), 1.0)

            # recentre in *linear* r_tilde using weighted median (use w for symmetry)
            rt_c = _weighted_median(rt, w)
            drt = rt - rt_c

            H_rt[alpha].splat(rt, w)
            H_drt[alpha].splat(drt, w)

            if size_sets and alpha == args.tercile_alpha:
                tgt = size_sets[alpha]["small"] if area <= terc_cuts[0] else \
                      size_sets[alpha]["mid"]   if area <= terc_cuts[1] else \
                      size_sets[alpha]["large"]
                tgt.splat(rt, w)

    # Metrics & save
    all_metrics = {}
    for alpha in alphas:
        rt_cent = H_rt[alpha].centers
        den = H_rt[alpha].density()
        pd.DataFrame({"r_tilde": rt_cent, "counts": H_rt[alpha].counts, "density": den}) \
          .to_parquet(agg_dir / f"rtilde_hist_alpha_{alpha:g}.parquet")

        drt_cent = H_drt[alpha].centers
        drt_cnt  = H_drt[alpha].counts
        pd.DataFrame({"delta_r_tilde": drt_cent, "counts": drt_cnt}) \
          .to_parquet(agg_dir / f"delta_rt_hist_alpha_{alpha:g}.parquet")

        # Peakiness around aggregate median (±0.2 in r_tilde)
        cdf = np.cumsum(H_rt[alpha].counts) / max(H_rt[alpha].counts.sum(), 1)
        mid_idx = np.searchsorted(cdf, 0.5)
        mid_idx = min(max(mid_idx, 0), len(rt_cent)-1)
        rt_mid = rt_cent[mid_idx]
        win = np.abs(rt_cent - rt_mid) <= 0.2
        P = (den[win].max() / np.median(den[win])) if np.any(win) and np.median(den[win])>0 else np.nan

        # Two-sided tail fits on Δr_tilde in [tail-dmin, tail-dmax]
        tot = drt_cnt.sum()
        mask_r = drt_cent >= 0; x_r = drt_cent[mask_r]; y_r = drt_cnt[mask_r]
        S_r = np.cumsum(y_r[::-1])[::-1] / max(tot,1)
        mask_l = drt_cent <= 0; x_l = -drt_cent[mask_l]; y_l = drt_cnt[mask_l]
        S_l = np.cumsum(y_l) / max(tot,1)

        def fit_tail(x, S, dmin, dmax):
            m = (x >= dmin) & (x <= dmax) & (S > 0)
            if not np.any(m):
                return dict(slope=np.nan, intercept=np.nan, r2=np.nan, n=0, window=[dmin,dmax])
            a,b,r2 = _fit_line(x[m], np.log(S[m]))
            return dict(slope=a, intercept=b, r2=r2, n=int(m.sum()), window=[dmin,dmax])

        fit_right = fit_tail(x_r, S_r, args.tail_dmin, args.tail_dmax)
        fit_left  = fit_tail(x_l, S_l, args.tail_dmin, args.tail_dmax)

        all_metrics[str(alpha)] = {
            "peakiness_P": float(P),
            "right_tail_fit": fit_right,
            "left_tail_fit":  fit_left,
        }

    (outdir / "metrics.json").write_text(pd.Series(all_metrics).to_json(indent=2))

    # Plots
    if args.plots:
        # Overlay densities across alpha
        plt.figure()
        for alpha in alphas:
            rt_cent = H_rt[alpha].centers
            den = H_rt[alpha].density()
            plt.plot(rt_cent, den, label=f"α={alpha:g}", lw=1.4)
        plt.xlabel(r"$\tilde r = r / R_{\rm eq}$")
        plt.ylabel(r"Density with $r^{-\alpha}$ correction")
        plt.title("H_α( r~ ) overlays")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "rtilde_alpha_overlays.png", dpi=160)

        # Survival overlays across alpha
        plt.figure()
        for alpha in alphas:
            drt_cent = H_drt[alpha].centers
            drt_cnt  = H_drt[alpha].counts
            tot = drt_cnt.sum()
            mR = drt_cent >= 0; xR = drt_cent[mR]; yR = drt_cnt[mR]
            SR = np.cumsum(yR[::-1])[::-1] / max(tot,1)
            mL = drt_cent <= 0; xL = -drt_cent[mL]; yL = drt_cnt[mL]
            SL = np.cumsum(yL) / max(tot,1)
            plt.semilogy(xR, SR, label=f"right α={alpha:g}")
            plt.semilogy(xL, SL, "--", label=f"left  α={alpha:g}")
        plt.xlabel(r"$\delta$ (|Δr~|)")
        plt.ylabel("Survival S(δ)")
        plt.title("Two-sided survival across α")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(plot_dir / "survival_alpha_overlays.png", dpi=160)

        # Size-terciles for a chosen alpha (if requested)
        if size_sets:
            a = args.tercile_alpha
            plt.figure()
            for name, H in size_sets[a].items():
                den = H.density()
                plt.plot(H.centers, den, label=name, lw=1.4)
            plt.xlabel(r"$\tilde r$")
            plt.ylabel(r"Density with $r^{-\alpha}$ correction")
            plt.title(f"Size-tercile overlays (α={a:g})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"size_terciles_alpha_{a:g}.png", dpi=160)

    print("[OK] Finished:", outdir)

if __name__ == "__main__":
    main()
