#!/usr/bin/env python3
"""
plot_riskset_scaled_rprof.py

Boundary radial profiles with censoring-aware (risk-set) aggregation and
optional scale collapse:
  --norm none  : r in pixels (physical)
  --norm rmax  : u = r / r_max  (support in [0,1])
  --norm rg    : u = r / R_g    (support varies; default u in [0, u_max])

Risk-set averaging removes size-survival leakage by averaging only over
clouds that have support at the bin (Kaplan–Meier style). We report:
  - riskset_mean (unweighted, one-cloud-one-vote)
  - riskset_median (unweighted)
  - riskset_mean (perimeter-weighted)
Also plots the classic perimeter-weighted aggregate (only when norm=none).

You may optionally filter clouds by size (area/perim).
"""

from __future__ import annotations
import argparse, math, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None
    import pandas as pd  # type: ignore


# ---------------- IO ----------------

def _infer_delta_r(r: np.ndarray) -> float:
    if r.size < 3: return float("nan")
    d = np.diff(r); d = d[d > 0]
    return float(np.median(d)) if d.size else float("nan")


def _load_rows(per_cloud_dir: Path,
               needed=("area","perim","rp_r","rp_counts","rp_f_ring")):
    files = sorted(per_cloud_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {per_cloud_dir}")
    if pq is not None:
        cols = list(needed)
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(columns=cols):
                tbl = pa.Table.from_batches([batch])
                names = set(tbl.schema.names)
                def get(name): return tbl[name] if name in names else None
                N = tbl.num_rows
                for i in range(N):
                    row = {}
                    # scalars
                    for nm in ("area","perim"):
                        col = get(nm); row[nm] = col[i].as_py() if col is not None else None
                    # lists
                    for nm in ("rp_r","rp_counts","rp_f_ring"):
                        col = get(nm)
                        if col is None: row[nm] = None
                        else:
                            val = col[i].as_py()
                            row[nm] = None if val is None else np.asarray(val, dtype=np.float64)
                    yield row
    else:
        usecols = list(needed)
        for fp in files:
            df = pd.read_parquet(fp, columns=usecols)
            for _, r in df.iterrows():
                row = {"area": r.get("area",None), "perim": r.get("perim",None)}
                for nm in ("rp_r","rp_counts","rp_f_ring"):
                    val = r.get(nm, None)
                    row[nm] = None if val is None else np.asarray(val, dtype=np.float64)
                yield row


# -------------- Per-cloud transforms --------------

def _compute_fring_if_missing(r: np.ndarray, cnt: np.ndarray, delta_r: float, fring: np.ndarray|None):
    if fring is not None: return fring.astype(np.float64, copy=False)
    denom = 2.0 * math.pi * r * delta_r
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(denom > 0, cnt / denom, np.nan)
    return y

def _rmax_index(cnt: np.ndarray) -> int:
    nz = np.where(cnt > 0)[0]
    return int(nz[-1]) if nz.size else -1

def _radius_of_gyration(r: np.ndarray, cnt: np.ndarray) -> float:
    s = np.sum(cnt)
    if s <= 0: return float("nan")
    return float(np.sqrt(np.sum(cnt * r * r) / s))

def _interp_to_grid(x_src: np.ndarray, y_src: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    m = np.isfinite(x_src) & np.isfinite(y_src)
    if m.sum() < 2:
        return np.full_like(x_grid, np.nan, dtype=np.float64)
    xs, ys = x_src[m], y_src[m]
    # Clamp outside range to NaN (not extrapolating)
    y = np.interp(x_grid, xs, ys, left=np.nan, right=np.nan)
    # np.interp doesn't propagate NaN outside — fix edges:
    y[x_grid < xs[0]] = np.nan
    y[x_grid > xs[-1]] = np.nan
    return y


# -------------- Aggregations --------------

def _riskset_stats(matrix: np.ndarray, support_mask: np.ndarray, weights: np.ndarray|None):
    """
    matrix: shape (N,L) of values (can contain NaN)
    support_mask: (N,L) boolean, True if cloud i has support at bin k
    weights: optional shape (N,) weights per cloud (e.g., perimeter)

    Returns: mean_unweighted, median, Mk, mean_weighted (or None)
    """
    N, L = matrix.shape
    Mk = support_mask.sum(axis=0).astype(np.int64)

    # replace NaN by 0 just for summations; mask controls inclusion
    A = np.nan_to_num(matrix, copy=False, nan=0.0)

    # unweighted mean over risk set
    sum_over = (A * support_mask).sum(axis=0)
    mean_unw = np.where(Mk > 0, sum_over / np.maximum(Mk, 1), np.nan)

    # weighted mean over risk set (perim weights)
    if weights is not None:
        W = weights[:, None] * support_mask
        wsum = W.sum(axis=0)
        mean_w = np.where(wsum > 0, (A * W).sum(axis=0) / wsum, np.nan)
    else:
        mean_w = None

    # median over risk set (unweighted)
    med = np.full(L, np.nan, dtype=np.float64)
    for k in range(L):
        if Mk[k] == 0: continue
        col = matrix[:, k]
        col = col[support_mask[:, k]]
        col = col[np.isfinite(col)]
        if col.size:
            med[k] = float(np.median(col))
    return mean_unw, med, Mk, mean_w


# -------------- Pipeline --------------

def compute_curves(per_cloud_dir: Path,
                   metric: str, min_size: float|None, max_size: float|None,
                   norm: str, u_max: float, du: float):
    """
    Returns dict with:
      grid_x  : r (pixels) if norm=none, else u (normalized)
      classic_fring : perimeter-weighted classic (only when norm=none)
      risk_mean, risk_median, risk_mean_wt
      Mk : risk set sizes
    """
    rows = _load_rows(per_cloud_dir)

    # Collect per-cloud raw data
    clouds = []  # list of dicts per cloud
    for row in rows:
        size_val = row.get(metric, None)
        if size_val is None: continue
        if (min_size is not None and size_val < min_size) or (max_size is not None and size_val >= max_size):
            continue

        r = row["rp_r"]; cnt = row["rp_counts"]; fr = row.get("rp_f_ring", None)
        if r is None or cnt is None: continue
        r = np.asarray(r, dtype=np.float64); cnt = np.asarray(cnt, dtype=np.float64)
        if r.size < 2 or cnt.size != r.size: continue

        delta_r = _infer_delta_r(r)
        fr = _compute_fring_if_missing(r, cnt, delta_r, fr)

        rmax_idx = _rmax_index(cnt)
        if rmax_idx <= 1:  # too tiny
            continue
        rmax = r[rmax_idx]
        Rg = _radius_of_gyration(r, cnt)

        perim_wt = row.get("perim", 1.0) or 1.0

        clouds.append(dict(r=r, cnt=cnt, fr=fr, rmax=rmax, Rg=Rg, perim=perim_wt))

    if not clouds:
        raise RuntimeError("No clouds after filtering.")

    # Classic perimeter-weighted (physical r)
    classic_fring = None
    r_phys = None
    if norm == "none":
        # align by padding to longest r
        Lmax = max(c["r"].size for c in clouds)
        # choose reference r grid: from the longest cloud
        ref = max(clouds, key=lambda c: c["r"].size)["r"]
        r_phys = ref.copy()
        counts_stack = []
        for c in clouds:
            cnt = c["cnt"]
            if cnt.size < Lmax:
                cnt = np.pad(cnt, (0, Lmax - cnt.size))
            counts_stack.append(cnt)
        counts_sum = np.sum(np.vstack(counts_stack), axis=0).astype(np.float64)
        delta_r_ref = _infer_delta_r(r_phys)
        denom = 2.0 * math.pi * r_phys * delta_r_ref
        with np.errstate(divide="ignore", invalid="ignore"):
            classic_fring = np.where(denom > 0, counts_sum / denom, np.nan)

    # Build normalized grid
    if norm == "none":
        grid_x = r_phys
        # For risk-set on physical grid, we also need to align/interp clouds onto r_phys
        # (use per-cloud as-is then pad)
        L = grid_x.size
        mat = np.full((len(clouds), L), np.nan, dtype=np.float64)
        sup = np.zeros((len(clouds), L), dtype=bool)
        for i, c in enumerate(clouds):
            r, fr = c["r"], c["fr"]
            Li = r.size
            mat[i, :Li] = fr
            sup[i, :Li] = ~np.isnan(fr)
            # ensure no support beyond rmax
            sup[i, Li:] = False
        weights = np.asarray([c["perim"] for c in clouds], dtype=np.float64)

    else:
        # choose normalized grid
        if norm == "rmax":
            grid_x = np.arange(0.0, 1.0 + 1e-9, du)  # u in [0,1]
        elif norm == "rg":
            grid_x = np.arange(0.0, u_max + 1e-9, du)
        else:
            raise ValueError("--norm must be one of none|rmax|rg")

        L = grid_x.size
        mat = np.full((len(clouds), L), np.nan, dtype=np.float64)
        sup = np.zeros((len(clouds), L), dtype=bool)
        weights = np.asarray([c["perim"] for c in clouds], dtype=np.float64)

        for i, c in enumerate(clouds):
            r, fr, rmax, Rg = c["r"], c["fr"], c["rmax"], c["Rg"]
            if norm == "rmax":
                scale = rmax if rmax > 0 else np.nan
            else:
                scale = Rg if (Rg is not None and np.isfinite(Rg) and Rg > 0) else np.nan
            if not (np.isfinite(scale) and scale > 0):
                continue

            u_src = r / scale
            # We only trust values up to the real support: u <= rmax/scale
            u_max_i = rmax / scale
            y = _interp_to_grid(u_src, fr, grid_x)
            # Build support mask
            sup_i = np.isfinite(y) & (grid_x <= (u_max_i + 1e-9))
            mat[i, :] = y
            sup[i, :] = sup_i

    # Risk-set stats
    risk_mean, risk_median, Mk, risk_mean_w = _riskset_stats(mat, sup, weights=weights)

    out = {
        "grid_x": grid_x,
        "classic_fring": classic_fring,
        "risk_mean": risk_mean,
        "risk_median": risk_median,
        "risk_mean_w": risk_mean_w,
        "Mk": Mk,
        "n_clouds": len(clouds),
    }
    return out


# -------------- Plotting & CLI --------------

def _plot_curves(x, curves: dict[str,np.ndarray], xlabel: str, title: str, outdir: Path, base: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for mode, fn in (("linear", plt.plot), ("semilogy", plt.semilogy), ("loglog", plt.loglog)):
        plt.figure()
        for label, y in curves.items():
            if y is None: continue
            if mode == "loglog":
                m = (x > 0) & np.isfinite(y) & (y > 0)
                fn(x[m], y[m], label=label)
            elif mode == "semilogy":
                m = (x >= 0) & np.isfinite(y) & (y >= 0)
                fn(x[m], y[m], label=label)
            else:
                m = np.isfinite(y)
                fn(x[m], y[m], label=label)
        plt.xlabel(xlabel)
        plt.ylabel("f_ring(r)")
        plt.title(f"{title} — {mode}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{base}_{mode}.png", dpi=220)
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Risk-set aggregates with optional scale collapse.")
    ap.add_argument("--per-cloud-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["area","perim"], default="area")
    ap.add_argument("--min-size", type=float, default=None)
    ap.add_argument("--max-size", type=float, default=None)
    ap.add_argument("--norm", choices=["none","rmax","rg"], default="none",
                    help="Normalization: none (pixels), rmax, or rg (radius of gyration).")
    ap.add_argument("--u-max", type=float, default=4.0, help="Only for --norm rg; max u to display.")
    ap.add_argument("--du", type=float, default=0.02, help="Normalized bin step (for rmax/rg).")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--save-txt", action="store_true")
    args = ap.parse_args()

    res = compute_curves(
        per_cloud_dir=args.per_cloud_dir,
        metric=args.metric, min_size=args.min_size, max_size=args.max_size,
        norm=args.norm, u_max=args.u_max, du=args.du
    )

    # Assemble curves for plotting
    title = f"{args.metric}" + (f" ∈ [{args.min_size:.0f}, {args.max_size:.0f})" if (args.min_size is not None and args.max_size is not None) else "")
    title += f" | n={res['n_clouds']} | norm={args.norm}"
    base = args.tag or (f"riskset_{args.metric}"
                        + (f"_{int(args.min_size)}to{int(args.max_size)}" if args.min_size is not None and args.max_size is not None else "")
                        + f"_{args.norm}_n{res['n_clouds']}")

    if args.norm == "none":
        xlabel = "r (pixels)"
        curves = {
            "classic_sum_counts (perim-wt)": res["classic_fring"],
            "riskset_mean (unweighted)": res["risk_mean"],
            "riskset_median (unweighted)": res["risk_median"],
            "riskset_mean (perim-wt)": res["risk_mean_w"],
        }
    else:
        xlabel = "u = r / scale"
        curves = {
            "riskset_mean (unweighted)": res["risk_mean"],
            "riskset_median (unweighted)": res["risk_median"],
            "riskset_mean (perim-wt)": res["risk_mean_w"],
        }

    _plot_curves(res["grid_x"], curves, xlabel, title, args.outdir, base)

    # Save artifacts
    meta = {
        "per_cloud_dir": str(args.per_cloud_dir),
        "metric": args.metric, "min_size": args.min_size, "max_size": args.max_size,
        "norm": args.norm, "u_max": args.u_max, "du": args.du,
        "n_clouds": int(res["n_clouds"]),
        "outputs": {k: str(args.outdir / f"{base}_{k}.png") for k in ("linear","semilogy","loglog")},
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / f"{base}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_txt:
        np.savetxt(args.outdir / f"{base}_x.txt", res["grid_x"])
        if res["classic_fring"] is not None:
            np.savetxt(args.outdir / f"{base}_classic_fring.txt", res["classic_fring"])
        np.savetxt(args.outdir / f"{base}_riskset_mean.txt", res["risk_mean"])
        np.savetxt(args.outdir / f"{base}_riskset_median.txt", res["risk_median"])
        np.savetxt(args.outdir / f"{base}_riskset_mean_wt.txt",
                   res["risk_mean_w"] if res["risk_mean_w"] is not None else np.full_like(res["grid_x"], np.nan))
        np.savetxt(args.outdir / f"{base}_riskset_Mk.txt", res["Mk"].astype(np.int64))

    print(f"[OK] norm={args.norm} n={res['n_clouds']} ; plots -> {args.outdir}")

if __name__ == "__main__":
    main()
