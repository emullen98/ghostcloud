#!/usr/bin/env python3
"""
plot_riskset_agg_rprof.py

Goal
-----
Aggregate boundary radial profiles while *removing* size-survival leakage.
At each radius bin r_k, average per-cloud f_ring_i(r_k) only over clouds
that actually have support there (r_k <= rmax_i). This "risk-set" averaging
is analogous to censoring-aware means and largely decouples the curve from
the cloud-size distribution.

Outputs
-------
- Classic perimeter-weighted aggregate (sum of counts).
- Risk-set MEAN of per-cloud f_ring (one-cloud-one-vote).
- Risk-set MEDIAN of per-cloud f_ring.
- (Optional) Perimeter-weighted risk-set mean.
- Plots: linear / semilogy / log-log overlays.
- Optional TXT dumps of r, classic, mean, median, and risk-set sizes M_k.

Examples
--------
# Basic (no size bin)
python plot_riskset_agg_rprof.py \
  --per-cloud-dir scratch/expA/per_cloud/expA_internal_W4000_H2666_p0.407400_seed987 \
  --outdir scratch/expA/riskset

# With size bin on area: [3000, 3300)
python -m clouds.r_dist_exps.plot_riskset_agg_rprof \
  --per-cloud-dir scratch/expC/per_cloud/expC_internal_W10000_H10000_p0.407400_seed987 \
  --metric area --min-size 3000 --max-size 3300 \
  --outdir scratch/expC/riskset --tag area_3000_3300

# Also dump TXT arrays
python -m clouds.r_dist_exps.plot_riskset_agg_rprof \
  --per-cloud-dir ... --outdir ... --save-txt
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None
    import pandas as pd  # type: ignore


# -------------------------- IO helpers --------------------------

def _infer_delta_r(r: np.ndarray) -> float:
    if r.size < 3:
        return float("nan")
    d = np.diff(r)
    d = d[d > 0]
    return float(np.median(d)) if d.size else float("nan")


def _load_rows(per_cloud_dir: Path,
               needed=("area", "perim", "rp_r", "rp_counts", "rp_f_ring")):
    files = sorted(per_cloud_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {per_cloud_dir}")

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
                    for nm in ("area", "perim"):
                        col = get(nm)
                        row[nm] = col[i].as_py() if col is not None else None
                    for nm in ("rp_r", "rp_counts", "rp_f_ring"):
                        col = get(nm)
                        if col is None:
                            row[nm] = None
                        else:
                            val = col[i].as_py()  # list or None
                            row[nm] = None if val is None else np.asarray(val, dtype=np.float64)
                    yield row
    else:
        usecols = [c for c in needed]
        for fp in files:
            df = pd.read_parquet(fp, columns=usecols)
            for _, r in df.iterrows():
                row = {}
                row["area"] = r.get("area", None)
                row["perim"] = r.get("perim", None)
                for nm in ("rp_r", "rp_counts", "rp_f_ring"):
                    val = r.get(nm, None)
                    row[nm] = None if val is None else np.asarray(val, dtype=np.float64)
                yield row


# -------------------------- Core aggregation --------------------------

def _align_and_stack(rows, metric, min_size=None, max_size=None):
    """
    Filters by size, aligns r-grids, and returns:
      - r (final r_master; longest grid across rows, extended by Δr)
      - delta_r (inferred)
      - counts_list (list of counts arrays, all padded to len(r))
      - fr_list (list of f_ring arrays, padded; NaN where undefined)
      - perims (weights; perim or 1.0 fallback)
      - rmax_idx (last bin index with positive counts for each cloud)
      - n_seen, n_used
    """
    r_master = None
    delta_r = None

    # Temporarily hold raw (un-padded) arrays so we can pad AFTER we know final r_master length
    raw_counts = []
    raw_fr = []
    raw_r = []
    perims = []
    rmax_idx = []
    n_seen = 0
    n_used = 0

    for row in rows:
        n_seen += 1
        size_val = row.get(metric, None)
        if size_val is None:
            continue
        if (min_size is not None and size_val < min_size) or (max_size is not None and size_val >= max_size):
            continue

        r = row.get("rp_r", None)
        cnt = row.get("rp_counts", None)
        fr = row.get("rp_f_ring", None)
        if r is None or cnt is None:
            continue

        r = np.asarray(r, dtype=np.float64)
        cnt = np.asarray(cnt, dtype=np.float64)
        if r.size == 0 or cnt.size == 0:
            continue

        if delta_r is None:
            delta_r = _infer_delta_r(r)
        if not (delta_r and np.isfinite(delta_r) and delta_r > 0):
            delta_r = _infer_delta_r(r)

        # compute f_ring if missing
        if fr is None:
            denom = 2.0 * math.pi * r * delta_r
            with np.errstate(divide="ignore", invalid="ignore"):
                fr = np.where(denom > 0, cnt / denom, np.nan)
        else:
            fr = np.asarray(fr, dtype=np.float64)

        # initialize or (virtually) extend r_master
        if r_master is None:
            r_master = r.copy()
        else:
            # sanity on shared prefix
            Lmin = min(r_master.size, r.size)
            if Lmin > 3 and not np.allclose(r_master[:Lmin], r[:Lmin], atol=1e-6, rtol=1e-6):
                raise ValueError("rp_r misalignment across rows. Ensure same Δr and zero-origin bins.")
            # if this row is longer, extend r_master (we'll pad previous rows later)
            if r.size > r_master.size:
                extra = r.size - r_master.size
                r_ext = r_master[-1] + delta_r * np.arange(1, extra + 1)
                r_master = np.concatenate([r_master, r_ext])

        raw_r.append(r)
        raw_counts.append(cnt)
        raw_fr.append(fr)
        perims.append(row.get("perim", 1.0) or 1.0)

        nz = np.where(cnt > 0)[0]
        rmax_idx.append(int(nz[-1]) if nz.size else -1)

        n_used += 1

    if r_master is None or n_used == 0:
        raise RuntimeError("No matching rows after filtering/alignment.")

    # Final padding of all arrays to the final r_master length
    L = r_master.size
    counts_list = []
    fr_list = []
    for cnt, fr, r in zip(raw_counts, raw_fr, raw_r):
        if r.size < L:
            pad = L - r.size
            cnt = np.pad(cnt, (0, pad))
            fr = np.pad(fr, (0, pad), constant_values=np.nan)
        counts_list.append(cnt)
        fr_list.append(fr)

    return {
        "r": r_master,
        "delta_r": float(delta_r),
        "counts_list": counts_list,
        "fr_list": fr_list,
        "perims": np.asarray(perims, dtype=np.float64),
        "rmax_idx": np.asarray(rmax_idx, dtype=np.int64),
        "n_seen": n_seen,
        "n_used": n_used,
    }


def _classic_perimeter_weighted(r, delta_r, counts_list):
    counts_sum = np.sum(np.vstack(counts_list), axis=0).astype(np.float64)
    # classic f_ring from summed counts
    denom = 2.0 * math.pi * r * delta_r
    with np.errstate(divide="ignore", invalid="ignore"):
        fring = np.where(denom > 0, counts_sum / denom, np.nan)
    return counts_sum, fring


def _riskset_stats(fr_list, rmax_idx, weights=None):
    """
    Risk-set mean/median at each radius: average only over clouds with rmax_i >= k.
    If weights provided, compute weighted mean (else unweighted mean). Median is unweighted.
    Returns mean, median, Mk (risk-set sizes).
    """
    A = np.vstack(fr_list)  # shape: (N, L)
    N, L = A.shape
    idxs = np.arange(L)[None, :]  # (1, L)
    # mask: cloud i contributes at bin k iff rmax_i >= k
    mask = (rmax_idx[:, None] >= idxs)  # (N, L)

    # count risk set per bin
    Mk = mask.sum(axis=0).astype(np.int64)

    # replace NaNs with 0 for computations; we'll mask by 'mask' anyway
    A_filled = np.nan_to_num(A, copy=False, nan=0.0)

    # unweighted mean over risk set
    sum_over = (A_filled * mask).sum(axis=0)
    mean_unw = np.where(Mk > 0, sum_over / np.maximum(Mk, 1), np.nan)

    # weighted mean over risk set (optional)
    if weights is not None:
        w = weights[:, None] * mask  # zero where not in risk set
        wsum = w.sum(axis=0)
        mean_w = np.where(wsum > 0, (A_filled * w).sum(axis=0) / wsum, np.nan)
    else:
        mean_w = None

    # median over risk set (unweighted): compute per-column with masking
    median_vals = np.full(L, np.nan, dtype=np.float64)
    for k in range(L):
        if Mk[k] == 0:
            continue
        col = A[:, k]
        col = col[mask[:, k]]
        col = col[~np.isnan(col)]
        if col.size:
            median_vals[k] = float(np.median(col))

    return mean_unw, median_vals, Mk, mean_w


# -------------------------- Plotting --------------------------

def _plot_overlays(r, curves: dict[str, np.ndarray], title, outdir: Path, base):
    outdir.mkdir(parents=True, exist_ok=True)

    for mode, fn in (("linear", plt.plot), ("semilogy", plt.semilogy), ("loglog", plt.loglog)):
        plt.figure()
        for label, y in curves.items():
            if mode == "loglog":
                m = (r > 0) & np.isfinite(y) & (y > 0)
                fn(r[m], y[m], label=label)
            elif mode == "semilogy":
                m = (r >= 0) & np.isfinite(y) & (y >= 0)
                fn(r[m], y[m], label=label)
            else:
                m = np.isfinite(y)
                fn(r[m], y[m], label=label)
        plt.xlabel("r (pixels)")
        plt.ylabel("f_ring(r)")
        plt.title(f"{title} — {mode}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{base}_{mode}.png", dpi=220)
        plt.close()


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Risk-set aggregates for boundary radial profiles.")
    ap.add_argument("--per-cloud-dir", type=Path, required=True,
                    help="Directory with per_cloud/*.parquet for one run_tag.")
    ap.add_argument("--metric", choices=["area", "perim"], default="area",
                    help="Optional size metric to bin on.")
    ap.add_argument("--min-size", type=float, default=None, help="Inclusive lower bound.")
    ap.add_argument("--max-size", type=float, default=None, help="Exclusive upper bound.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default=None, help="Filename suffix.")
    ap.add_argument("--save-txt", action="store_true", help="Dump r and curves to TXT.")
    args = ap.parse_args()

    rows = _load_rows(args.per_cloud_dir)
    aligned = _align_and_stack(
        rows,
        metric=args.metric,
        min_size=args.min_size,
        max_size=args.max_size
    )

    r = aligned["r"]
    delta_r = aligned["delta_r"]
    counts_list = aligned["counts_list"]
    fr_list = aligned["fr_list"]
    perims = aligned["perims"]
    rmax_idx = aligned["rmax_idx"]

    counts_sum, fring_classic = _classic_perimeter_weighted(r, delta_r, counts_list)
    mean_unw, median_unw, Mk, mean_w = _riskset_stats(fr_list, rmax_idx, weights=perims)

    base = args.tag or (f"riskset_{args.metric}"
                        + (f"_{int(args.min_size)}to{int(args.max_size)}" if args.min_size is not None and args.max_size is not None else "")
                        + f"_n{aligned['n_used']}")

    title = (f"{args.metric}"
             + (f" ∈ [{args.min_size:.0f}, {args.max_size:.0f})" if args.min_size is not None and args.max_size is not None else "")
             + f" | n={aligned['n_used']}")

    curves = {
        "classic_sum_counts (perim-wt)": fring_classic,
        "riskset_mean (unweighted)": mean_unw,
        "riskset_median (unweighted)": median_unw,
    }
    if mean_w is not None:
        curves["riskset_mean (perim-wt)"] = mean_w

    _plot_overlays(r, curves, title, args.outdir, base)

    # Save a tiny JSON summary + optional TXT
    summary = {
        "per_cloud_dir": str(args.per_cloud_dir),
        "metric": args.metric,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "n_seen": int(aligned["n_seen"]),
        "n_used": int(aligned["n_used"]),
        "delta_r": float(delta_r),
        "outputs": {k: str(args.outdir / f"{base}_{k}.png") for k in ("linear", "semilogy", "loglog")},
    }
    (args.outdir).mkdir(parents=True, exist_ok=True)
    with open(args.outdir / f"{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_txt:
        np.savetxt(args.outdir / f"{base}_r.txt", r)
        np.savetxt(args.outdir / f"{base}_classic_fring.txt", fring_classic)
        np.savetxt(args.outdir / f"{base}_riskset_mean.txt", mean_unw)
        np.savetxt(args.outdir / f"{base}_riskset_median.txt", median_unw)
        np.savetxt(args.outdir / f"{base}_riskset_Mk.txt", Mk.astype(np.int64))

    print(f"[OK] Risk-set plots written to {args.outdir} ; n_used={aligned['n_used']} ; Δr≈{delta_r:.6g}")


if __name__ == "__main__":
    main()
