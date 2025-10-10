#!/usr/bin/env python3
"""
plot_sizebinned_agg_rprof.py

Filter clouds by a size bin (area or perimeter), aggregate their boundary radial
profiles, and plot counts(r), pdf(r), and f_ring(r) in linear, semilogy, and log–log.

Examples
--------
# Example matching your spec: bin-start=1000, bin-size=300 -> [1000, 1300)
python -m clouds.r_dist_exps.plot_sizebinned_agg_rprof \
  --per-cloud-dir scratch/expC/per_cloud/expC_internal_W10000_H10000_p0.407400_seed987 \
  --bin-start 5000 --bin-size 300 --metric area \
  --outdir scratch/expC/sizebins

# Using explicit min/max:
python plot_sizebinned_agg_rprof.py \
  --per-cloud-dir scratch/expA/per_cloud/expA_internal_W4000_H2666_p0.407400_seed987 \
  --min-size 5000 --max-size 8000 --metric perim
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
except Exception as e:
    print("[WARN] PyArrow not available; falling back to pandas.read_parquet (slower).", file=sys.stderr)
    pa = None
    pq = None
    import pandas as pd


def _infer_delta_r(r: np.ndarray) -> float:
    if r.size < 3:
        return float("nan")
    diffs = np.diff(r)
    # robust-ish: median of positive diffs
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else float("nan")


def _safe_scalar(x):
    """PyArrow scalar -> Python value (None if null). Works w/ pandas too."""
    if pa is not None and isinstance(x, pa.Scalar):
        return x.as_py()
    return x


def _extend_sum(acc: np.ndarray | None, vec: np.ndarray) -> np.ndarray:
    """Elementwise sum with zero-extend to the max length."""
    if acc is None:
        return vec.astype(np.float64, copy=True)
    if acc.size >= vec.size:
        acc[:vec.size] += vec
        return acc
    # need to grow acc
    out = np.zeros(vec.size, dtype=np.float64)
    out[:acc.size] = acc
    out += vec
    return out


def _load_rows_from_parquet(per_cloud_dir: Path,
                            needed_cols=("area", "perim", "rp_r", "rp_counts", "rp_pdf", "rp_f_ring")):
    """Generator yielding dicts for each row across all *.parquet parts."""
    files = sorted(per_cloud_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {per_cloud_dir}")

    if pq is not None:
        cols = list(needed_cols)
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(columns=cols):
                tbl = pa.Table.from_batches([batch])
                cols_map = {name: tbl[name] if name in tbl.schema.names else None for name in needed_cols}
                n = tbl.num_rows
                for i in range(n):
                    row = {}
                    for name in ("area", "perim"):
                        col = cols_map.get(name)
                        row[name] = _safe_scalar(col[i]) if col is not None else None
                    for name in ("rp_r", "rp_counts", "rp_pdf", "rp_f_ring"):
                        col = cols_map.get(name)
                        if col is None:
                            row[name] = None
                        else:
                            arr = col[i].as_py()  # list or None
                            row[name] = None if arr is None else np.asarray(arr, dtype=np.float64)
                    yield row
    else:
        # pandas fallback (loads whole file)
        usecols = [c for c in needed_cols]  # let pandas ignore missing
        for fp in files:
            df = pd.read_parquet(fp, columns=usecols)
            for _, r in df.iterrows():
                row = {}
                row["area"] = r.get("area", None)
                row["perim"] = r.get("perim", None)
                for name in ("rp_r", "rp_counts", "rp_pdf", "rp_f_ring"):
                    val = r.get(name, None)
                    row[name] = None if val is None else np.asarray(val, dtype=np.float64)
                yield row


def aggregate_for_size_bin(per_cloud_dir: Path,
                           metric: str,
                           min_size: float,
                           max_size: float):
    """
    Aggregate r-profiles only for clouds whose `metric` (area or perim)
    falls within [min_size, max_size).

    Returns:
        r (np.ndarray): bin centers
        counts (np.ndarray): sum of rp_counts across selected clouds
        pdf (np.ndarray): normalized density over r (sum pdf*Δr = 1)
        fring (np.ndarray): counts / (2π r Δr)
        stats (dict): diagnostics
    """
    assert metric in ("area", "perim"), "metric must be 'area' or 'perim'"

    r_master = None
    delta_r = None
    counts_sum = None
    n_rows_seen = 0
    n_rows_used = 0
    size_vals = []

    for row in _load_rows_from_parquet(per_cloud_dir):
        n_rows_seen += 1
        size_val = row[metric]
        if size_val is None:
            continue
        if not (min_size <= size_val < max_size):
            continue

        r = row["rp_r"]
        cnt = row["rp_counts"]
        if r is None or cnt is None:
            continue

        r = np.asarray(r, dtype=np.float64)
        cnt = np.asarray(cnt, dtype=np.float64)

        if delta_r is None:
            delta_r = _infer_delta_r(r)

        # Initialize or grow master arrays
        if r_master is None:
            r_master = r.copy()
            counts_sum = cnt.copy()
        else:
            # grow to max length across all rows
            if r.size > r_master.size:
                extra_bins = r.size - r_master.size
                r_ext = r_master[-1] + delta_r * np.arange(1, extra_bins + 1)
                r_master = np.concatenate([r_master, r_ext])
                counts_sum = np.pad(counts_sum, (0, extra_bins))
            elif r.size < r_master.size:
                cnt = np.pad(cnt, (0, r_master.size - r.size))

            counts_sum += cnt

        size_vals.append(float(size_val))
        n_rows_used += 1

    if counts_sum is None or r_master is None or n_rows_used == 0:
        raise RuntimeError("No matching rows found for the specified size bin.")

    if not (np.isfinite(delta_r) and delta_r > 0):
        delta_r = _infer_delta_r(r_master)

    total_counts = float(np.sum(counts_sum))
    pdf = counts_sum / (total_counts * delta_r)

    # f_ring = counts / (2π r Δr); avoid r=0
    fring = np.full_like(counts_sum, np.nan, dtype=np.float64)
    denom = 2.0 * math.pi * r_master * delta_r
    mask = (denom > 0) & (counts_sum > 0)
    fring[mask] = counts_sum[mask] / denom[mask]

    stats = {
        "n_rows_seen": n_rows_seen,
        "n_rows_used": n_rows_used,
        "metric": metric,
        "min_size": float(min_size),
        "max_size": float(max_size),
        "delta_r": float(delta_r),
        "size_mean": float(np.mean(size_vals)) if size_vals else float("nan"),
        "size_median": float(np.median(size_vals)) if size_vals else float("nan"),
    }
    return r_master, counts_sum, pdf, fring, stats


def _three_style_plots(r, y, title_root: str, ylabel: str, outdir: Path, fname_root: str):
    outdir.mkdir(parents=True, exist_ok=True)

    # Linear
    plt.figure()
    plt.plot(r, y)
    plt.xlabel("r (pixels)")
    plt.ylabel(ylabel)
    plt.title(f"{title_root} — linear")
    plt.tight_layout()
    plt.savefig(outdir / f"{fname_root}_linear.png", dpi=200)
    plt.close()

    # Semilogy
    plt.figure()
    plt.semilogy(r, y)
    plt.xlabel("r (pixels)")
    plt.ylabel(ylabel)
    plt.title(f"{title_root} — semilogy")
    plt.tight_layout()
    plt.savefig(outdir / f"{fname_root}_semilogy.png", dpi=200)
    plt.close()

    # Log–log
    plt.figure()
    # filter positive for log-log
    m = (r > 0) & (y > 0)
    plt.loglog(r[m], y[m])
    plt.xlabel("r (pixels)")
    plt.ylabel(ylabel)
    plt.title(f"{title_root} — log–log")
    plt.tight_layout()
    plt.savefig(outdir / f"{fname_root}_loglog.png", dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Aggregate and plot r-profiles for clouds within a size bin.")
    p.add_argument("--per-cloud-dir", type=Path, required=True,
                   help="Directory containing per_cloud parquet parts (…/per_cloud/<run_tag>/).")
    p.add_argument("--metric", choices=["area", "perim"], default="area",
                   help="Which size metric to bin on (default: area).")

    # Option A: bin-start + bin-size  => [bin-start, bin-start+bin-size)
    p.add_argument("--bin-start", type=float, default=None,
                   help="Lower bound of bin (inclusive).")
    p.add_argument("--bin-size", type=float, default=None,
                   help="Bin width (exclusive upper bound at bin-start+bin-size).")

    # Option B: explicit min/max  => [min-size, max-size)
    p.add_argument("--min-size", type=float, default=None, help="Inclusive lower size bound.")
    p.add_argument("--max-size", type=float, default=None, help="Exclusive upper size bound.")

    p.add_argument("--outdir", type=Path, required=True, help="Folder to write plots and a small JSON summary.")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional suffix for filenames; otherwise derived from metric and bin.")
    args = p.parse_args()

    # Resolve bin bounds
    if args.bin_start is not None and args.bin_size is not None:
        min_size = args.bin_start
        max_size = args.bin_start + args.bin_size
    elif args.min_size is not None and args.max_size is not None:
        min_size, max_size = args.min_size, args.max_size
    else:
        p.error("Specify either --bin-start AND --bin-size, or --min-size AND --max-size.")

    r, counts, pdf, fring, stats = aggregate_for_size_bin(
        args.per_cloud_dir, args.metric, min_size, max_size
    )

    # Filenames
    if args.tag:
        base = args.tag
    else:
        base = f"{args.metric}_{int(min_size)}to{int(max_size)}"

    # Plots
    title_root = f"{args.metric} ∈ [{min_size:.0f}, {max_size:.0f}) | n={stats['n_rows_used']}"
    _three_style_plots(r, counts, title_root, "counts(r)", args.outdir, f"{base}_counts")
    _three_style_plots(r, pdf,    title_root, "pdf(r)",    args.outdir, f"{base}_pdf")
    _three_style_plots(r, fring,  title_root, "f_ring(r)", args.outdir, f"{base}_fring")

    # Summary JSON (handy for book-keeping)
    summary = {
        "per_cloud_dir": str(args.per_cloud_dir),
        "metric": args.metric,
        "min_size": float(min_size),
        "max_size": float(max_size),
        "n_rows_used": int(stats["n_rows_used"]),
        "n_rows_seen": int(stats["n_rows_seen"]),
        "delta_r": stats["delta_r"],
        "size_mean": stats["size_mean"],
        "size_median": stats["size_median"],
        "outputs": {
            "counts_linear": str(args.outdir / f"{base}_counts_linear.png"),
            "counts_semilogy": str(args.outdir / f"{base}_counts_semilogy.png"),
            "counts_loglog": str(args.outdir / f"{base}_counts_loglog.png"),
            "pdf_linear": str(args.outdir / f"{base}_pdf_linear.png"),
            "pdf_semilogy": str(args.outdir / f"{base}_pdf_semilogy.png"),
            "pdf_loglog": str(args.outdir / f"{base}_pdf_loglog.png"),
            "fring_linear": str(args.outdir / f"{base}_fring_linear.png"),
            "fring_semilogy": str(args.outdir / f"{base}_fring_semilogy.png"),
            "fring_loglog": str(args.outdir / f"{base}_fring_loglog.png"),
        }
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / f"{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Used {stats['n_rows_used']} clouds out of {stats['n_rows_seen']} rows.")
    print(f"[OK] Δr ~ {stats['delta_r']:.6g}. Plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()
