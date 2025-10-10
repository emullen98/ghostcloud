#!/usr/bin/env python3
"""
rdist_plot_lattice_agg.py

Quick plots for lattice-level aggregates already saved by siteperc_rprof_from_model.py:

  - <agg_dir>/<run_tag>_rprof_counts.txt
  - <agg_dir>/<run_tag>_rprof_pdf.txt
  - <agg_dir>/<run_tag>_rprof_r.txt
  - <agg_dir>/<run_tag>_rprof_fring.txt (optional)

Generates linear, semilog, and loglog views for fast inspection.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _load_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser(description="Plot lattice-level r-profile aggregates.")
    ap.add_argument("--agg-dir", required=True, type=Path, help="Aggregate dir created by the generator.")
    ap.add_argument("--run-tag", required=True, type=str, help="Run tag used by the generator.")
    ap.add_argument("--outdir", required=True, type=Path, help="Directory to save plots.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    r_path = args.agg_dir / f"{args.run_tag}_rprof_r.txt"
    counts_path = args.agg_dir / f"{args.run_tag}_rprof_counts.txt"
    pdf_path = args.agg_dir / f"{args.run_tag}_rprof_pdf.txt"
    fring_path = args.agg_dir / f"{args.run_tag}_rprof_fring.txt"

    r = _load_txt(r_path)
    counts = _load_txt(counts_path)
    pdf = _load_txt(pdf_path)
    fring = _load_txt(fring_path) if fring_path.exists() else None

    # Linear plots
    plt.figure()
    plt.plot(r, counts, lw=1.5, label="counts")
    if fring is not None:
        plt.plot(r, fring, lw=1.2, label="f_ring (counts / 2πrΔr)")
    plt.xlabel("r (px)")
    plt.ylabel("Value")
    plt.title("Aggregate counts (linear)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "agg_counts_linear.png", dpi=160)

    plt.figure()
    plt.plot(r, pdf, lw=1.5)
    plt.xlabel("r (px)")
    plt.ylabel("pdf (density over r)")
    plt.title("Aggregate pdf (linear)")
    plt.tight_layout()
    plt.savefig(args.outdir / "agg_pdf_linear.png", dpi=160)

    # Semilog y
    plt.figure()
    plt.semilogy(r, counts, lw=1.5, label="counts")
    if fring is not None:
        plt.semilogy(r, fring, lw=1.2, label="f_ring")
    plt.xlabel("r (px)")
    plt.ylabel("Value")
    plt.title("Aggregate counts (semilog y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "agg_counts_semilogy.png", dpi=160)

    plt.figure()
    plt.semilogy(r, pdf, lw=1.5)
    plt.xlabel("r (px)")
    plt.ylabel("pdf")
    plt.title("Aggregate pdf (semilog y)")
    plt.tight_layout()
    plt.savefig(args.outdir / "agg_pdf_semilogy.png", dpi=160)

    # Log-log
    # (Avoid r=0; your r-centers start at 0.5*Δr so OK.)
    plt.figure()
    plt.loglog(r, counts, lw=1.5, label="counts")
    if fring is not None:
        plt.loglog(r, fring, lw=1.2, label="f_ring")
    plt.xlabel("r (px)")
    plt.ylabel("Value")
    plt.title("Aggregate counts (log-log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "agg_counts_loglog.png", dpi=160)

    plt.figure()
    plt.loglog(r, pdf, lw=1.5)
    plt.xlabel("r (px)")
    plt.ylabel("pdf")
    plt.title("Aggregate pdf (log-log)")
    plt.tight_layout()
    plt.savefig(args.outdir / "agg_pdf_loglog.png", dpi=160)

    print("[OK] Wrote plots to", str(args.outdir))

if __name__ == "__main__":
    main()
