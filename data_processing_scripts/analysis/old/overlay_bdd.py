#!/usr/bin/env python3
"""
overlay_bdd.py — overlay Boundary Distance Distribution (BDD) curves
=====================================================================

Reads one or more BDD CSVs (columns: u,y,coverage,width) and overlays
density vs u curves on a single plot.

----------------------------------------------------------------------
🧩 INPUT FILES
  - Usually saved by your bdd.py script automatically:
      * {SUFFIX}__bdd_len.csv
      * {SUFFIX}__bdd_eq.csv
  - Each file must contain columns: u,y,coverage,width

----------------------------------------------------------------------
💾 OUTPUT
  - Default: "overlay_bdd.png" in the current working directory.
  - You can override with:
        --out /path/to/my_overlay.png
  - The script will create the output directory if needed.

----------------------------------------------------------------------
🎯 EXAMPLES
  # Overlay all length-weighted BDD curves (log–log)
  python overlay_bdd.py scratch/.../bd_distribution/*__bdd_len.csv \
      --logx --logy --out scratch/.../bd_distribution/overlay_len.png

  # Overlay equal-cloud curves, keeping only bins with ≥10 contributing clouds
  python overlay_bdd.py scratch/.../bd_distribution/*__bdd_eq.csv \
      --logx --logy --min-coverage 10 \
      --out scratch/.../bd_distribution/overlay_eq.png

  # Mix specific files
  python overlay_bdd.py runA__bdd_len.csv runB__bdd_len.csv runC__bdd_len.csv \
      --logx --logy --out combined_overlay.png

----------------------------------------------------------------------
🔧 OPTIONS
  --logx / --logy          : Log scaling for x or y axis.
  --min-coverage N         : Hide bins with coverage < N (default 0 = keep all).
  --legend stem|name       : Legend labels from filename stem or full name.
  --out FILE               : Output PNG path.
  --title TITLE            : Custom plot title.

----------------------------------------------------------------------
"""

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Overlay BDD curves from CSVs (u,y,coverage,width).")
    ap.add_argument("csvs", nargs="+", help="Paths to BDD CSV files.")
    ap.add_argument("--out", default="overlay_bdd.png", help="Output PNG path.")
    ap.add_argument("--title", default="BDD overlays", help="Plot title.")
    ap.add_argument("--logx", action="store_true", help="Log-scale x (u).")
    ap.add_argument("--logy", action="store_true", help="Log-scale y (density).")
    ap.add_argument("--min-coverage", type=int, default=0, help="Hide bins with coverage < this.")
    ap.add_argument("--legend", choices=["stem","name"], default="stem",
                    help="Legend labels from filename stem (default) or full name.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(9.0,7.0))
    plotted = 0
    for path in args.csvs:
        df = pd.read_csv(path)
        if not {"u","y"}.issubset(df.columns):
            print(f"[WARN] Skipping {path}: missing required columns 'u' and 'y'.")
            continue
        x = df["u"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        if "coverage" in df.columns and args.min_coverage > 0:
            cov = df["coverage"].to_numpy()
            keep = cov >= args.min_coverage
            x, y = x[keep], y[keep]
        if args.logx:
            mask = np.isfinite(x) & (x > 0)
            x, y = x[mask], y[mask]
        if args.logy:
            mask = np.isfinite(y) & (y > 0)
            x, y = x[mask], y[mask]
        label = os.path.splitext(os.path.basename(path))[0] if args.legend=="stem" else os.path.basename(path)
        plt.plot(x, y, label=label, lw=2.0)
        plotted += 1

    plt.xlabel("u = r / Rg_area")
    plt.ylabel("Density (per unit-u)")
    plt.title(args.title)
    if args.logx: plt.xscale("log")
    if args.logy: plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[OK] Saved {args.out} ({plotted} curves)")

if __name__ == "__main__":
    main()
