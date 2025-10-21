#!/usr/bin/env python3
"""
overlay_autocorr.py — overlay aggregated autocorrelation curves
===============================================================

Reads one or more autocorrelation CSVs and overlays aggregated C(r),
numerator, or denominator vs r index.

----------------------------------------------------------------------
🧩 INPUT FILES
  You can pass either of the following:
   • Split tiny CSVs (two columns: r,y)
       {SUFFIX}__autocorr_Cr.csv
       {SUFFIX}__autocorr_num.csv
       {SUFFIX}__autocorr_den.csv
   • Or the full curves.csv (columns: r_index, agg_num, agg_den, agg_ratio)

----------------------------------------------------------------------
💾 OUTPUT
  - Default: "overlay_autocorr.png" in the current working directory.
  - Override with:
        --out /path/to/my_overlay.png
  - Output directory will be created automatically.

----------------------------------------------------------------------
🎯 EXAMPLES
  # Overlay aggregated C(r) curves from split CSVs
  python overlay_autocorr.py scratch/.../autocorr/*__autocorr_Cr.csv \
      --logx --logy --out scratch/.../autocorr/overlay_Cr.png

  # Overlay curves directly from curves.csv, plotting agg_ratio
  python overlay_autocorr.py scratch/.../autocorr/*__curves.csv \
      --field agg_ratio --logy --out overlay_ratio.png

  # Overlay numerator curves
  python overlay_autocorr.py scratch/.../autocorr/*__curves.csv \
      --field agg_num --logy --out overlay_num.png

----------------------------------------------------------------------
🔧 OPTIONS
  --field FIELD       : Which column to plot (agg_ratio|agg_num|agg_den|y). Default = agg_ratio.
  --logx / --logy     : Log scaling for axes.
  --legend stem|name  : Legend labels from filename stem or full name.
  --out FILE          : Output PNG path.
  --title TITLE       : Custom plot title.

----------------------------------------------------------------------
"""

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_xy(path: str, field: str):
    """Load (x,y) pairs from either split CSVs (r,y) or curves.csv."""
    df = pd.read_csv(path)
    if {"r","y"}.issubset(df.columns):
        return df["r"].to_numpy(float), df["y"].to_numpy(float)
    if "r_index" in df.columns and field in df.columns:
        return df["r_index"].to_numpy(float), df[field].to_numpy(float)
    raise ValueError(f"{path}: missing required columns for {field}")

def main():
    ap = argparse.ArgumentParser(description="Overlay autocorr aggregates from CSVs.")
    ap.add_argument("csvs", nargs="+", help="Paths to CSVs (split r,y or curves.csv).")
    ap.add_argument("--out", default="overlay_autocorr.png", help="Output PNG path.")
    ap.add_argument("--title", default="Autocorr overlays", help="Plot title.")
    ap.add_argument("--field", default="agg_ratio",
                    choices=["agg_ratio","agg_num","agg_den","y"],
                    help="Which column to plot (default agg_ratio).")
    ap.add_argument("--logx", action="store_true", help="Log-scale x.")
    ap.add_argument("--logy", action="store_true", help="Log-scale y.")
    ap.add_argument("--legend", choices=["stem","name"], default="stem",
                    help="Legend labels from filename stem (default) or full name.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(9.0,7.0))
    plotted = 0
    for path in args.csvs:
        try:
            x, y = load_xy(path, args.field)
        except Exception as e:
            print(f"[WARN] {e}")
            continue
        if args.logx:
            mask = np.isfinite(x) & (x > 0)
            x, y = x[mask], y[mask]
        if args.logy:
            mask = np.isfinite(y) & (y > 0)
            x, y = x[mask], y[mask]
        label = os.path.splitext(os.path.basename(path))[0] if args.legend=="stem" else os.path.basename(path)
        plt.plot(x, y, label=label, lw=2.0)
        plotted += 1

    plt.xlabel("r index (bin #)")
    ylabel = {"agg_ratio":"aggregated num / den",
              "agg_num":"aggregated numerator",
              "agg_den":"aggregated denominator",
              "y":"value"}[args.field]
    plt.ylabel(ylabel)
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
