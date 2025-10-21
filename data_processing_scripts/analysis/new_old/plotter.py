#!/usr/bin/env python3
"""
plotter.py — unified plotter for Autocorr and BDD analyses
==========================================================

Supports two plotting modes:
  • overlay   → all curves on one figure (default)
  • separate  → one figure per input CSV

Each curve file can come from Autocorr (r,y pairs) or BDD (u,y pairs).

----------------------------------------------------------------------
📄 SAMPLE RUN COMMANDS
----------------------------------------------------------------------
# Overlay Autocorr components (numerator, denom, C(r)) on one plot
python analysis/plotter.py autocorr \
    --csv scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__autocorr_num.csv \
          scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__autocorr_den.csv \
          scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__autocorr_Cr.csv \
    --field y \
    --logx --logy \
    --legend stem \
    --mode overlay \
    --title "Autocorr Components (num, denom, C(r))" \
    --out scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__overlay_autocorr_components.png

# Create SEPARATE plots (one per CSV)
python analysis/plotter.py autocorr \
    --csv scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__autocorr_num.csv \
          scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__autocorr_den.csv \
          scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__autocorr_Cr.csv \
    --field y \
    --logx --logy \
    --legend stem \
    --mode separate \
    --title "Autocorr Component" \
    --out-template scratch/all_clouds_data/analysis/autocorr/SITEPERC_all__autocorr_{stem}.png
"""

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------

def _load_xy(path: str, field: str):
    """
    Load (x, y) data from a CSV file.
    Supports:
        • r,y or u,y columns (for split CSVs)
        • curves.csv with r_index + agg_* columns (for autocorr aggregates)
    """
    df = pd.read_csv(path)
    # split CSV style (r,y) or (u,y)
    if {"r", "y"}.issubset(df.columns):
        return df["r"].to_numpy(float), df["y"].to_numpy(float)
    if {"u", "y"}.issubset(df.columns):
        return df["u"].to_numpy(float), df["y"].to_numpy(float)
    # curves.csv style (r_index + agg_*)
    if "r_index" in df.columns and field in df.columns:
        return df["r_index"].to_numpy(float), df[field].to_numpy(float)
    raise ValueError(f"{path}: missing required columns for {field}")


def _plot_single_curve(x, y, out_path, title, xlabel, ylabel, logx, logy):
    """
    Create and save a single (x, y) plot.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(9.0, 7.0))
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    plt.plot(x, y, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_path}")


def _plot_overlay(curves, out_path, title, xlabel, ylabel, logx, logy, legend_mode):
    """
    Overlay multiple (x, y) curves on one figure.
    curves: list of (path, x, y)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(9.0, 7.0))
    for path, x, y in curves:
        label = os.path.splitext(os.path.basename(path))[0] if legend_mode == "stem" else os.path.basename(path)
        plt.plot(x, y, lw=2.0, label=label)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved overlay {out_path} ({len(curves)} curves)")

# ----------------------------------------------------------------------
# AUTOCORR plotting logic
# ----------------------------------------------------------------------
def plot_autocorr(args):
    """
    Plot autocorr CSVs (split or aggregate).
    Supports both overlay and separate modes.
    """
    curves = []
    for path in args.csv:
        try:
            x, y = _load_xy(path, args.field)
        except Exception as e:
            print(f"[WARN] {e}")
            continue
        if args.logx:
            mask = np.isfinite(x) & (x > 0)
            x, y = x[mask], y[mask]
        if args.logy:
            mask = np.isfinite(y) & (y > 0)
            x, y = x[mask], y[mask]
        curves.append((path, x, y))

    ylabel = {
        "agg_ratio": "aggregated num / den",
        "agg_num": "aggregated numerator",
        "agg_den": "aggregated denominator",
        "y": "value"
    }[args.field]

    if args.mode == "overlay":
        _plot_overlay(curves, args.out, args.title, "r (bin index)", ylabel,
                      args.logx, args.logy, args.legend)
    else:  # separate
        if not args.out_template:
            raise ValueError("--out-template required when --mode separate")
        for path, x, y in curves:
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = args.out_template.format(stem=stem, name=os.path.basename(path))
            _plot_single_curve(x, y, out_path, args.title, "r (bin index)", ylabel,
                               args.logx, args.logy)

# ----------------------------------------------------------------------
# BDD plotting logic
# ----------------------------------------------------------------------
def plot_bdd(args):
    """
    Plot BDD CSVs (u,y,coverage,width).
    Supports both overlay and separate modes.
    """
    curves = []
    for path in args.csv:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] {e}")
            continue
        if not {"u", "y"}.issubset(df.columns):
            print(f"[WARN] Skipping {path}: missing columns 'u' and 'y'.")
            continue
        x = df["u"].to_numpy(float)
        y = df["y"].to_numpy(float)
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
        curves.append((path, x, y))

    ylabel = "Density (per unit-u)"

    if args.mode == "overlay":
        _plot_overlay(curves, args.out, args.title, "u = r / Rg_area", ylabel,
                      args.logx, args.logy, args.legend)
    else:
        if not args.out_template:
            raise ValueError("--out-template required when --mode separate")
        for path, x, y in curves:
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = args.out_template.format(stem=stem, name=os.path.basename(path))
            _plot_single_curve(x, y, out_path, args.title, "u = r / Rg_area", ylabel,
                               args.logx, args.logy)

# ----------------------------------------------------------------------
# CLI setup
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified plotter for autocorr and BDD CSVs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------------- AUTOCORR ----------------
    ap = sub.add_parser("autocorr", help="Plot autocorr CSVs (r,y or curves.csv).")
    ap.add_argument("--csv", nargs="+", required=True, help="List of CSV paths.")
    ap.add_argument("--field", default="agg_ratio",
                    choices=["agg_ratio","agg_num","agg_den","y"],
                    help="Column to plot (default: agg_ratio).")
    ap.add_argument("--logx", action="store_true", help="Log-scale x.")
    ap.add_argument("--logy", action="store_true", help="Log-scale y.")
    ap.add_argument("--legend", choices=["stem","name"], default="stem",
                    help="Legend labels from filename stem or full name.")
    ap.add_argument("--title", default="Autocorr plot", help="Plot title.")
    ap.add_argument("--mode", choices=["overlay","separate"], default="overlay",
                    help="overlay = all curves together; separate = one PNG per file.")
    ap.add_argument("--out", help="Output PNG for overlay mode.")
    ap.add_argument("--out-template",
                    help="Output template for separate mode (use {stem} or {name}).")
    ap.set_defaults(func=plot_autocorr)

    # ---------------- BDD ----------------
    bp = sub.add_parser("bdd", help="Plot BDD CSVs (u,y,coverage,width).")
    bp.add_argument("--csv", nargs="+", required=True, help="List of CSV paths.")
    bp.add_argument("--logx", action="store_true", help="Log-scale x.")
    bp.add_argument("--logy", action="store_true", help="Log-scale y.")
    bp.add_argument("--min-coverage", type=int, default=0, help="Hide bins with coverage < N.")
    bp.add_argument("--legend", choices=["stem","name"], default="stem",
                    help="Legend labels from filename stem or full name.")
    bp.add_argument("--title", default="BDD plot", help="Plot title.")
    bp.add_argument("--mode", choices=["overlay","separate"], default="overlay",
                    help="overlay = all curves together; separate = one PNG per file.")
    bp.add_argument("--out", help="Output PNG for overlay mode.")
    bp.add_argument("--out-template",
                    help="Output template for separate mode (use {stem} or {name}).")
    bp.set_defaults(func=plot_bdd)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
