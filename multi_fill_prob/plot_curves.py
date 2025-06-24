import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from slice_analysis_utils import *
from pathlib import Path
import pandas as pd

def main():
    base_dir = Path("/scratch/mbiju2/storm/multi_fill_prob_20250601_124857/analysis")
    fill_probs = ["0_4066", "0_4069", "0_4072", "0_4074", "0_4076"]

    summary_paths = []
    labels = []

    for prob in fill_probs:
        analysis_dir = base_dir / f"fill_prob_{prob}"
        summary_csv = analysis_dir / "summary_filtered_slice_d_vals.csv"

        if not summary_csv.exists():
            print(f"[WARN] Missing summary file: {summary_csv}")
            continue

        # Save processed delta D CSV
        try:
            df = load_summary_with_deltas(summary_csv)
            df.to_csv(analysis_dir / f"filtered_delta_d_processed_{prob}.csv", index=False)
        except Exception as e:
            print(f"[WARN] Skipping {summary_csv}: {e}")
            continue

        # Plot single curve
        single_out = base_dir / f"filtered_delta_D_curve_fill_prob_{prob}.png"
        plot_single_fill_prob(summary_csv, single_out)

        summary_paths.append(summary_csv)
        labels.append(prob)

    # Plot all fill probs on one figure
    combined_out = base_dir / "filtered_delta_D_curves_combined.png"
    plot_fill_prob_curves(summary_paths, labels, combined_out)

if __name__ == "__main__":
    main()