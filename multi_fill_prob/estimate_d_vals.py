import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from slice_analysis_utils import *
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Estimate D for a single slice and append to summary.")
    parser.add_argument("analysis_dir", type=str, help="Path to fill_prob_XXXX analysis directory")
    parser.add_argument("slice_id", type=int, help="Slice ID to analyze (0–29 or -1 for full cloud)")

    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir)
    summary_csv = analysis_dir / "summary_filtered_slice_d_vals.csv"
    slice_path = get_slice_log_csv_path(analysis_dir, args.slice_id)

    print(f"[INFO] Estimating D for slice_id={args.slice_id} from {slice_path}")
    d_est = estimate_d_from_csv(slice_path)
    append_d_to_summary(summary_csv, args.slice_id, d_est)
    print(f"[INFO] Appended D={d_est:.6f} for slice_id={args.slice_id} to {summary_csv}")

if __name__ == "__main__":
    main()
