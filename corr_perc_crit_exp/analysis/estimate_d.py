import sys
import os
from pathlib import Path
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../utils')))
from slice_analysis_utils import *

def main():
    parser = argparse.ArgumentParser(description="Estimate D for a given gamma, p, and slice_id.")
    parser.add_argument("gamma_dir", type=str, help="Gamma directory name, e.g., g_0p1975")
    parser.add_argument("p_dir", type=str, help="p directory name, e.g., p_0p460")
    parser.add_argument("slice_id", type=int, help="Slice ID (-1 for full cloud, 0-29 for slices)")
    parser.add_argument("--base_dir", type=str, default="/scratch/mbiju2/storm/cp_crit_exp_aggregated_20250711", help="Base directory for data")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    analysis_dir = base_dir / "analysis" / args.gamma_dir / args.p_dir

    slice_csv_path = get_slice_log_csv_path(analysis_dir, args.slice_id)
    summary_csv = analysis_dir / "summary_filtered_slice_d_vals.csv"

    print(f"[INFO] Estimating D for {slice_csv_path}")

    try:
        d_est = estimate_d_from_csv(slice_csv_path)
        append_d_to_summary(summary_csv, args.slice_id, d_est)
        print(f"[INFO] Appended D={d_est:.6f} for slice_id={args.slice_id} to {summary_csv}")
    except Exception as e:
        print(f"[ERROR] Failed to estimate D for {slice_csv_path}: {e}")

if __name__ == "__main__":
    main()
