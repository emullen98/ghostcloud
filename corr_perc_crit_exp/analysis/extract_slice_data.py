import sys
import os
from pathlib import Path
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../utils')))
from slice_analysis_utils import *

def main():
    parser = argparse.ArgumentParser(description="Extract and filter clouds for a given gamma, p, and slice_id.")
    parser.add_argument("gamma_dir", type=str, help="Gamma directory name, e.g., g_0p1975")
    parser.add_argument("p_dir", type=str, help="p directory name, e.g., p_0p460")
    parser.add_argument("slice_id", type=int, help="Slice ID (-1 for full cloud, 0-29 for slices)")
    parser.add_argument("output_csv", type=str, help="Output CSV path")

    parser.add_argument("--base_dir", type=str, default="/scratch/mbiju2/storm/cp_crit_exp_aggregated_20250711", help="Base directory for data")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    input_dir = base_dir / args.gamma_dir / args.p_dir

    print(f"[INFO] Processing gamma={args.gamma_dir}, p={args.p_dir}, slice_id={args.slice_id}")

    # Use your stream_slice_data_for_id, yields (area, perimeter) tuples
    raw_iter = stream_slice_data_for_id(
        fill_prob_dir=input_dir,
        slice_id=args.slice_id,
        area_key="mirrored_area",
        perim_key="mirrored_perimeter",
        slice_id_key="slice_id",
    )

    # Apply area filter
    filtered_iter = filter_min_area(raw_iter, min_area=3000)

    # Log-transform
    log_iter = convert_to_log_values(filtered_iter, verbose=True)

    # Write
    output_path = Path(args.output_csv)
    write_slice_data_to_csv(log_iter, output_path)

    print(f"[DONE] Written to {args.output_csv}")

if __name__ == "__main__":
    main()
