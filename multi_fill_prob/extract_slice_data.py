import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from slice_analysis_utils import *
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract all clouds of a given slice ID across a fill_prob directory.")
    parser.add_argument("fill_prob_dir", type=str, help="Path to fill_prob_X or a single lattice_run dir")
    parser.add_argument("slice_id", type=int, help="Slice ID to extract (0–29, or -1 for full cloud)")
    parser.add_argument("output_csv", type=str, help="Output CSV path")

    parser.add_argument("--area_key", type=str, default="mirrored_area", help="Column name for area")
    parser.add_argument("--perim_key", type=str, default="mirrored_perimeter", help="Column name for perimeter")
    parser.add_argument("--slice_id_key", type=str, default="slice_id", help="Column name for slice ID")

    args = parser.parse_args()
    fill_prob_dir = Path(args.fill_prob_dir)
    output_csv = Path(args.output_csv)

    print(f"[START] Extracting slice_id={args.slice_id} from {fill_prob_dir}")
    raw_data = stream_slice_data_for_id(
        fill_prob_dir=fill_prob_dir,
        slice_id=args.slice_id,
        area_key=args.area_key,
        perim_key=args.perim_key,
        slice_id_key=args.slice_id_key
    )

    log_data = convert_to_log_values(raw_data, verbose=True)
    write_slice_data_to_csv(log_data, output_csv)
    print(f"[DONE] Completed writing to {output_csv}")

if __name__ == "__main__":
    main()
