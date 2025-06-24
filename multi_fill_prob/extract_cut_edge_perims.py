import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from slice_analysis_utils import *
from pathlib import Path
import argparse
import filelock

def main():
    parser = argparse.ArgumentParser(description="Extract cut-edge vs half-perimeter data for a given slice ID across a fill_prob directory.")
    parser.add_argument("fill_prob_dir", type=str, help="Path to fill_prob_X or a single lattice_run dir")
    parser.add_argument("slice_id", type=int, help="Slice ID to extract (0–29, or -1 for full cloud)")
    parser.add_argument("output_csv", type=str, help="Output CSV path")

    parser.add_argument("--cut_edge_key", type=str, default="exposed_edge_length", help="Column name for cut edge length")
    parser.add_argument("--perim_key", type=str, default="mirrored_perimeter", help="Column name for pre-mirroring perimeter")
    parser.add_argument("--slice_id_key", type=str, default="slice_id", help="Column name for slice ID")

    args = parser.parse_args()
    fill_prob_dir = Path(args.fill_prob_dir)
    output_csv = Path(args.output_csv)

    print(f"[START] Extracting slice_id={args.slice_id} from {fill_prob_dir}")
    raw_data = stream_cut_edge_vs_half_perim_for_id(
        fill_prob_dir=fill_prob_dir,
        slice_id=args.slice_id,
        cut_edge_key=args.cut_edge_key,
        perim_key=args.perim_key,
        slice_id_key=args.slice_id_key
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cut_edge", "slice_perimeter"])
        row_count = 0
        for cut_edge, half_perim in raw_data:
            writer.writerow([cut_edge, half_perim])
            row_count += 1

    print(f"[DONE] Wrote {row_count} rows to {output_csv}")

if __name__ == "__main__":
    main()
