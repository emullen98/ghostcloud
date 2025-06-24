import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from slice_analysis_utils import *
import csv
from pathlib import Path
import argparse
from filelock import FileLock

def main():
    parser = argparse.ArgumentParser(description="Average cut edge/perimeter ratio for a slice and append to summary.")
    parser.add_argument("analysis_dir", type=str, help="Path to fill_prob_XXXX analysis directory")
    parser.add_argument("slice_id", type=int, help="Slice ID to analyze (0–29)")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    cut_edge_csv = get_cut_edge_csv_path(analysis_dir, args.slice_id)
    summary_csv = analysis_dir / "summary_cut_edge_ratio.csv"

    if not cut_edge_csv.exists():
        print(f"[WARN] Missing cut edge CSV: {cut_edge_csv}")
        return

    ratios = []
    with open(cut_edge_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cut_edge = float(row["cut_edge"])
                perim = float(row["slice_perimeter"])
                if perim > 0:
                    ratios.append(cut_edge / perim)
            except Exception as e:
                print(f"[WARN] Skipping row: {row} ({e})")

    if not ratios:
        print(f"[WARN] No valid ratios found in {cut_edge_csv}")
        return

    avg_ratio = sum(ratios) / len(ratios)

    # Append or update the summary CSV
    # If file doesn't exist, create and write header

    lock_path = str(summary_csv) + ".lock"
    with FileLock(lock_path):
        write_header = not summary_csv.exists()
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["slice_id", "avg_cut_edge_perimeter_ratio"])
            writer.writerow([args.slice_id, avg_ratio])

    print(f"[INFO] Appended avg ratio {avg_ratio:.6f} for slice_id={args.slice_id} to {summary_csv}")

if __name__ == "__main__":
    main()