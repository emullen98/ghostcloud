import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path("/scratch/mbiju2/storm/multi_fill_prob_20250601_124857/analysis")
    fill_probs = ["0_4066", "0_4069", "0_4072", "0_4074", "0_4076"]

    plt.figure(figsize=(8, 6))

    for prob in fill_probs:
        summary_csv = base_dir / f"fill_prob_{prob}" / "summary_cut_edge_ratio.csv"
        if not summary_csv.exists():
            print(f"[WARN] Missing: {summary_csv}")
            continue

        df = pd.read_csv(summary_csv)
        df = df.sort_values("slice_id")
        plt.plot(df["slice_id"], df["avg_cut_edge_perimeter_ratio"], marker='o', label=f"fill_prob {prob}")

    plt.xlabel("Slice ID")
    plt.ylabel("Average Cut Edge / Perimeter Ratio")
    plt.title("Cut Edge/Perimeter Ratio vs. Slice ID")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "cut_edge_ratio_curves.png")
    plt.show()

if __name__ == "__main__":
    main()