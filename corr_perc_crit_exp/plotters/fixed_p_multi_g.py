import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_summary(path):
    df = pd.read_csv(path)
    d_full = df[df["slice_id"] == -1]["D_estimate"].values[0]
    df = df[df["slice_id"] != -1].copy()
    df["slice_ratio"] = (df["slice_id"] + 1) / (df["slice_id"].max() + 1)
    df["delta_D"] = df["D_estimate"] - d_full
    return df, d_full

# === CONFIG ===
p_dir = "p_0p460"
gammas = ["g_0p1975", "g_0p200", "g_0p2025", "g_0p205"]

base_path = Path("/scratch/mbiju2/storm/cp_crit_exp_aggregated_20250711/analysis")

plt.figure(figsize=(10, 6))

for gamma_dir in gammas:
    summary_path = base_path / gamma_dir / p_dir / "summary_filtered_slice_d_vals.csv"
    if not summary_path.exists():
        print(f"[WARN] Missing: {summary_path}")
        continue
    df, d_full = load_summary(summary_path)
    plt.plot(df["slice_ratio"], df["D_estimate"], label=f"{gamma_dir}", marker="o")

plt.xlabel("Slice ratio")
plt.ylabel("D estimate")
plt.title(f"D estimates vs slices for p={p_dir}")
plt.legend()
plt.tight_layout()
plt.savefig(f"d_vs_slices_fixed_p_{p_dir}.png")
print(f"[INFO] Plot saved to d_vs_slices_fixed_p_{p_dir}.png")
