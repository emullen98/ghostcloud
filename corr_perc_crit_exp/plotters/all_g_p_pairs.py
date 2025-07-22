import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_summary(path):
    df = pd.read_csv(path)
    d_full_row = df[df["slice_id"] == -1]
    if d_full_row.empty:
        raise ValueError(f"No full cloud (slice_id == -1) in {path}")
    d_full = d_full_row["D_estimate"].values[0]

    df_slices = df[df["slice_id"] != -1].copy()
    df_slices["slice_ratio"] = (df_slices["slice_id"] + 1) / (df_slices["slice_id"].max() + 1)
    df_slices["delta_D"] = df_slices["D_estimate"] - d_full
    df_slices = df_slices.sort_values("slice_ratio")
    return df_slices, d_full

# === CONFIG ===
gammas = ["g_0p1975", "g_0p200", "g_0p2025", "g_0p205"]
ps = ["p_0p460", "p_0p465", "p_0p470", "p_0p475", "p_0p480", "p_0p485", "p_0p490", "p_0p495"]

base_path = Path("/scratch/mbiju2/storm/cp_crit_exp_aggregated_20250711/analysis")

plt.figure(figsize=(14, 10))

for gamma_dir in gammas:
    for p_dir in ps:
        summary_path = base_path / gamma_dir / p_dir / "summary_filtered_slice_d_vals.csv"
        if not summary_path.exists():
            print(f"[WARN] Missing: {summary_path}")
            continue
        try:
            df_slices, d_full = load_summary(summary_path)
            label = f"{gamma_dir}, {p_dir}"
            plt.plot(
                df_slices["slice_ratio"],
                df_slices["delta_D"],
                label=label,
                marker="o",
                alpha=0.7
            )
        except Exception as e:
            print(f"[ERROR] Skipping {summary_path}: {e}")

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Slice ratio")
plt.ylabel("ΔD (D_slice - D_full)")
plt.title("Delta D vs Slice Ratio (All gamma–p pairs)")
plt.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig("delta_D_all_gamma_p.png")
print("[INFO] Plot saved to delta_D_all_gamma_p.png")
