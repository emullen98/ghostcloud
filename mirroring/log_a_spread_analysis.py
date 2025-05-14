import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === User settings ===
csv_dirs    = ["earlier_runs/finished_may2/csvs", "earlier_runs/finished_may3/csvs"]  # ← replace with your actual folders
results_csv = "results/bucket_D_estimates.csv"            # ← your file with columns: slice_id, D_est

# === 1) Load slice estimates ===
df_res = pd.read_csv(results_csv)  # must have exactly: slice_id, D_est

# === 2) Re-load raw data and compute D_full ===
all_dfs = []
for d in csv_dirs:
    for fp in sorted(glob.glob(os.path.join(d, "lattice_*.csv"))):
        all_dfs.append(pd.read_csv(fp))
df_all = pd.concat(all_dfs, ignore_index=True)

# full‐clouds are slice_id == -1
df_full = df_all[df_all["slice_id"] == -1]
if df_full.empty:
    raise RuntimeError("No full‐cloud rows (slice_id == -1).")
logA = np.log(df_full["area"])
logP = np.log(df_full["perimeter"])
slope, _, _, _, _ = linregress(logA, logP)
D_full = 2 * slope

# === 3) Compute slice fractions & D_error ===
max_slice    = int(df_all["slice_id"].max())
num_slices   = max_slice + 1
df_res["slice_fraction"] = df_res["slice_id"] / num_slices
df_res["D_error"]        = df_res["D_est"] - D_full

# === 4) Compute ln(A) spread per slice ===
records = []
df_slices = df_all[df_all["slice_id"] >= 0]
for sid in df_res["slice_id"].unique():
    sub = df_slices[df_slices["slice_id"] == sid]
    if sub.empty:
        continue
    lnA = np.log(sub["area"])
    records.append({
        "slice_id": sid,
        "lnA_spread": lnA.max() - lnA.min()
    })
df_spread = pd.DataFrame(records)

# === 5) Merge & plot ===
df_plot = pd.merge(df_res, df_spread, on="slice_id")

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(df_plot["slice_fraction"], df_plot["D_error"], 'o-', label="Error in D")
ax1.set_xlabel("Slice Fraction")
ax1.set_ylabel("D_error (D_est - D_full)")

ax2 = ax1.twinx()
ax2.plot(df_plot["slice_fraction"], df_plot["lnA_spread"], 'x--', label="Spread of ln(Area)")
ax2.set_ylabel("Spread of ln(Area)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right")

plt.title("D_error & ln(A) Spread vs Slice Fraction")
plt.grid(True)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/error_and_spread_vs_slice_fraction.png", dpi=300)
plt.show()
