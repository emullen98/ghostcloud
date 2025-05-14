import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === Load & Combine ===
def load_all_csvs_from_dirs(dirs):
    all_dfs = []
    for dir_path in dirs:
        csv_files = sorted(glob.glob(os.path.join(dir_path, "lattice_*.csv")))
        for file in csv_files:
            df = pd.read_csv(file)
            df["source_file"] = os.path.basename(file)
            all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No CSV files found in the given directories.")
    return pd.concat(all_dfs, ignore_index=True)

# === Bin by log-area ===
def split_into_area_bins(df_full, num_bins=10):
    log_areas = np.log10(df_full["area"])
    bin_edges = np.linspace(log_areas.min(), log_areas.max(), num_bins + 1)
    real_edges = 10 ** bin_edges
    labels = [f"[{int(real_edges[i])}, {int(real_edges[i+1])})" for i in range(num_bins)]
    df_full["size_bin"] = pd.cut(log_areas, bins=bin_edges, labels=labels, include_lowest=True)
    return df_full, labels

# === Downsample ===
def balance_buckets(df_full_binned):
    counts = df_full_binned["size_bin"].value_counts()
    min_count = counts.min()
    print("\nCloud counts per size bin:")
    for bin_name, count in sorted(counts.items()):
        print(f"  {bin_name}: {count} clouds")
    print(f"\nBalancing to {min_count} clouds per bin...\n")
    return df_full_binned.groupby("size_bin", group_keys=False).apply(
        lambda g: g.sample(min(len(g), min_count), random_state=42)
    ).reset_index(drop=True)

# === Fractal dimension D ===
def compute_D(areas, perims):
    if len(areas) < 2 or len(perims) < 2:
        return None
    log_area = np.log(areas)
    log_perim = np.log(perims)
    slope, _, _, _, _ = linregress(log_area, log_perim)
    return 2 * slope

# === Per-slice D curve ===
def compute_bucket_D_curves(df_all, df_balanced, num_slices):
    results = []
    for size_bin in sorted(df_balanced["size_bin"].unique()):
        cloud_ids = df_balanced[df_balanced["size_bin"] == size_bin]["cloud_id"].values
        for slice_id in range(num_slices):
            df_slice = df_all[(df_all["slice_id"] == slice_id) & (df_all["cloud_id"].isin(cloud_ids))]
            D_est = compute_D(df_slice["area"], df_slice["perimeter"])
            results.append({"size_bin": size_bin, "slice_id": slice_id, "D_est": D_est})
    return pd.DataFrame(results)

# === Plotting ===
def plot_loglog_full(df_full, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    areas = df_full["area"]
    perims = df_full["perimeter"]
    log_area = np.log(areas)
    log_perim = np.log(perims)
    D_full = compute_D(areas, perims)

    plt.figure(figsize=(6, 5))
    plt.scatter(log_area, log_perim, alpha=0.4, s=10)
    slope, intercept, *_ = linregress(log_area, log_perim)
    xfit = np.linspace(log_area.min(), log_area.max(), 100)
    yfit = slope * xfit + intercept
    plt.plot(xfit, yfit, color='red', label=f"Slope = {slope:.3f} → D = {2*slope:.3f}")
    plt.xlabel("log(Area)")
    plt.ylabel("log(Perimeter)")
    plt.title("Full Clouds: log(Perimeter) vs log(Area)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(save_dir, "full_loglog_D_plot.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return D_full

def plot_bucket_D_curves(df_bucket_D, D_full, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for size_bin in sorted(df_bucket_D["size_bin"].unique()):
        subset = df_bucket_D[df_bucket_D["size_bin"] == size_bin]
        plt.plot(subset["slice_id"], subset["D_est"], label=size_bin)
    plt.axhline(y=D_full, color='black', linestyle='--', label=f"Full D = {D_full:.3f}")
    plt.xlabel("Slice Index")
    plt.ylabel("Estimated D")
    plt.title("D Estimates vs Slice Index (Balanced, by Size Bin)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(save_dir, "bucketed_D_estimates_vs_full_D.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# === Main ===
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_mirrored_clouds.py <csv_dir1> [<csv_dir2> ...]")
        sys.exit(1)

    csv_dirs = sys.argv[1:]
    print(f"Loading CSVs from: {csv_dirs}")
    df_all = load_all_csvs_from_dirs(csv_dirs)

    df_full = df_all[df_all["slice_id"] == -1].copy()

    print("Splitting into size bins...")
    df_binned, bin_labels = split_into_area_bins(df_full, num_bins=1)

    print("Balancing bins...")
    df_balanced = balance_buckets(df_binned)

    print("Plotting full log-log D fit...")
    D_full = plot_loglog_full(df_full, save_dir="plots")

    print("Computing per-bin D curves...")
    df_bucket_D = compute_bucket_D_curves(df_all, df_balanced, num_slices=50)

    print("Plotting D vs slice index curves...")
    plot_bucket_D_curves(df_bucket_D, D_full, save_dir="plots")

    print("Saving output tables...")
    os.makedirs("results", exist_ok=True)
    df_binned.to_csv("results/full_clouds_binned.csv", index=False)
    df_balanced.to_csv("results/full_clouds_balanced.csv", index=False)
    df_bucket_D.to_csv("results/bucket_D_estimates.csv", index=False)

    print("Done.")
