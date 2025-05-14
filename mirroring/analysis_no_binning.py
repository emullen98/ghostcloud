import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def load_all_csvs_from_dirs(dirs):
    all_dfs = []
    offset = 0

    for dir_path in dirs:
        csv_files = sorted(glob.glob(os.path.join(dir_path, "lattice_*.csv")))
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            df["source_file"] = os.path.basename(file_path)

            # Shift this CSV’s cloud_id by the current offset
            df["cloud_id"] = df["cloud_id"] + offset

            # Append…
            all_dfs.append(df)

            # Then bump offset by how many unique cloud_ids were in this file
            offset += df["cloud_id"].nunique()

    if not all_dfs:
        raise RuntimeError("No CSV files found in the given directories.")

    return pd.concat(all_dfs, ignore_index=True)

# === Fractal dimension D ===
def compute_D(areas, perims):
    if len(areas) < 2 or len(perims) < 2:
        return None
    log_area = np.log(areas)
    log_perim = np.log(perims)
    slope, _, _, _, _ = linregress(log_area, log_perim)
    return 2 * slope

# === Per-slice D curve ===
def compute_bucket_D_curves(df_all, num_slices):
    results = []
    for slice_id in range(num_slices):
        df_slice = df_all[(df_all["slice_id"] == slice_id)]
        D_est = compute_D(df_slice["area"], df_slice["perimeter"])
        results.append({"slice_id": slice_id, "D_est": D_est})
    return pd.DataFrame(results)

# === Per-slice D curve ===
def compute_combined_slice_D_estimate(df_all):
    df_slice = df_all[(df_all["slice_id"] != -1)]
    D_est = compute_D(df_slice["area"], df_slice["perimeter"])
    return D_est

# === Per-slice D curve ===
def compute_combined_slice_D_estimate_d_x_min(df_all):
    df_slice = df_all[(df_all["slice_id"]) != -1 & (df_all["area"] > 3000)]
    D_est = compute_D(df_slice["area"], df_slice["perimeter"])
    return D_est

# === Plotting ===
def plot_loglog(df_full, save_dir="plots", plot_title="deafualt_title", file_name="defualt_plot_name"):
    print(len(df_full))
    os.makedirs(save_dir, exist_ok=True)
    areas = df_full["area"]
    perims = df_full["perimeter"]
    log_area = np.log(areas)
    log_perim = np.log(perims)
    D_full = compute_D(areas, perims)

    plt.figure(figsize=(6, 5))
    plt.scatter(log_area, log_perim, alpha=0.4, s=1)
    slope, intercept, *_ = linregress(log_area, log_perim)
    xfit = np.linspace(log_area.min(), log_area.max(), 100)
    yfit = slope * xfit + intercept
    plt.plot(xfit, yfit, color='red', label=f"Slope = {slope:.3f} → D = {2*slope:.3f}")
    plt.xlabel("log(Area)")
    plt.ylabel("log(Perimeter)")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(save_dir, file_name + ".png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return D_full

def plot_bucket_D_curves(df_bucket_D, D_full, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(df_bucket_D["slice_id"], df_bucket_D["D_est"])
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

def plot_combined_D_curves(df_combined_D, D_full, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(df_combined_D["slice_id"], df_combined_D["D_est"])
    plt.axhline(y=D_full, color='black', linestyle='--', label=f"Full D = {D_full:.3f}")
    plt.xlabel("Slice Index")
    plt.ylabel("Estimated D")
    plt.title("D Estimates vs Slice Index (Balanced, by Size Bin)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(save_dir, "combined_D_estimates_vs_full_D.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_combined_D_curves(df_combined_D, D_full, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(df_combined_D["slice_id"], df_combined_D["D_est"])
    plt.axhline(y=D_full, color='black', linestyle='--', label=f"Full D = {D_full:.3f}")
    plt.xlabel("Slice Index")
    plt.ylabel("Estimated D")
    plt.title("D Estimates vs Slice Index (Balanced, by Size Bin)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(save_dir, "combined_D_estimates_vs_full_D.png")
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

    df_combined = df_all[df_all["slice_id"] != -1].copy()

    print("Plotting full log-log D fit...")
    D_full = plot_loglog(df_full, save_dir="plots", plot_title= "Full Clouds: log(Perimeter) vs log(Area)",file_name="full_loglog_D_plot")

    print("Computing per-bin D curves...")
    df_bucket_D = compute_bucket_D_curves(df_all, num_slices=30)

    print("Computing combined D estimate...")
    # combined_D_est = compute_combined_slice_D_estimate(df_all)
    combined_D_est_x_min = compute_combined_slice_D_estimate_d_x_min(df_all)

    plot_loglog(df_combined, save_dir="plots", plot_title= "All Sliced Clouds: log(Perimeter) vs log(Area)",file_name="sliced_combined_loglog_D_plot")

    print("Plotting D vs slice index curves...")
    plot_bucket_D_curves(df_bucket_D, D_full, save_dir="plots")

    # print("Combined D estimate for all slice ratios and sliced clouds is "+str(combined_D_est))

    print("XMIN_Combined D estimate for all slice ratios and sliced clouds is "+str(combined_D_est_x_min))

    print("Saving output tables...")
    os.makedirs("results", exist_ok=True)
    df_bucket_D.to_csv("results/combined_D_estimates.csv", index=False)

    print("Done.")
