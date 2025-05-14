import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure, convolve
from scipy.stats import linregress
from datetime import datetime
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

def generate_lattice(width, height, prob):
    return np.random.rand(height, width) < prob

def filter_features_under_threshold(labeled_sky, threshold):
    counts = np.bincount(labeled_sky.ravel())
    valid = counts > threshold
    valid[0] = False
    mask = valid[labeled_sky]
    filtered_labels, new_count = label(mask, structure=generate_binary_structure(rank=2, connectivity=1))
    return filtered_labels, new_count

def count_edge_contacts(coords, shape):
    rows, cols = coords[:, 0], coords[:, 1]
    height, width = shape
    return np.array([
        np.sum(cols == 0),
        np.sum(rows == 0),
        np.sum(cols == width - 1),
        np.sum(rows == height - 1)
    ])

def interpret_edge_touch(edge_counts):
    left, top, right, bottom = edge_counts > 0
    num_touched = np.count_nonzero(edge_counts)
    if num_touched == 0: return 0
    if num_touched == 1: return 1
    if num_touched == 2:
        if (left and right) or (top and bottom): return -1
        else: return 2
    return -1

def crop_to_bounding_box(mask):
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min():rows.max()+1, cols.min():cols.max()+1]

def slice_cloud_into_segments(cloud, num_slices):
    h, w = cloud.shape
    base_width = w // num_slices
    remainder = w % num_slices
    slices, naive_r_exposed, left_edges, right_edges = [], [], [], []

    start_col = 0
    for i in range(num_slices):
        slice_width = base_width + (1 if i < remainder else 0)
        end_col = start_col + slice_width
        segment = cloud[:, start_col:end_col]
        slices.append(segment)
        left_edges.append(segment[:, 0].astype(bool))
        right_edges.append(segment[:, -1].astype(bool))
        naive_r_exposed.append(np.count_nonzero(segment[:, -1]))
        start_col = end_col

    shared_edges = [np.count_nonzero(np.logical_and(right_edges[i], left_edges[i + 1]))
                    for i in range(num_slices - 1)]
    return slices, shared_edges, naive_r_exposed

def compute_area(mask):
    return np.count_nonzero(mask)

def compute_perimeter(mask):
    mask = mask.astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)
    neighbor_counts = convolve(mask, kernel, mode='constant', cval=0)
    return int(np.sum(mask * (4 - neighbor_counts)))

def compute_D(areas, perims):
    if len(areas) < 2:
        return None
    log_area = np.log(areas)
    log_perim = np.log(perims)
    slope, _, _, _, _ = linregress(log_area, log_perim)
    return 2 * slope

def process_lattice(index, seed, lattice_size, fill_prob, area_thresh, num_slices):
    print(f"[Lattice {index}] Running on PID {os.getpid()}")
    np.random.seed(seed)

    full_sky = generate_lattice(lattice_size, lattice_size, fill_prob)
    full_sky = binary_fill_holes(full_sky)
    labelled_sky, num_clouds = label(full_sky)
    labelled_sky, num_clouds = filter_features_under_threshold(labelled_sky, area_thresh)

    records = []

    for cloud_id in tqdm(range(1, num_clouds + 1), desc=f"Lattice {index} Clouds", leave=False):
        mask = (labelled_sky == cloud_id)
        coords = np.argwhere(mask)
        if interpret_edge_touch(count_edge_contacts(coords, mask.shape)) == 0:
            cloud = crop_to_bounding_box(mask)
            if cloud.shape[1] < 2 * num_slices:
                continue  # Too narrow to slice reliably
            area = compute_area(cloud)
            perim = compute_perimeter(cloud)

            records.append({
                "cloud_id": cloud_id,
                "slice_id": -1,
                "area": area,
                "perimeter": perim
            })

            slices, shared_edges, r_exposed = slice_cloud_into_segments(cloud, num_slices)
            raw_areas = [compute_area(s) for s in slices]
            raw_perims = [compute_perimeter(s) for s in slices]

            for i in range(num_slices):
                mirr_area = np.sum(raw_areas[:i+1]) * 2
                raw_perim = np.sum(raw_perims[:i+1])
                shared = np.sum(shared_edges[:i]) if i > 0 else 0
                right_cut = r_exposed[i]
                mirr_perim = (raw_perim - 2 * shared - right_cut) * 2

                records.append({
                    "cloud_id": cloud_id,
                    "slice_id": i,
                    "area": mirr_area,
                    "perimeter": mirr_perim
                })

    df = pd.DataFrame(records)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"csvs/lattice_{index}_{timestamp}.csv"

    # Ensure csvs/ directory exists in subprocess
    os.makedirs("csvs", exist_ok=True)

    df.to_csv(csv_path, index=False)
    print(f"[Lattice {index}] Saved {len(df)} rows to {csv_path}")
    return df

def load_all_csvs(csv_dir="csvs"):
    all_dfs = []
    for file in sorted(glob.glob(os.path.join(csv_dir, "lattice_*.csv"))):
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError(f"No CSV files found in '{csv_dir}'. Did the lattice processing run correctly?")
    return pd.concat(all_dfs, ignore_index=True)

def split_into_area_bins(df_full, num_bins=10):
    log_areas = np.log10(df_full["area"])
    bin_edges = np.linspace(log_areas.min(), log_areas.max(), num_bins + 1)

    # Real area ranges from log-space edges
    real_edges = 10 ** bin_edges
    labels = [
        f"[{int(real_edges[i])}, {int(real_edges[i+1])})"
        for i in range(num_bins)
    ]

    df_full["size_bin"] = pd.cut(
        log_areas, bins=bin_edges, labels=labels, include_lowest=True
    )
    return df_full, labels

def balance_buckets(df_full_binned):
    counts = df_full_binned["size_bin"].value_counts()
    min_count = counts.min()
    balanced_df = df_full_binned.groupby("size_bin", group_keys=False).apply(
        lambda g: g.sample(min(len(g), min_count), random_state=42)
    )
    return balanced_df.reset_index(drop=True)

def compute_bucket_D_curves(df_all, df_balanced, num_slices):
    results = []
    for size_bin in sorted(df_balanced["size_bin"].unique()):
        cloud_ids = df_balanced[df_balanced["size_bin"] == size_bin]["cloud_id"].values
        for slice_id in range(num_slices):
            df_slice = df_all[(df_all["slice_id"] == slice_id) & (df_all["cloud_id"].isin(cloud_ids))]
            D_est = compute_D(df_slice["area"], df_slice["perimeter"])
            results.append({"size_bin": size_bin, "slice_id": slice_id, "D_est": D_est})
    return pd.DataFrame(results)

def plot_loglog_full(df_full, save_dir="plots"):
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

if __name__ == "__main__":
    import multiprocessing as mp

    print(f"Main process PID: {os.getpid()}")
    print(f"Available CPUs: {mp.cpu_count()}")

    os.makedirs("csvs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # === PARAMETERS ===
    LATTICE_SIZE = 10000
    FILL_PROB = 0.405
    AREA_THRESHOLD = 3000
    NUM_LATTICES = 200
    NUM_SLICES = 30

    seeds = np.random.randint(0, 100000, size=NUM_LATTICES)
    max_workers = min(NUM_LATTICES, mp.cpu_count())

    # ✅ Use 'spawn' to avoid macOS fork bugs
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(process_lattice, i, seeds[i], LATTICE_SIZE, FILL_PROB, AREA_THRESHOLD, NUM_SLICES)
            for i in range(NUM_LATTICES)
        ]
        for _ in tqdm(as_completed(futures), total=NUM_LATTICES, desc="Lattices Finished"):
            pass

    print("Loading and analyzing results...")
    df_all = load_all_csvs()
    df_full = df_all[df_all["slice_id"] == -1].copy()

    print("Binning full clouds into log-area bins...")
    df_binned, bin_labels = split_into_area_bins(df_full, num_bins=10)

    print("Balancing bins for fair comparison...")
    df_balanced = balance_buckets(df_binned)

    print("Plotting log-log curve and computing full D...")
    D_full = plot_loglog_full(df_full, save_dir="plots")

    print("Computing D estimates per size bin per slice...")
    df_bucket_D = compute_bucket_D_curves(df_all, df_balanced, num_slices=NUM_SLICES)

    print("Plotting D curves...")
    plot_bucket_D_curves(df_bucket_D, D_full, save_dir="plots")

    print("Saving output tables...")
    df_binned.to_csv("results/full_clouds_binned.csv", index=False)
    df_balanced.to_csv("results/full_clouds_balanced.csv", index=False)
    df_bucket_D.to_csv("results/bucket_D_estimates.csv", index=False)

    print("Done.")
