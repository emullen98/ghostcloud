import gzip
import csv
from pathlib import Path
from typing import Iterator, Tuple, Optional
import math
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt


def stream_slice_data_for_id(
    fill_prob_dir: Path,
    slice_id: int,
    area_key: str = "mirrored_area",
    perim_key: str = "mirrored_perimeter",
    slice_id_key: str = "slice_id"
) -> Iterator[Tuple[float, float]]:
    """
    Yields (area, perimeter) for a given slice_id across all lattice CSVs.
    Handles both fill_prob_dir and single run dirs.
    """
    slice_id_str = str(slice_id)

    # Determine if we're given a single run dir or a full fill_prob dir
    if (fill_prob_dir / "slice_data.csv.gz").exists():
        lattice_dirs = [fill_prob_dir]
    else:
        lattice_dirs = sorted(fill_prob_dir.glob("run_*"))
        if not lattice_dirs:
            print(f"[ERROR] No run_* dirs found in {fill_prob_dir}")
            return

    file_count = 0
    row_count = 0
    matched_rows = 0

    for lattice_dir in lattice_dirs:
        csv_path = lattice_dir / "slice_data.csv.gz"
        if not csv_path.exists():
            print(f"[WARN] Missing file: {csv_path}")
            continue

        file_count += 1
        try:
            with gzip.open(csv_path, "rt", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if area_key not in reader.fieldnames or perim_key not in reader.fieldnames:
                    print(f"[WARN] Missing expected columns in {csv_path}: {reader.fieldnames}")
                    continue

                for row in reader:
                    row_count += 1
                    if row.get(slice_id_key) == slice_id_str:
                        try:
                            yield float(row[area_key]), float(row[perim_key])
                            matched_rows += 1
                        except ValueError:
                            print(f"[WARN] Invalid float values in {csv_path}: {row}")
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path}: {e}")

    print(f"[INFO] Processed {file_count} files, {row_count} rows, {matched_rows} matched slice_id={slice_id}")

def stream_slice_data_for_id(
    fill_prob_dir: Path,
    slice_id: int,
    area_key: str = "mirrored_area",
    perim_key: str = "mirrored_perimeter",
    slice_id_key: str = "slice_id"
) -> Iterator[Tuple[float, float]]:
    """
    Yields (area, perimeter) for a given slice_id across all lattice CSVs.
    Handles both fill_prob_dir and single run dirs.
    """
    slice_id_str = str(slice_id)

    # Determine if we're given a single run dir or a full fill_prob dir
    if (fill_prob_dir / "slice_data.csv.gz").exists():
        lattice_dirs = [fill_prob_dir]
    else:
        lattice_dirs = sorted(fill_prob_dir.glob("run_*"))
        if not lattice_dirs:
            print(f"[ERROR] No run_* dirs found in {fill_prob_dir}")
            return

    file_count = 0
    row_count = 0
    matched_rows = 0

    for lattice_dir in lattice_dirs:
        csv_path = lattice_dir / "slice_data.csv.gz"
        if not csv_path.exists():
            print(f"[WARN] Missing file: {csv_path}")
            continue

        file_count += 1
        try:
            with gzip.open(csv_path, "rt", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if area_key not in reader.fieldnames or perim_key not in reader.fieldnames:
                    print(f"[WARN] Missing expected columns in {csv_path}: {reader.fieldnames}")
                    continue

                for row in reader:
                    row_count += 1
                    if row.get(slice_id_key) == slice_id_str:
                        try:
                            yield float(row[area_key]), float(row[perim_key])
                            matched_rows += 1
                        except ValueError:
                            print(f"[WARN] Invalid float values in {csv_path}: {row}")
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path}: {e}")

    print(f"[INFO] Processed {file_count} files, {row_count} rows, {matched_rows} matched slice_id={slice_id}")

def stream_cut_edge_vs_half_perim_for_id(
    fill_prob_dir: Path,
    slice_id: int,
    cut_edge_key: str = "exposed_edge_length",
    perim_key: str = "mirrored_perimeter",
    slice_id_key: str = "slice_id"
) -> Iterator[Tuple[float, float]]:
    """
    Yields (cut_edge_length, pre_mirror_perimeter / 2) for a given slice_id
    across all lattice CSVs. Compatible with fill_prob_dir or single run_* dir.
    """
    slice_id_str = str(slice_id)

    # Determine if we're given a single run dir or a full fill_prob dir
    if (fill_prob_dir / "slice_data.csv.gz").exists():
        lattice_dirs = [fill_prob_dir]
    else:
        lattice_dirs = sorted(fill_prob_dir.glob("run_*"))
        if not lattice_dirs:
            print(f"[ERROR] No run_* dirs found in {fill_prob_dir}")
            return

    file_count = 0
    row_count = 0
    matched_rows = 0

    for lattice_dir in lattice_dirs:
        csv_path = lattice_dir / "slice_data.csv.gz"
        if not csv_path.exists():
            print(f"[WARN] Missing file: {csv_path}")
            continue

        file_count += 1
        try:
            with gzip.open(csv_path, "rt", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if cut_edge_key not in reader.fieldnames or perim_key not in reader.fieldnames:
                    print(f"[WARN] Missing expected columns in {csv_path}: {reader.fieldnames}")
                    continue

                for row in reader:
                    row_count += 1
                    if row.get(slice_id_key) == slice_id_str:
                        try:
                            cut_edge = float(row[cut_edge_key])
                            half_perim = float(row[perim_key]) / 2
                            if half_perim > 0:
                                yield cut_edge, half_perim
                                matched_rows += 1
                        except ValueError:
                            print(f"[WARN] Invalid float values in {csv_path}: {row}")
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path}: {e}")

    print(f"[INFO] Processed {file_count} files, {row_count} rows, {matched_rows} matched slice_id={slice_id}")

def filter_max_area(
    data: Iterator[Tuple[float, float]],
    max_area: float
) -> Iterator[Tuple[float, float]]:
    """
    Filters out clouds with area > max_area.
    Returns same (area, perimeter) tuple stream.
    """
    for area, perim in data:
        if area <= max_area:
            yield area, perim

def filter_min_area(
    data: Iterator[Tuple[float, float]],
    min_area: float
) -> Iterator[Tuple[float, float]]:
    """
    Filters out clouds with area < min_area.
    Returns same (area, perimeter) tuple stream.
    """
    for area, perim in data:
        if area >= min_area:
            yield area, perim

def convert_to_log_values(
    data_iter: Iterator[Tuple[float, float]],
    base: float = math.e,
    verbose: bool = False
) -> Iterator[Tuple[float, float]]:
    """
    Converts (area, perimeter) tuples to (log_area, log_perimeter).
    """
    log_fn = math.log if base == math.e else lambda x: math.log(x, base)
    skipped = 0
    yielded = 0

    for area, perim in data_iter:
        try:
            if area > 0 and perim > 0:
                yielded += 1
                yield log_fn(area), log_fn(perim)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    if verbose:
        print(f"[INFO] Converted {yielded} pairs to log-space, skipped {skipped} invalid rows")


def write_slice_data_to_csv(
    data: Iterator[Tuple[float, float]],
    output_path: Path,
    header: Tuple[str, str] = ("log_area", "log_perimeter")
):
    """
    Writes (log_area, log_perimeter) tuples to a CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for a, p in data:
            writer.writerow([a, p])
            rows_written += 1

    if rows_written == 0:
        print(f"[WARN] No data written to {output_path} — check slice ID or filters.")
    else:
        print(f"[INFO] Wrote {rows_written} rows to {output_path}")

def estimate_d_from_csv(csv_path: Path) -> float:
    """
    Estimates D from a CSV file with log_area and log_perimeter columns.
    Raises if fewer than 2 rows.
    """
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        raise ValueError(f"Not enough data in {csv_path} to estimate D")
    slope, _, _, _, _ = linregress(df["log_area"], df["log_perimeter"])
    return 2*slope

def append_d_to_summary(
    summary_csv_path: Path,
    slice_id: int,
    d_estimate: float,
    lock_path: Optional[Path] = None
):
    """
    Appends a single line to the summary CSV with locking.
    Creates the file with header if not exists.

    Parameters:
    - summary_csv_path: Path to the output summary CSV
    - slice_id: slice index (can be -1 for full cloud)
    - d_estimate: the estimated D value
    - lock_path: optional lock file path (default: summary_csv_path + '.lock')
    """
    from filelock import FileLock

    if lock_path is None:
        lock_path = summary_csv_path.with_suffix(".lock")

    header = "slice_id,D_estimate\n"
    row = f"{slice_id},{d_estimate:.6f}\n"

    with FileLock(str(lock_path)):
        new_file = not summary_csv_path.exists()
        with summary_csv_path.open("a") as f:
            if new_file:
                f.write(header)
            f.write(row)


def get_slice_log_csv_path(base_dir: Path, slice_id: int) -> Path:
    """
    Returns the correct log_data.csv path for the given slice_id.
    - For slice_id -1: full_cloud_log_data.csv
    - For slice_id >= 0: cloud_slice_XX_log_data.csv
    """
    if slice_id == -1:
        return base_dir / "full_cloud_log_data.csv"
    else:
        return base_dir / f"cloud_slice_{slice_id:02d}_filtered_log_data.csv"


def get_cut_edge_csv_path(analysis_dir: Path, slice_id: int) -> Path:
    return analysis_dir / f"cloud_slice_{slice_id:02d}_cut_edge_data.csv"


def load_summary_with_deltas(summary_path: Path) -> pd.DataFrame:
    """
    Loads summary CSV with slice_id and D_estimate, computes slice_ratio and delta_D.
    """
    df = pd.read_csv(summary_path)
    if -1 not in df["slice_id"].values:
        raise ValueError(f"No full cloud entry (slice_id == -1) in {summary_path}")
    
    d_full = df[df["slice_id"] == -1]["D_estimate"].values[0]
    df = df[df["slice_id"] != -1].copy()
    df["slice_ratio"] = (df["slice_id"] + 1) / (df["slice_id"].max() + 1)
    df["delta_D"] = df["D_estimate"] - d_full
    return df

def plot_fill_prob_curves(summary_paths: list[Path], labels: list[str], output_path: Path):

    plt.figure(figsize=(10, 6))
    for path, label in zip(summary_paths, labels):
        try:
            df = load_summary_with_deltas(path)
            df = df.sort_values("slice_ratio")
            plt.plot(df["slice_ratio"], df["delta_D"], marker="o", label=label)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Slice Ratio")
    plt.ylabel("Delta D (D_slice - D_full)")
    plt.title("Delta D vs Slice Ratio Across Fill Probs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Combined plot saved to {output_path}")

def plot_single_fill_prob(summary_path: Path, output_path: Path):

    df = load_summary_with_deltas(summary_path)
    df = df.sort_values("slice_ratio")
    plt.figure(figsize=(8, 5))
    plt.plot(df["slice_ratio"], df["delta_D"], marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Slice Ratio")
    plt.ylabel("Delta D (D_slice - D_full)")
    plt.title(summary_path.parent.name)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Plot saved to {output_path}")