import gzip
import csv
from pathlib import Path
from typing import Iterator, Tuple
import math

def stream_slice_data_for_id(
    fill_prob_dir: Path,
    slice_id: int,
    area_key: str = "mirrored_area",
    perim_key: str = "mirrored_perimeter",
    slice_id_key: str = "slice_id"
) -> Iterator[Tuple[float, float]]:
    """
    Yields (area, perimeter) for a given slice_id across all lattice CSVs.
    Handles both fill_prob_dir and single lattice_run dirs.
    """
    slice_id_str = str(slice_id)

    # Determine if we're given a single lattice_run dir or a full fill_prob dir
    if (fill_prob_dir / "slice_data.csv.gz").exists():
        lattice_dirs = [fill_prob_dir]
    else:
        lattice_dirs = sorted(fill_prob_dir.glob("lattice_run_*"))
        if not lattice_dirs:
            print(f"[ERROR] No lattice_run_* dirs found in {fill_prob_dir}")
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
