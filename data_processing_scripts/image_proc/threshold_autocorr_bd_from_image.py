#!/usr/bin/env python3
"""
threshold_autocorr_bd_from_image.py

For a given grayscale image:
  - normalize to [0,1]
  - for each threshold in the list: binarize -> flood fill -> WK autocorr
  - per-cloud: write WK fields (opt), + boundary-distance histogram, COM, Rg
  - accumulate numerators/denominators across ALL thresholds
  - save a single aggregate C(r) and C(r)_bnd per image
  - save a metadata JSON with input file and parameters
"""

import os
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

import clouds.utils.autocorr_utils as autocorr_utils
import clouds.utils.cloud_utils as cloud_utils
from clouds.utils.autocorr_utils import xp, xp_backend

from clouds.utils.image_utils import (
    load_gray_float,
    normalize_to_unit,
    threshold_binary,
)

# Parquet helpers + minimal metrics (added utils)
from clouds.utils.autocorr_utils import (
    ParquetWriter, CloudRow,
    compute_com,                # NEW
    radius_of_gyration,         # NEW
    boundary_distances_min,     # NEW
    boundary_histogram_min,     # NEW
)

# ---------------------------
# Helpers
# ---------------------------
def _to_numpy(arr):
    if xp_backend == "cupy":
        return xp.asnumpy(arr)
    return np.asarray(arr)

def parse_thresholds_file(path: str) -> List[float]:
    vals: List[float] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            v = float(s)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Threshold {v} from {path} not in [0,1]")
            vals.append(v)
    seen = set()
    out: List[float] = []
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Threshold + WK + boundary-distance per-cloud datagen for a single image."
    )
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument(
        "--thresholds-file",
        type=str,
        required=True,
        help="Path to a text file with one threshold per line (blank lines and lines starting with '#' are ignored).",
    )
    ap.add_argument("--min-area", type=int, default=1000)
    ap.add_argument("--max-area", type=int, default=7_500_000)
    ap.add_argument("--outdir", type=str, default="scratch/threshold_autocorr_bd")
    ap.add_argument("--prefix", type=str, default="thr_autocorr_bd")
    ap.add_argument("--rows-per-flush", type=int, default=400)
    ap.add_argument("--max-bytes-per-flush", type=int, default=128 * 1024 * 1024)

    # WK save controls (unchanged defaults)
    ap.add_argument("--save-cr", action="store_true",
                    help="Save per-cloud C(r) for centers=all to Parquet (default: off).")
    ap.add_argument("--save-cr-bnd", action="store_true",
                    help="Save per-cloud C(r) for centers=boundary to Parquet (default: off).")
    ap.add_argument("--save-numden", action="store_true",
                    help="Save per-cloud numerators/denominators (all & boundary) to Parquet (default: on).")
    ap.set_defaults(save_numden=True)

    # Boundary-distance controls (shared Δr; zero-origin alignment)
    ap.add_argument("--bd-bin-width", type=float, default=1.0,
                    help="Δr for boundary-distance histogram (shared across clouds).")

    args = ap.parse_args()

    img_path = Path(args.image)
    img_stem = img_path.stem
    run_tag = f"{args.prefix}_{img_stem}"

    os.makedirs(args.outdir, exist_ok=True)

    # aggregate outputs (names kept similar, script renamed)
    cr_outfile_all = os.path.join(args.outdir, f"{run_tag}_Cr.txt")
    cr_outfile_bnd = os.path.join(args.outdir, f"{run_tag}_Cr_bnd.txt")
    meta_outfile   = os.path.join(args.outdir, f"{run_tag}_meta.json")

    # per-cloud parquet shard dir + writer
    per_cloud_dir = os.path.join(args.outdir, "per_cloud", run_tag)
    os.makedirs(per_cloud_dir, exist_ok=True)
    writer = ParquetWriter(
        outdir=per_cloud_dir,
        basename="cloud_metrics",   # NEW basename (we store more than C(r))
        rows_per_flush=args.rows_per_flush,
        max_bytes_per_flush=args.max_bytes_per_flush,
    )

    thresholds = parse_thresholds_file(args.thresholds_file)

    print("=== Threshold + WK + BD datagen ===")
    print(f"Image:                 {img_path}")
    print(f"Run tag:               {run_tag}")
    print(f"Thresholds:            {thresholds}")
    print(f"Area filter:           [{args.min_area}, {args.max_area}]")
    print(f"Outdir:                {args.outdir}")
    print(f"XP backend:            {xp_backend}")
    print(f"Per-cloud parquet dir: {per_cloud_dir}")
    print(f"BD bin width (Δr):     {args.bd_bin_width}")
    print("==============================")

    # Load + normalize image
    raw = load_gray_float(img_path)
    img01 = normalize_to_unit(raw)

    # Aggregated totals across ALL thresholds (WK)
    total_num_all = xp.zeros(0, dtype=xp.float64)
    total_den_all = xp.zeros(0, dtype=xp.float64)
    total_num_bnd = xp.zeros(0, dtype=xp.float64)
    total_den_bnd = xp.zeros(0, dtype=xp.float64)

    # Per-threshold cloud counts (keys as strings for JSON)
    num_clouds_by_threshold = {}

    # Maintain ring cache upper bound
    R_global_built = 0

    try:
        cloud_counter = 0
        for t in thresholds:
            lattice_bin = threshold_binary(img01, t)

            filled, _ = cloud_utils.flood_fill_and_label_features(lattice_bin)
            cropped_clouds = cloud_utils.extract_cropped_clouds_by_size(
                filled, min_area=args.min_area, max_area=args.max_area
            )

            num_clouds_by_threshold[f"{t:.12g}"] = len(cropped_clouds)

            if len(cropped_clouds) == 0:
                print(f"[INFO] threshold={t:.3f}: no clouds in area range; skipping.")
                continue

            h_list = [c.shape[0] for c in cropped_clouds]
            w_list = [c.shape[1] for c in cropped_clouds]
            r_arr, R_needed = autocorr_utils.rmax_diagonal_batch(h_list, w_list)
            R_needed = int(_to_numpy(R_needed))
            if R_needed > R_global_built:
                autocorr_utils.build_rings_to(R_needed)
                R_global_built = R_needed

            for r_max, cloud in zip(r_arr, cropped_clouds):
                r_max = int(_to_numpy(r_max))
                cloud_xp = xp.asarray(cloud, dtype=xp.uint8)

                # --- geometry tags
                perim = cloud_utils.compute_perimeter(cloud)
                area  = cloud_utils.compute_area(cloud)

                # --- WK autocorr (all + boundary)
                padded, _ = autocorr_utils.pad_for_wk(cloud_xp, r_max, guard=0)
                out = autocorr_utils.wk_radial_autocorr_dual(
                    padded, r_max, dtype_fft=xp.float64
                )

                num_all, den_all = out["all"]
                cr_all = xp.where(den_all > 0, num_all / den_all, xp.nan)

                num_bnd, den_bnd = out["boundary"]
                has_bnd = (den_bnd.sum() > 0)
                cr_bnd = (
                    xp.where(den_bnd > 0, num_bnd / den_bnd, xp.nan)
                    if has_bnd
                    else None
                )

                # --- minimal boundary-distance metrics (shared Δr; zero-origin)
                cy, cx = compute_com(cloud)  # floats (row, col)
                rgA = radius_of_gyration(cloud, center=(cy, cx), pixels="area")
                r_bnd = boundary_distances_min(cloud, center=(cy, cx))
                bd_r, bd_counts = boundary_histogram_min(
                    r_bnd, bin_width=args.bd_bin_width, include_zero_bin=True
                )
                Nb = int(r_bnd.size)

                # --- decide what to persist (WK)
                cr_save      = _to_numpy(cr_all) if args.save_cr else None
                cr_bnd_save  = (None if (cr_bnd is None or not args.save_cr_bnd) else _to_numpy(cr_bnd))

                num_all_save = _to_numpy(num_all) if args.save_numden else None
                den_all_save = _to_numpy(den_all) if args.save_numden else None
                num_bnd_save = None
                den_bnd_save = None
                if args.save_numden and cr_bnd is not None:
                    num_bnd_save = _to_numpy(num_bnd)
                    den_bnd_save = _to_numpy(den_bnd)

                # --- per-cloud row
                writer.add(
                    CloudRow(
                        cloud_idx=cloud_counter,
                        perim=int(perim),
                        area=int(area),

                        # WK (optional)
                        cr=cr_save,
                        cr_bnd=cr_bnd_save,
                        num_all=num_all_save,
                        den_all=den_all_save,
                        num_bnd=num_bnd_save,
                        den_bnd=den_bnd_save,

                        # threshold tag
                        threshold=float(t),

                        # NEW: minimal boundary-distance + COM/Rg tags
                        com=np.array([cy, cx], dtype=np.float64),
                        rg_area=float(rgA),
                        rg_bnd=None,  # filled later if you choose boundary Rg

                        bd_r=bd_r,
                        bd_counts=bd_counts,

                        bd_bin_width=float(args.bd_bin_width),
                        center_method="com",
                        boundary_connectivity="4c",
                        bd_n=Nb,
                    )
                )
                cloud_counter += 1

                # --- aggregate across ALL thresholds (WK only)
                total_num_all = autocorr_utils.extend_and_add(total_num_all, xp.asarray(num_all))
                total_den_all = autocorr_utils.extend_and_add(total_den_all, xp.asarray(den_all))
                if cr_bnd is not None:
                    total_num_bnd = autocorr_utils.extend_and_add(total_num_bnd, xp.asarray(num_bnd))
                    total_den_bnd = autocorr_utils.extend_and_add(total_den_bnd, xp.asarray(den_bnd))

                # cleanup
                del padded, cloud_xp, num_all, den_all, cr_all, num_bnd, den_bnd, cr_bnd, out
                if xp_backend == "cupy":
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()

    finally:
        writer.close()

    # --- aggregates (WK)
    C_r_all = xp.where(total_den_all > 0, total_num_all / total_den_all, xp.nan)
    with open(cr_outfile_all, "w") as f:
        for v in _to_numpy(C_r_all):
            f.write(f"{v}\n")
    print(f"[OK] Aggregated C(r)      -> {cr_outfile_all}")

    if int(_to_numpy(total_den_bnd.sum())) > 0:
        C_r_bnd = xp.where(total_den_bnd > 0, total_num_bnd / total_den_bnd, xp.nan)
        with open(cr_outfile_bnd, "w") as f:
            for v in _to_numpy(C_r_bnd):
                f.write(f"{v}\n")
        print(f"[OK] Aggregated C(r)_bnd  -> {cr_outfile_bnd}")
    else:
        with open(cr_outfile_bnd, "w") as f:
            pass
        print(f"[OK] No boundary centers aggregated.")

    # --- metadata (record Δr + tags for reproducibility)
    meta = {
        "image": str(img_path),
        "run_tag": run_tag,
        "thresholds": thresholds,
        "thresholds_file": args.thresholds_file,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "rows_per_flush": args.rows_per_flush,
        "max_bytes_per_flush": args.max_bytes_per_flush,
        "xp_backend": xp_backend,
        "num_clouds_by_threshold": num_clouds_by_threshold,
        "save_cr": args.save_cr,
        "save_cr_bnd": args.save_cr_bnd,
        "save_numden": args.save_numden,

        # NEW: boundary-distance config
        "bd_bin_width": args.bd_bin_width,
        "center_method": "com",
        "boundary_connectivity": "4c",
    }
    with open(meta_outfile, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Metadata -> {meta_outfile}")

    print(f"[OK] Per-cloud parts under {per_cloud_dir}/")


if __name__ == "__main__":
    main()
