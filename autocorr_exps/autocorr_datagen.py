#!/usr/bin/env python3
"""
autocorr_datagen.py

Compute Wiener–Khinchin (WK) radial autocorrelation for all clouds in a
single site-percolation lattice, saving:

  1) Per-cloud C(r) rows (Parquet, sharded) under: <outdir>/per_cloud/
     - columns: cloud_idx, perim, area, cr, cr_bnd (nullable)
  2) Aggregated C(r) across all clouds as simple text files (one value/line):
     - <run_tag>_Cr.txt        (centers = all cloud pixels)
     - <run_tag>_Cr_bnd.txt    (centers = 4-connected boundary pixels)

We intentionally DO NOT save per-cloud numerators/denominators nor an
aggregated num/den file—C(r) is sufficient for analysis, totals are only
used in-memory to compute the aggregate C(r).
"""

import os
import argparse
import numpy as np

import clouds.utils.autocorr_utils as autocorr_utils
import clouds.utils.cloud_utils as cloud_utils
from clouds.utils.autocorr_utils import (
    xp, xp_backend, ParquetWriter, CloudRow,
)

# ---------------------------
# Host copy helper (NumPy)
# ---------------------------
def _to_numpy(arr):
    if xp_backend == "cupy":
        return xp.asnumpy(arr)
    return np.asarray(arr)

def main():
    parser = argparse.ArgumentParser(
        description="WK autocorr datagen: save per-cloud C(r) (Parquet) and aggregated C(r)."
    )
    parser.add_argument("--lattice-size", type=int, required=True, help="Square lattice size L.")
    parser.add_argument("--fill-prob", type=float, required=True, help="Site percolation p.")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed.")
    parser.add_argument("--min-area", type=int, default=1000, help="Min cloud area to include.")
    parser.add_argument("--max-area", type=int, default=7500000, help="Max cloud area to include.")
    parser.add_argument("--outdir", type=str, default="scratch", help="Output directory.")
    parser.add_argument("--prefix", type=str, default="wk_datagen", help="Filename prefix.")
    parser.add_argument("--rows-per-flush", type=int, default=400, help="Per-cloud rows per Parquet flush.")
    parser.add_argument("--max-bytes-per-flush", type=int, default=128 * 1024 * 1024,
                        help="Approx bytes threshold per flush (~128MB).")
    args = parser.parse_args()

    L = args.lattice_size
    P = args.fill_prob
    SEED = args.seed
    MIN_AREA = args.min_area
    MAX_AREA = args.max_area

    # Build run tag first (used for both aggregates and per-cloud dir)
    run_tag = f"{args.prefix}_L{L}_p{P:.4f}_seed{SEED}"

    # ---- Aggregated outputs (kept at top level of outdir)
    os.makedirs(args.outdir, exist_ok=True)
    cr_outfile_all = os.path.join(args.outdir, f"{run_tag}_Cr.txt")         # all centers
    cr_outfile_bnd = os.path.join(args.outdir, f"{run_tag}_Cr_bnd.txt")     # boundary centers

    # ---- Per-cloud outputs in their own subdirectory:
    #      <outdir>/per_cloud/<run_tag>/cr.part00000.parquet, ...
    per_cloud_dir = os.path.join(args.outdir, "per_cloud", run_tag)
    os.makedirs(per_cloud_dir, exist_ok=True)

    writer = ParquetWriter(
        outdir=per_cloud_dir,
        basename="cr",  # files will be cr.part00000.parquet, cr.part00001.parquet, ...
        rows_per_flush=args.rows_per_flush,
        max_bytes_per_flush=args.max_bytes_per_flush,
    )

    # 0) Print all params
    print(f"=== WK autocorr datagen ===")
    print(f"Parameters:")
    print(f"  Lattice size L:        {L}")
    print(f"  Fill prob p:           {P}")
    print(f"  RNG seed:              {SEED}")
    print(f"  Min cloud area:        {MIN_AREA}")
    print(f"  Max cloud area:        {MAX_AREA}")
    print(f"  Output dir:            {args.outdir}")
    print(f"  Filename prefix:       {args.prefix}")
    print(f"  Rows/flush (Parquet):  {args.rows_per_flush}")
    print(f"  Max bytes/flush:       {args.max_bytes_per_flush}")
    print(f"  XP backend:            {xp_backend}")
    print(f"Outputs:")
    print(f"  Aggregate C(r):        {cr_outfile_all} (all centers)")
    print(f"  Aggregate C(r)_bnd:    {cr_outfile_bnd} (boundary centers)")
    print(f"  Per-cloud C(r) rows:   {per_cloud_dir}/cr.partXXXXX.parquet")
    print(f"==========================")

    # 1) Lattice
    raw_lattice = cloud_utils.generate_site_percolation_lattice(L, L, P, seed=SEED)

    # 2) Clouds
    flood_filled_lattice, _ = cloud_utils.flood_fill_and_label_features(raw_lattice)
    cropped_clouds = cloud_utils.extract_cropped_clouds_by_size(
        flood_filled_lattice, min_area=MIN_AREA, max_area=MAX_AREA
    )

    if len(cropped_clouds) == 0:
        # Still write empty aggregates for parity
        for path in (cr_outfile_all, cr_outfile_bnd):
            with open(path, "w") as f:
                pass
        writer.close()
        print(f"[OK] No clouds >= min_area={MIN_AREA}. Wrote empty aggregate files.")
        print(f"[OK] No per-cloud parts written.")
        return

    # 3) r_max per cloud + ring cache
    h_list = [c.shape[0] for c in cropped_clouds]
    w_list = [c.shape[1] for c in cropped_clouds]
    r_arr, R_global = autocorr_utils.rmax_diagonal_batch(h_list, w_list)
    R_global = int(_to_numpy(R_global))
    autocorr_utils.build_rings_to(R_global)

    # 4) Aggregated totals (kept only in-memory to compute aggregate C(r))
    total_num_all  = xp.zeros(0, dtype=xp.float64)
    total_den_all  = xp.zeros(0, dtype=xp.float64)
    total_num_bnd  = xp.zeros(0, dtype=xp.float64)
    total_den_bnd  = xp.zeros(0, dtype=xp.float64)

    try:
        for ci, (cloud, r_max) in enumerate(zip(cropped_clouds, r_arr)):
            r_max = int(_to_numpy(r_max))
            cloud_xp = xp.asarray(cloud, dtype=xp.uint8)
            cloud_perim = cloud_utils.compute_perimeter(cloud)
            cloud_area  = cloud_utils.compute_area(cloud)

            # Minimal linear padding
            padded_cloud, _ = autocorr_utils.pad_for_wk(cloud_xp, r_max, guard=0)

            # WK autocorr (dual): returns dict {"all": (N_r, D_r), "boundary": (N_r, D_r)}
            out = autocorr_utils.wk_radial_autocorr_dual(
                padded_cloud, r_max, dtype_fft=xp.float64
            )

            # Per-cloud C(r)
            num_all, den_all = out["all"]
            cr_all = xp.where(den_all > 0, num_all / den_all, xp.nan)

            num_bnd, den_bnd = out["boundary"]
            # Boundary centers could (pathologically) be empty; handle safely
            has_bnd = (den_bnd.sum() > 0)
            cr_bnd = xp.where(den_bnd > 0, num_bnd / den_bnd, xp.nan) if has_bnd else None

            # Persist per-cloud row (cr_bnd optional)
            writer.add(CloudRow(
                cloud_idx=ci,
                perim=cloud_perim,
                area=cloud_area,
                cr=_to_numpy(cr_all),
                cr_bnd=(None if cr_bnd is None else _to_numpy(cr_bnd)),
            ))

            # Aggregate totals to compute overall C(r) at the end
            total_num_all = autocorr_utils.extend_and_add(total_num_all, xp.asarray(num_all))
            total_den_all = autocorr_utils.extend_and_add(total_den_all, xp.asarray(den_all))

            if cr_bnd is not None:
                total_num_bnd = autocorr_utils.extend_and_add(total_num_bnd, xp.asarray(num_bnd))
                total_den_bnd = autocorr_utils.extend_and_add(total_den_bnd, xp.asarray(den_bnd))

            # Cleanup GPU temporaries; keep allocator pool warm
            del padded_cloud, cloud_xp, num_all, den_all, cr_all, num_bnd, den_bnd, cr_bnd, out
            if xp_backend == "cupy":
                import cupy as cp
                cp.cuda.Stream.null.synchronize()

    finally:
        writer.close()

    # 5) Save aggregated C(r) for both center sets
    C_r_all = xp.where(total_den_all > 0, total_num_all / total_den_all, xp.nan)
    with open(cr_outfile_all, "w") as f:
        for val in _to_numpy(C_r_all):
            f.write(f"{val}\n")

    # Boundary aggregate only if we accumulated any boundary centers
    if int(_to_numpy(total_den_bnd.sum())) > 0:
        C_r_bnd = xp.where(total_den_bnd > 0, total_num_bnd / total_den_bnd, xp.nan)
        with open(cr_outfile_bnd, "w") as f:
            for val in _to_numpy(C_r_bnd):
                f.write(f"{val}\n")
        print(f"[OK] Aggregated C(r)_bnd -> {cr_outfile_bnd}")
    else:
        # Write an empty file for parity/metadata-driven pipelines
        with open(cr_outfile_bnd, "w") as f:
            pass
        print(f"[OK] No boundary centers aggregated. Wrote empty {cr_outfile_bnd}")

    print(f"[OK] Aggregated C(r)      -> {cr_outfile_all}")
    print(f"[OK] Per-cloud C(r) Parquet parts under: {per_cloud_dir}/")

if __name__ == "__main__":
    main()
