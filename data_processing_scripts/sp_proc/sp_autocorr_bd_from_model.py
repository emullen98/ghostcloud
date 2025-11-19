#!/usr/bin/env python3
"""
sp_autocorr_bd_from_model.py

Single-lattice SP pipeline (one p):
  - generate site-perc lattice at (width, height, p, seed)
  - flood fill + crop clouds (area-range filter)
  - per cloud: COM, Rg(area), boundary-distance histogram (Δr shared), WK arrays (optional)
  - aggregate lattice-level C(r) across all clouds
  - write per-cloud Parquet shards + aggregate text + metadata JSON

Per-cloud schema matches the image pipeline, but stores p_val instead of threshold.
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

from clouds.utils.autocorr_utils import (
    ParquetWriter, CloudRow,
    compute_com,
    radius_of_gyration,
    boundary_distances_min,
    boundary_histogram_min,
)

# ---------------------------
# Helpers
# ---------------------------
def _to_numpy(arr):
    if xp_backend == "cupy":
        return xp.asnumpy(arr)
    return np.asarray(arr)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="WK + boundary-distance per-cloud datagen for a single site-percolation lattice."
    )
    # Lattice parameters (single p)
    ap.add_argument("--width",  type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--p",      type=float, required=True)
    ap.add_argument("--seed",   type=int, required=True)

    # Cloud filters
    ap.add_argument("--min-area", type=int, default=1000)
    ap.add_argument("--max-area", type=int, default=7_500_000)

    # Cloud generation / preprocessing
    ap.add_argument("--order", choices=["LF_bbox", "FL"], default="FL",
                    help="Preprocess order: LF_bbox (label→size→crop→per-bbox fill) or FL (global fill→label→size→crop). Default: FL to match previous behavior.")
    ap.add_argument("--cl", type=int, choices=[4, 8], default=4,
                    help="Label (foreground) connectivity: 4 or 8. Default: 4.")
    ap.add_argument("--cf", type=int, choices=[4, 8], default=4,
                    help="Flood (background) connectivity for hole filling: 4 or 8. Default: 4.")
    ap.add_argument("--bbox-pad", type=int, default=1,
                    help="Padding (in px) around component bounding boxes for per-bbox fill. 1 recommended to avoid frame artifacts.")

    # IO
    ap.add_argument("--outdir", type=str, default="scratch/sp_autocorr_bd")
    ap.add_argument("--prefix", type=str, default="sp_autocorr_bd")
    ap.add_argument("--rows-per-flush", type=int, default=400)
    ap.add_argument("--max-bytes-per-flush", type=int, default=128 * 1024 * 1024)

    # Save controls (WK)
    ap.add_argument("--save-cr", action="store_true",
                    help="Save per-cloud C(r) for centers=all to Parquet (default: off).")
    ap.add_argument("--save-cr-bnd", action="store_true",
                    help="Save per-cloud C(r) for centers=boundary to Parquet (default: off).")
    ap.add_argument("--save-numden", action="store_true",
                    help="Save per-cloud numerators/denominators (all & boundary) to Parquet (default: on).")
    ap.set_defaults(save_numden=True)

    # Boundary-distance (shared Δr)
    ap.add_argument("--bd-bin-width", type=float, default=1.0,
                    help="Δr for boundary-distance histogram (shared across clouds).")

    args = ap.parse_args()

    run_tag = (
        f"{args.prefix}_W{args.width}_H{args.height}_p{args.p:.6f}_seed{args.seed}"
        f"_ord{args.order}_cl{args.cl}_cf{args.cf}"
    )
    os.makedirs(args.outdir, exist_ok=True)

    # aggregate outputs
    cr_outfile_all = os.path.join(args.outdir, f"{run_tag}_Cr.txt")
    # we omit C(r)_bnd text unless you enable/save boundary num/den
    meta_outfile   = os.path.join(args.outdir, f"{run_tag}_meta.json")

    # per-cloud parquet shard dir + writer
    per_cloud_dir = os.path.join(args.outdir, "per_cloud", run_tag)
    os.makedirs(per_cloud_dir, exist_ok=True)
    writer = ParquetWriter(
        outdir=per_cloud_dir,
        basename="cloud_metrics",
        rows_per_flush=args.rows_per_flush,
        max_bytes_per_flush=args.max_bytes_per_flush,
    )

    print("=== SP + WK + BD datagen (single lattice) ===")
    print(f"Lattice:               {args.height} x {args.width}  seed={args.seed}")
    print(f"p:                     {args.p:.6f}")
    print(f"Area filter:           [{args.min_area}, {args.max_area}]")
    print(f"Outdir:                {args.outdir}")
    print(f"XP backend:            {xp_backend}")
    print(f"Per-cloud parquet dir: {per_cloud_dir}")
    print(f"BD bin width (Δr):     {args.bd_bin_width}")
    print(f"Processing order:       {args.order}")
    print(f"Connectivity (label):   {args.cl}")
    print(f"Connectivity (flood):   {args.cf}")
    print(f"BBox pad (px):          {args.bbox_pad}")
    print("==============================")


    # Aggregated totals (WK)
    total_num_all = xp.zeros(0, dtype=xp.float64)
    total_den_all = xp.zeros(0, dtype=xp.float64)
    total_num_bnd = xp.zeros(0, dtype=xp.float64)
    total_den_bnd = xp.zeros(0, dtype=xp.float64)

    # Build rings cache lazily
    R_global_built = 0

    try:
        cloud_counter = 0

        # --- generate lattice (bool)
        lattice = cloud_utils.generate_site_percolation_lattice(
            width=args.width, height=args.height, fill_prob=args.p, seed=args.seed
        )

        cropped_clouds = cloud_utils.preprocess_and_crop_clouds(
            lattice,
            order=args.order,      # "LF_bbox" or "FL"
            cl=args.cl,            # 4 or 8 (label connectivity)
            cf=args.cf,            # 4 or 8 (flood connectivity)
            min_area=args.min_area,
            max_area=args.max_area,
            bbox_pad=args.bbox_pad,
        )


        if len(cropped_clouds) == 0:
            print("[INFO] no clouds in area range; nothing to write.")
        else:
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

                # --- geometry
                raw_perim, hull_perim, accessible_perim = cloud_utils.compute_perimeters(cloud, cf=args.cf)
                area  = cloud_utils.compute_area(cloud)

                # --- WK autocorr
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

                # --- BD metrics (shared Δr)
                cy, cx = compute_com(cloud)
                rgA = radius_of_gyration(cloud, center=(cy, cx), pixels="area")
                r_bnd = boundary_distances_min(cloud, center=(cy, cx))
                bd_r, bd_counts = boundary_histogram_min(
                    r_bnd, bin_width=args.bd_bin_width, include_zero_bin=True
                )
                Nb = int(r_bnd.size)

                # --- persist selections
                cr_save      = _to_numpy(cr_all) if args.save_cr else None
                cr_bnd_save  = None
                if args.save_cr_bnd and cr_bnd is not None:
                    cr_bnd_save = _to_numpy(cr_bnd)

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
                        perim_raw=int(raw_perim),
                        perim_hull=int(hull_perim),
                        perim_accessible=int(accessible_perim),
                        area=int(area),

                        # WK (optional)
                        cr=cr_save,
                        cr_bnd=cr_bnd_save,
                        num_all=num_all_save,
                        den_all=den_all_save,
                        num_bnd=num_bnd_save,
                        den_bnd=den_bnd_save,

                        # site-perc tag
                        p_val=float(args.p),

                        # BD + COM/Rg
                        com=np.array([cy, cx], dtype=np.float64),
                        rg_area=float(rgA),
                        rg_bnd=None,

                        bd_r=bd_r,
                        bd_counts=bd_counts,

                        bd_bin_width=float(args.bd_bin_width),
                        center_method="com",
                        boundary_connectivity="4c",
                        bd_n=Nb,
                    )
                )
                cloud_counter += 1

                # --- aggregate (WK)
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
    print(f"[OK] Aggregated C(r) -> {cr_outfile_all}")

    # --- metadata
    meta = {
        "width": args.width,
        "height": args.height,
        "p": args.p,
        "seed": args.seed,
        "run_tag": run_tag,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "processing_order": args.order,
        "connectivity_label": args.cl,
        "connectivity_flood": args.cf,
        "dual_topology": ( (args.cl == 4 and args.cf == 8) or (args.cl == 8 and args.cf == 4) ),
        "area_filter_stage": ("pre-fill" if args.order == "LF_bbox" else "post-fill"),
        "bbox_pad": args.bbox_pad,
        "n_cropped_clouds": len(cropped_clouds),
        "rows_per_flush": args.rows_per_flush,
        "max_bytes_per_flush": args.max_bytes_per_flush,
        "xp_backend": xp_backend,
        "bd_bin_width": args.bd_bin_width,
        "center_method": "com",
    }
    with open(meta_outfile, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Metadata -> {meta_outfile}")

    print(f"[OK] Per-cloud parts under {per_cloud_dir}/")


if __name__ == "__main__":
    main()
