#!/usr/bin/env python3
"""
siteperc_rprof_from_model.py

For a single site-percolation lattice:
  - generate lattice at (width, height, p, seed)
  - flood fill + label + crop clouds (area/contact filters)
  - for each cloud: boundary radial profile histogram (shared binning)
  - aggregate counts across ALL clouds in the lattice (C(r)-style simple sum)

Saves:
  (A) Per-cloud Parquet rows (using ParquetWriter/CloudRow) with:
        p_val, area, perim, and (rp_r, rp_counts, rp_pdf, [rp_f_ring if --ring-corrected])
  (B) Lattice-level aggregate text files:
        *_rprof_counts.txt (summed counts)
        *_rprof_pdf.txt    (density over r)
        *_rprof_r.txt      (r-bin centers)
        *_rprof_fring.txt  (optional ring-corrected aggregate)
  (C) Metadata JSON with parameters + summary

Bin alignment:
  - All clouds use the same Δr = --bin-width
  - Bins are zero-origin aligned (include_zero_bin=True)
  - r_centers = (k + 0.5) * Δr (k = 0,1,...)
"""

"""
python -m clouds.r_dist_exps.siteperc_rprof_from_model \
  --width 10000 \
  --height 10000 \
  --p 0.4074 \
  --seed 987 \
  --contact-type internal \
  --min-area 3000 \
  --max-area 2000000 \
  --bin-width 1.0 \
  --center-method com \
  --outdir scratch/expA \
  --prefix expC_internal

"""

import os
import json
import argparse
from pathlib import Path

import numpy as np

import clouds.utils.autocorr_utils as autocorr_utils
import clouds.utils.cloud_utils as cloud_utils
from clouds.utils.autocorr_utils import xp, xp_backend, to_numpy, extend_and_add
from clouds.utils.autocorr_utils import (
    ParquetWriter, CloudRow,
    boundary_distances, radial_histogram_boundary,
)


# ---------------------------
# Helpers
# ---------------------------
def _to_numpy(arr):
    if xp_backend == "cupy":
        import cupy as cp
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Per-cloud Parquet + lattice aggregate for boundary radial profile (site percolation)."
    )
    # Lattice generation
    ap.add_argument("--width",  type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--p",      type=float, required=True, help="Site fill probability in [0,1].")
    ap.add_argument("--seed",   type=int, required=True)

    # Cloud filtering
    ap.add_argument("--min-area", type=int, default=1000)
    ap.add_argument("--max-area", type=int, default=7_500_000)
    ap.add_argument("--contact-type", type=str, default="all",
                    choices=["internal","single_edge","two_edge","mirrorable","valid","non_mirrorable","all"],
                    help="Edge-contact filter for cropped clouds (default: all).")

    # Radial profile controls
    ap.add_argument("--bin-width", type=float, default=1.0)
    ap.add_argument("--center-method", type=str, default="com",
                    choices=["com","max_inscribed"])
    ap.add_argument("--ring-corrected", action="store_true",
                    help="Also compute/save ring-corrected series: counts / (2π r Δr).")

    # IO
    ap.add_argument("--outdir", type=str, default="scratch/siteperc_rprof_single")
    ap.add_argument("--prefix", type=str, default="siteperc_rprof")
    ap.add_argument("--rows-per-flush", type=int, default=400)
    ap.add_argument("--max-bytes-per-flush", type=int, default=128 * 1024 * 1024)

    args = ap.parse_args()

    # Paths
    run_tag = f"{args.prefix}_W{args.width}_H{args.height}_p{args.p:.6f}_seed{args.seed}"
    os.makedirs(args.outdir, exist_ok=True)

    per_cloud_dir = os.path.join(args.outdir, "per_cloud", run_tag)
    os.makedirs(per_cloud_dir, exist_ok=True)

    agg_dir = os.path.join(args.outdir, "aggregates", run_tag)
    os.makedirs(agg_dir, exist_ok=True)

    counts_outfile = os.path.join(agg_dir, f"{run_tag}_rprof_counts.txt")
    pdf_outfile    = os.path.join(agg_dir, f"{run_tag}_rprof_pdf.txt")
    r_outfile      = os.path.join(agg_dir, f"{run_tag}_rprof_r.txt")
    fring_outfile  = os.path.join(agg_dir, f"{run_tag}_rprof_fring.txt")
    meta_outfile   = os.path.join(args.outdir, f"{run_tag}_meta.json")

    writer = ParquetWriter(
        outdir=per_cloud_dir,
        basename="rprof",
        rows_per_flush=args.rows_per_flush,
        max_bytes_per_flush=args.max_bytes_per_flush,
    )

    print("=== Site Percolation: Boundary Radial Profile (single lattice) ===")
    print(f"Run tag:        {run_tag}")
    print(f"Lattice:        {args.height} x {args.width}")
    print(f"p, seed:        {args.p}, {args.seed}")
    print(f"Area filter:    [{args.min_area}, {args.max_area}]")
    print(f"Contact type:   {args.contact_type}")
    print(f"Bin width:      {args.bin_width}")
    print(f"Center method:  {args.center_method}")
    print(f"Ring-corrected: {args.ring_corrected}")
    print(f"XP backend:     {xp_backend}")
    print(f"Per-cloud dir:  {per_cloud_dir}")
    print(f"Aggregate dir:  {agg_dir}")
    print("===============================================================")

    # Generate lattice (bool)
    lattice = cloud_utils.generate_site_percolation_lattice(
        width=args.width, height=args.height, fill_prob=args.p, seed=args.seed
    )

    # Fill + label, then crop valid clouds
    labeled, _ = cloud_utils.flood_fill_and_label_features(lattice)
    cropped_clouds = cloud_utils.extract_cropped_clouds_by_size(
        labeled,
        min_area=args.min_area,
        max_area=args.max_area,
        contact_type=args.contact_type
    )

    print(f"[INFO] clouds retained: {len(cropped_clouds)}")

    # Aggregate counts across clouds (C(r)-style)
    total_counts = xp.zeros(0, dtype=xp.float64)
    max_len = 0  # to reconstruct r-centers later

    try:
        cloud_counter = 0
        for cloud in cropped_clouds:
            # Distances from chosen center to boundary pixels
            bd = autocorr_utils.boundary_distances(
                cloud.astype(np.uint8, copy=False),
                center_method=args.center_method,
                return_coords=False
            )
            # Histogram with zero-origin alignment so bins match across clouds
            hist = radial_histogram_boundary(
                bd["dist"],
                bin_width=args.bin_width,
                include_zero_bin=True,   # <<< ensures common bin edges
                make_pdf=True,
                ring_corrected=args.ring_corrected,
                subpixel_splat=True
            )

            r_centers  = hist["r_centers"]
            counts     = hist["counts"]
            pdf        = hist["pdf"]
            f_ring     = hist.get("f_ring", None)

            # Geometry tags
            perim = cloud_utils.compute_perimeter(cloud)
            area  = cloud_utils.compute_area(cloud)

            # (A) Per-cloud Parquet row
            writer.add(
                CloudRow(
                    cloud_idx=cloud_counter,
                    perim=int(perim),
                    area=int(area),
                    threshold=None,
                    p_val=float(args.p),
                    rp_r=r_centers,
                    rp_counts=counts,
                    rp_pdf=pdf,
                    rp_f_ring=(f_ring if args.ring_corrected else None),
                )
            )
            cloud_counter += 1

            # (B) Aggregate counts
            total_counts = extend_and_add(total_counts, xp.asarray(counts, dtype=xp.float64))
            max_len = max(max_len, counts.shape[0])

            # light GPU sync
            if xp_backend == "cupy":
                import cupy as cp
                cp.cuda.Stream.null.synchronize()

    finally:
        writer.close()

    # --- Finalize lattice-level aggregate ---
    total_counts_np = _to_numpy(total_counts)

    # r-centers reconstructed from zero-origin bins (k+0.5)*Δr
    r_centers_agg = (np.arange(max_len, dtype=np.float64) + 0.5) * float(args.bin_width)

    # Aggregate pdf: density over r so that sum(pdf * Δr) = 1
    delta = float(args.bin_width)
    total_sum = float(total_counts_np.sum())
    pdf_agg = (total_counts_np / (total_sum * delta)) if total_sum > 0 else np.zeros_like(total_counts_np)

    # Optional aggregate ring-corrected
    f_ring_agg = None
    if args.ring_corrected:
        ring_len = 2.0 * np.pi * r_centers_agg * delta
        f_ring_agg = np.zeros_like(total_counts_np, dtype=np.float64)
        safe = ring_len > 0
        f_ring_agg[safe] = total_counts_np[safe] / ring_len[safe]

    # Save aggregate outputs (newline-delimited, like your C(r) dumps)
    with open(counts_outfile, "w") as f:
        for v in total_counts_np:
            f.write(f"{v}\n")
    with open(pdf_outfile, "w") as f:
        for v in pdf_agg:
            f.write(f"{v}\n")
    with open(r_outfile, "w") as f:
        for v in r_centers_agg:
            f.write(f"{v}\n")
    if args.ring_corrected:
        with open(fring_outfile, "w") as f:
            for v in f_ring_agg:
                f.write(f"{v}\n")

    print(f"[OK] Aggregate counts -> {counts_outfile}")
    print(f"[OK] Aggregate pdf    -> {pdf_outfile}")
    print(f"[OK] Aggregate r      -> {r_outfile}")
    if args.ring_corrected:
        print(f"[OK] Aggregate f_ring -> {fring_outfile}")

    # Metadata
    meta = {
        "run_tag": run_tag,
        "width": args.width,
        "height": args.height,
        "p": args.p,
        "seed": args.seed,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "contact_type": args.contact_type,
        "bin_width": args.bin_width,
        "center_method": args.center_method,
        "ring_corrected": args.ring_corrected,
        "xp_backend": xp_backend,
        "per_cloud_dir": per_cloud_dir,
        "aggregate_dir": agg_dir,
        "num_clouds": len(cropped_clouds),
        "bin_alignment": "zero-origin; r_centers = (k+0.5)*bin_width",
    }
    with open(meta_outfile, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Metadata -> {meta_outfile}")


if __name__ == "__main__":
    main()
