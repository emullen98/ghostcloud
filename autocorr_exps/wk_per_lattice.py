#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt  # kept to mirror your import style

import clouds.utils.autocorr_utils as autocorr_utils
import clouds.utils.cloud_utils as cloud_utils
from clouds.utils.autocorr_utils import xp  # xp = numpy or cupy per your utils

def _to_numpy(arr):
    return xp.asnumpy(arr) if hasattr(xp, "asnumpy") else np.asarray(arr)

def main():
    parser = argparse.ArgumentParser(
        description="Compute C(r) via WK (analytic binning) for a single lattice/seed."
    )
    parser.add_argument("--lattice-size", type=int, required=True)
    parser.add_argument("--fill-prob", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="scratch")
    parser.add_argument("--prefix", type=str, default="wk_matching")
    args = parser.parse_args()

    LATTICE_SIZE = args.lattice_size
    FILL_PROB = args.fill_prob
    SEED = args.seed
    MIN_AREA = args.min_area

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f"{args.prefix}_{LATTICE_SIZE}_seed_{SEED}.txt")
    numden_outfile = os.path.join(args.outdir, f"{args.prefix}_{LATTICE_SIZE}_seed_{SEED}_numden.txt")

    # 1) Generate lattice (site percolation)
    raw_lattice = cloud_utils.generate_site_percolation_lattice(
        LATTICE_SIZE, LATTICE_SIZE, FILL_PROB, seed=SEED
    )

    # 2) Flood fill & extract cropped clouds
    flood_filled_lattice, _ = cloud_utils.flood_fill_and_label_features(raw_lattice)
    cropped_clouds = cloud_utils.extract_cropped_clouds_by_size(
        flood_filled_lattice, min_area=MIN_AREA
    )

    # Early exit if nothing to do
    if len(cropped_clouds) == 0:
        with open(outfile, "w") as f:
            pass
        print(f"[OK] No clouds >= min_area={MIN_AREA}. Wrote empty {outfile}")
        return

    # 3) Compute per-cloud r_max via diagonal rule and prebuild ring cache once
    h_list = [c.shape[0] for c in cropped_clouds]
    w_list = [c.shape[1] for c in cropped_clouds]
    r_arr, R_global = autocorr_utils.rmax_diagonal_batch(h_list, w_list)  # uses global xp
    R_global = int(_to_numpy(R_global))                 # python int

    # Build the ring-index cache once for the run (views will serve smaller r)
    autocorr_utils.build_rings_to(R_global)

    # 4) Accumulate N(r), D(r) across all clouds using optimized WK
    total_num = xp.zeros(0, dtype=float)
    total_denom = xp.zeros(0, dtype=float)

    for cloud, r_max in zip(cropped_clouds, r_arr):
        r_max = int(_to_numpy(r_max))
        cloud = xp.asarray(cloud, dtype=xp.uint8)

        # r_max = floor(sqrt(2 * side^2)) — matches your earlier scheme

        # Pad by exactly r_max (guarantees denominator shortcut is valid)
        padded_cloud, _ = autocorr_utils.pad_for_wk(cloud, r_max, guard=0)

        # Optimized WK: rFFT for numerator; denominator = |f| * ring_counts(r)
        num_temp, denom_temp = autocorr_utils.wk_radial_autocorr(
            padded_cloud, r_max, dtype_fft=xp.float64
        )

        # Ensure xp arrays (extend_and_add expects xp arrays)
        num_temp = xp.asarray(num_temp)
        denom_temp = xp.asarray(denom_temp)

        total_num = autocorr_utils.extend_and_add(total_num, num_temp)
        total_denom = autocorr_utils.extend_and_add(total_denom, denom_temp)

    # 5) Final C(r) and save (one value per line)
    C_r = xp.where(total_denom > 0, total_num / total_denom, xp.nan)
    C_r_np = _to_numpy(C_r)
    with open(outfile, "w") as f:
        for val in C_r_np:
            f.write(f"{val}\n")

    with open(numden_outfile, "w") as f:
        for num, den in zip(total_num, total_denom):
            f.write(f"{num}\t{den}\n")

    print(f"[OK] Wrote {outfile}")
    print(f"[OK] Wrote {numden_outfile}")

if __name__ == "__main__":
    main()
