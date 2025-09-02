#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt  # kept to mirror your import style
import clouds.utils.autocorr_utils as autocorr_utils
import clouds.utils.cloud_utils as cloud_utils
from clouds.utils.autocorr_utils import xp  # xp = numpy or cupy per your utils

# NOTE: assumes autocorr_utils exposes:
# - pad_image(array, pad)
# - wk_radial_autocorr_matching(padded_cloud, r_max) -> (N_r, D_r)
# - extend_and_add(a, b)

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

    # 3) Accumulate N(r), D(r) across all clouds
    total_num = xp.zeros(0, dtype=float)
    total_denom = xp.zeros(0, dtype=float)

    for cloud in cropped_clouds:
        cloud = xp.asarray(cloud, dtype=xp.uint8)
        h, w = cloud.shape
        square_side = int(max(h, w))
        # r_max = floor(sqrt(2 * side^2)) — matches your earlier scheme
        max_radius = int(math.floor(math.sqrt(2 * (square_side ** 2))))

        padded_cloud = autocorr_utils.pad_image(cloud, max_radius)
        # Helper should implement WK + r = ceil(sqrt(dx^2+dy^2)) binning
        num_temp, denom_temp = autocorr_utils.wk_radial_autocorr_matching(
            padded_cloud, max_radius
        )

        # ensure xp arrays
        num_temp = xp.asarray(num_temp)
        denom_temp = xp.asarray(denom_temp)

        total_num = autocorr_utils.extend_and_add(total_num, num_temp)
        total_denom = autocorr_utils.extend_and_add(total_denom, denom_temp)

    # 4) Final C(r) and save (one value per line)
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
