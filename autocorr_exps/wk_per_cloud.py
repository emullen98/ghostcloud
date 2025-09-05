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
        description="Compute C(r) via WK (analytic binning) for a single lattice/seed, and save per-cloud C(r)."
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

    # previous outputs (kept)
    outfile = os.path.join(args.outdir, f"{args.prefix}_{LATTICE_SIZE}_seed_{SEED}.txt")
    numden_outfile = os.path.join(args.outdir, f"{args.prefix}_{LATTICE_SIZE}_seed_{SEED}_numden.txt")

    # NEW: per-cloud NPZ output
    percloud_npz = os.path.join(args.outdir, f"{args.prefix}_{LATTICE_SIZE}_seed_{SEED}_percloud_cr.npz")

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
        with open(numden_outfile, "w") as f:
            pass
        # write empty NPZ in CSR form
        np.savez_compressed(
            percloud_npz,
            cr_values=np.array([], dtype=np.float32),
            cr_offsets=np.array([0], dtype=np.int64),
            cloud_ids=np.array([], dtype=np.int64),
            r_max=np.array([], dtype=np.int32),
            lattice_size=np.array(LATTICE_SIZE, dtype=np.int64),
            fill_prob=np.array(FILL_PROB, dtype=np.float64),
            seed=np.array(SEED, dtype=np.int64),
            min_area=np.array(MIN_AREA, dtype=np.int64),
        )
        print(f"[OK] No clouds >= min_area={MIN_AREA}. Wrote empty {outfile}, {numden_outfile}, {percloud_npz}")
        return

    # 3) Compute per-cloud r_max via diagonal rule and prebuild ring cache once
    h_list = [c.shape[0] for c in cropped_clouds]
    w_list = [c.shape[1] for c in cropped_clouds]
    r_arr, R_global = autocorr_utils.rmax_diagonal_batch(h_list, w_list)  # uses global xp
    R_global = int(_to_numpy(R_global))  # python int

    # Build the ring-index cache once for the run (views will serve smaller r)
    autocorr_utils.build_rings_to(R_global)

    # 4) Accumulate N(r), D(r) across all clouds using optimized WK (as before)
    total_num = xp.zeros(0, dtype=xp.float64)
    total_denom = xp.zeros(0, dtype=xp.float64)

    # NEW: simple collectors for per-cloud C(r)
    cloud_ids = []
    cr_chunks = []     # list of 1D numpy float32 arrays
    rmax_list = []

    for cloud_id, (cloud, r_max) in enumerate(zip(cropped_clouds, r_arr)):
        r_max = int(_to_numpy(r_max))
        cloud = xp.asarray(cloud, dtype=xp.uint8)

        # Pad by exactly r_max (guarantees denominator shortcut is valid)
        padded_cloud, _ = autocorr_utils.pad_for_wk(cloud, r_max, guard=0)

        # Optimized WK: rFFT for numerator; denominator = |f| * ring_counts(r)
        num_temp, denom_temp = autocorr_utils.wk_radial_autocorr(
            padded_cloud, r_max, dtype_fft=xp.float64
        )

        # Ensure xp arrays (extend_and_add expects xp arrays)
        num_temp = xp.asarray(num_temp)
        denom_temp = xp.asarray(denom_temp)

        # Accumulate to totals
        total_num = autocorr_utils.extend_and_add(total_num, num_temp)
        total_denom = autocorr_utils.extend_and_add(total_denom, denom_temp)

        # --- NEW: per-cloud C(r) capture (trial-run: compute locally then store) ---
        # Use safe division with NaNs when denom == 0
        cr_cloud_xp = xp.where(denom_temp > 0, num_temp / denom_temp, xp.nan)
        cr_cloud_np = _to_numpy(cr_cloud_xp).astype(np.float32, copy=False)

        cloud_ids.append(cloud_id)         # sequential id
        cr_chunks.append(cr_cloud_np)      # variable-length 1D array
        rmax_list.append(r_max)            # for metadata

    # 5) Final overall C(r) and save (as before)
    C_r = xp.where(total_denom > 0, total_num / total_denom, xp.nan)
    C_r_np = _to_numpy(C_r)

    with open(outfile, "w") as f:
        for val in C_r_np:
            f.write(f"{val}\n")

    with open(numden_outfile, "w") as f:
        # Convert totals to numpy once to avoid device/host confusion in file I/O
        total_num_np = _to_numpy(total_num)
        total_denom_np = _to_numpy(total_denom)
        for num, den in zip(total_num_np, total_denom_np):
            f.write(f"{num}\t{den}\n")

    # 6) Pack and save per-cloud C(r) as CSR-style NPZ (compact & fast)
    lengths = np.fromiter((arr.size for arr in cr_chunks), dtype=np.int64, count=len(cr_chunks))
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    values = np.concatenate(cr_chunks, dtype=np.float32)

    np.savez_compressed(
        percloud_npz,
        cr_values=values,
        cr_offsets=offsets,
        cloud_ids=np.asarray(cloud_ids, dtype=np.int64),
        r_max=np.asarray(rmax_list, dtype=np.int32),
        lattice_size=np.array(LATTICE_SIZE, dtype=np.int64),
        fill_prob=np.array(FILL_PROB, dtype=np.float64),
        seed=np.array(SEED, dtype=np.int64),
        min_area=np.array(MIN_AREA, dtype=np.int64),
    )

    print(f"[OK] Wrote {outfile}")
    print(f"[OK] Wrote {numden_outfile}")
    print(f"[OK] Wrote per-cloud C(r) NPZ: {percloud_npz}")


if __name__ == "__main__":
    main()
