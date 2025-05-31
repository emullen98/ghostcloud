"""
Created May 20 2025
Updated May 20 2025

Show that the downsampling technique for the correlation function does an excellent job of preserving its structure

Creates a site percolation lattice at the percolation threshold (p_c = 0.5927...)

Computes the correlation function in two ways: one with no downsampling, the other with downsampling
"""


import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange
from numba.typed import List
from scipy.ndimage import label, find_objects
import common_code as cc


def _generate_lattice(lx, ly, prob=0.5927):
    my_arr = np.random.choice([0, 1], size=(ly, lx), p=[1 - prob, prob]).astype(int)

    my_max_distance = int(round(np.hypot(my_arr.shape[0], my_arr.shape[1])))

    # my_arr = binary_fill_holes(my_arr).astype(int)
    my_arr, num_features = label(my_arr)

    return my_arr, my_max_distance, num_features


def _build_offset_distance(n_rows, n_cols):
    """
    Precompute (dx, dy) offsets and their rounded Euclidean distances.
    Handles all directions, including diagonals.
    """
    max_distance = int(np.ceil(np.hypot(n_rows - 1, n_cols - 1)))

    offsets = []
    distances = []

    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            if dx == 0 and dy == 0:
                continue
            euclid_dist = np.hypot(dx, dy)
            d = int(round(euclid_dist))
            if 0 < d <= max_distance:
                offsets.append((dx, dy))
                distances.append(d)

    return (np.array(offsets, dtype=np.int64),
            np.array(distances, dtype=np.int64))


@njit(parallel=True)
def _pair_correlation(grid: np.ndarray, offsets, distances, unique_r, frac):
    """
    Compute the pair connectivity (correlation) function g(r) for a binary grid.

    Parameters
    ----------
    grid : 2D numpy array of 0s and 1s
    offsets : Nx2 array of (dx, dy) offsets
    distances : length-N array of rounded distances corresponding to each offset
    unique_r : sorted 1D array of unique distances to evaluate

    Returns
    -------
    1D array of correlation values for each r in unique_r
    """
    n_r = unique_r.shape[0]  # number of possible euclidean distances between points
    occ_pairs_result = np.zeros(n_r, dtype=np.float64)
    tot_pairs_result = np.zeros(n_r, dtype=np.float64)

    # Occupied sites as a list of tuples: [(i1, j1), (i2, j2), ...]
    occ = List()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                occ.append((i, j))
    n_occ = len(occ)
    if n_occ == 0:
        return occ_pairs_result, tot_pairs_result

    rand_idxs = np.random.choice(a=n_occ, replace=False, size=int(frac * n_occ))

    # Loop over distances in parallel
    for ir in prange(n_r):  # n_r is number of possible euclidean distances between points

        r = unique_r[ir]

        # Count how many offsets correspond to this r
        # For example, for a euclidean distance of 1 there are 8 offsets that yield this distance
        count_offsets = 0
        for d in distances:
            if d == r:
                count_offsets += 1

        # Total number of pairs to check for this distance
        # For example, say there are 5 occupied sites in the lattice
        # Then for each occupied site, we must check the 8 sites that are a euclidean distance of r = 1 away from this site
        # And repeat the procedure for each euclidean distance r
        tot_pairs = n_occ * count_offsets

        if tot_pairs == 0:
            occ_pairs_result[ir] = 0.0
            tot_pairs_result[ir] = 0.0
            continue

        # sum occupied pairs
        occ_pairs = 0
        # for p in range(n_occ):
        for p in rand_idxs:
            x, y = occ[p]
            for k in range(offsets.shape[0]):
                if distances[k] == r:
                    dx, dy = offsets[k]
                    tx, ty = x + dx, y + dy
                    if 0 <= tx < grid.shape[0] and 0 <= ty < grid.shape[1]:
                        occ_pairs += grid[tx, ty]
        occ_pairs_result[ir] = occ_pairs
        tot_pairs_result[ir] = tot_pairs

    return occ_pairs_result, tot_pairs_result


def compute_corr_func(lx, ly, frac: int | float = 1):
    arr, max_dist, num_features = _generate_lattice(lx, ly)

    slices = find_objects(arr)

    occ_count = np.zeros(max_dist + 1)
    tot_count = np.zeros(max_dist + 1)
    for i in range(num_features):
        cloud = arr[slices[i]]
        cloud = np.where(cloud == i + 1, 1, 0)
        if cloud.size == 1:
            continue

        offsets, distances = _build_offset_distance(cloud.shape[0], cloud.shape[1])
        unique_r = np.unique(distances)

        occ_count_temp, tot_count_temp = _pair_correlation(cloud, offsets, distances, unique_r, frac=frac)
        occ_count_temp = np.pad(occ_count_temp, (1, len(occ_count) - (len(occ_count_temp) + 1)), mode='constant', constant_values=(0, 0))
        tot_count_temp = np.pad(tot_count_temp, (1, len(tot_count) - (len(tot_count_temp) + 1)), mode='constant', constant_values=(0, 0))
        occ_count += occ_count_temp
        tot_count += tot_count_temp

    corr_func = np.divide(occ_count, tot_count, out=np.zeros_like(occ_count, dtype=float), where=tot_count != 0)
    corr_func[0] = 1

    distances = np.arange(max_dist + 1)

    return corr_func, distances


if __name__ == '__main__':
    cf, dists = compute_corr_func(200, 200)
    cf_ds, dists_ds = compute_corr_func(200, 200, frac=0.01)
    plt.plot(dists, cf, label='full')
    plt.plot(dists_ds, cf_ds*100, label='downsampled')
    plt.legend()
    plt.loglog()
    plt.show()

