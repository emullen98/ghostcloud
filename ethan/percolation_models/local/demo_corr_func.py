import numpy as np
from numba import njit, prange
from scipy.ndimage import binary_fill_holes, label
import timeit


@njit()
def old_corr_func_jit(labeled_lattice, frac=None):
    """
    :param labeled_lattice: (2-D array of ints or floats, required)
    :param frac: (Float, optional) Fraction of sites to randomly draw for calculating g(r).
    :return: [0] Possible integer distances in the lattice; [1] Correlation function at each possible integer distance
    """
    lattice_shape = labeled_lattice.shape
    coords = np.indices(lattice_shape).reshape(len(lattice_shape), -1).T
    max_distance = int(round(np.sqrt(np.sum(np.array(lattice_shape) ** 2))))

    if frac is None:
        pass
    else:
        new_len = int(frac * len(coords))
        idxs = np.random.choice(a=np.arange(len(coords)), size=new_len, replace=False)
        new_coords = np.empty(shape=(len(idxs), coords.shape[1]), dtype=coords.dtype)

        for i in range(len(idxs)):
            new_coords[i] = coords[idxs[i]]
        coords = new_coords

    correlation_function = np.zeros(max_distance + 1)
    correlation_function[0] = 1
    counts = np.zeros(max_distance + 1)

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            inner_coord = coords[i]
            outer_coord = coords[j]
            inner_site = labeled_lattice[inner_coord[0], inner_coord[1]]
            outer_site = labeled_lattice[outer_coord[0], outer_coord[1]]
            if inner_site == 0 and outer_site == 0:
                pass
            elif (inner_site == 0 and outer_site != 0) or (inner_site != 0 and outer_site == 0):
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 1
            elif (inner_site != 0 and outer_site != 0) and inner_site != outer_site:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 2
            elif (inner_site != 0 and outer_site != 0) and inner_site == outer_site:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 2
                correlation_function[r] += 2

    for r in range(max_distance + 1):
        if counts[r] > 0:
            correlation_function[r] /= counts[r]

    possible_distances = np.arange(0, max_distance + 1)

    return possible_distances, correlation_function


@njit(parallel=True)
def old_corr_func_jit_nopython(labeled_lattice, frac=None):
    """
    :param labeled_lattice: (2-D array of ints or floats, required)
    :param frac: (Float, optional) Fraction of sites to randomly draw for calculating g(r).
    :return: [0] Possible integer distances in the lattice; [1] Correlation function at each possible integer distance
    """
    lattice_shape = labeled_lattice.shape
    coords = np.indices(lattice_shape).reshape(len(lattice_shape), -1).T
    max_distance = int(round(np.sqrt(np.sum(np.array(lattice_shape) ** 2))))

    if frac is None:
        pass
    else:
        new_len = int(frac * len(coords))
        idxs = np.random.choice(a=np.arange(len(coords)), size=new_len, replace=False)
        new_coords = np.empty(shape=(len(idxs), coords.shape[1]), dtype=coords.dtype)

        for i in range(len(idxs)):
            new_coords[i] = coords[idxs[i]]
        coords = new_coords

    correlation_function = np.zeros(max_distance + 1)
    correlation_function[0] = 1
    counts = np.zeros(max_distance + 1)

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            inner_coord = coords[i]
            outer_coord = coords[j]
            inner_site = labeled_lattice[inner_coord[0], inner_coord[1]]
            outer_site = labeled_lattice[outer_coord[0], outer_coord[1]]
            if inner_site == 0 and outer_site == 0:
                pass
            elif (inner_site == 0 and outer_site != 0) or (inner_site != 0 and outer_site == 0):
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 1
            elif (inner_site != 0 and outer_site != 0) and inner_site != outer_site:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 2
            elif (inner_site != 0 and outer_site != 0) and inner_site == outer_site:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 2
                correlation_function[r] += 2

    for r in range(max_distance + 1):
        if counts[r] > 0:
            correlation_function[r] /= counts[r]

    possible_distances = np.arange(0, max_distance + 1)

    return possible_distances, correlation_function


def new_corr_func(labeled_lattice: np.ndarray, frac: float = None):
    """
    Computes C(R) = P(site at distance R from an occupied site is in the same cluster).

    Returns
    -------
    r_vals :  np.ndarray, shape=(R_max+1,)
    C       :  np.ndarray, same shape, correlation values
    """
    H, W = labeled_lattice.shape

    # 1) build coordinate lists
    coords_all = np.stack(np.indices((H, W)), axis=2).reshape(-1, 2)
    #    every (row,col) pair

    # 2) filter out the occupied ones
    labels_flat = labeled_lattice[coords_all[:,0], coords_all[:,1]]
    mask_occ   = labels_flat != 0
    coords_occ = coords_all[mask_occ]

    # 3) optional subsampling of occupied references
    if frac is not None and 0 < frac < 1:
        n_ref   = int(frac * coords_occ.shape[0])
        choice  = np.random.choice(coords_occ.shape[0], n_ref, replace=False)
        coords_occ = coords_occ[choice]

    # 4) maximum integer‐rounded radius
    r_max = int(np.hypot(H, W))

    # 5) call the fast numba core
    r_vals, denom_counts, num_counts = _corr_core(
        labeled_lattice, coords_occ, coords_all, r_max
    )

    # 6) form C(R) = num_counts / denom_counts, with C(0)=1 by definition
    C = np.zeros_like(num_counts, dtype=np.float64)
    np.divide(num_counts, denom_counts, out=C, where=denom_counts>0)
    C[0] = 1.0

    return r_vals, C


def _corr_core(lattice, coords_occ, coords_all, r_max):
    """
    Numba‐JIT’d inner loop.

    For each reference i in coords_occ:
      for each target j in coords_all:
        if j != i:
          compute r = round(dist(i,j))
          denom_counts[r] += 1
          if lattice[i] == lattice[j]:
            num_counts[r] += 1
    """
    N_ref = coords_occ.shape[0]
    N_all = coords_all.shape[0]

    denom_counts = np.zeros(r_max + 1, np.int64)
    num_counts   = np.zeros(r_max + 1, np.int64)

    for ii in range(N_ref):
        xi, yi = coords_occ[ii]
        ci = lattice[xi, yi]   # always >0

        for jj in range(N_all):
            xj, yj = coords_all[jj]
            # skip the same site
            if xi == xj and yi == yj:
                continue

            # integer‐rounded radius
            dx = xi - xj
            dy = yi - yj
            r = int(round((dx*dx + dy*dy)**0.5))
            if r > r_max:
                continue

            # increment denominator
            denom_counts[r] += 1

            # if same cluster label, increment numerator
            if lattice[xj, yj] == ci:
                num_counts[r] += 1

    # distances array 0..r_max
    rs = np.arange(r_max + 1)
    return rs, denom_counts, num_counts


def new_corr_func_jit(labeled_lattice: np.ndarray, frac: float = None):
    """
    Computes C(R) = P(site at distance R from an occupied site is in the same cluster).

    Returns
    -------
    r_vals :  np.ndarray, shape=(R_max+1,)
    C       :  np.ndarray, same shape, correlation values
    """
    H, W = labeled_lattice.shape

    # 1) build coordinate lists
    coords_all = np.stack(np.indices((H, W)), axis=2).reshape(-1, 2)
    #    every (row,col) pair

    # 2) filter out the occupied ones
    labels_flat = labeled_lattice[coords_all[:,0], coords_all[:,1]]
    mask_occ   = labels_flat != 0
    coords_occ = coords_all[mask_occ]

    # 3) optional subsampling of occupied references
    if frac is not None and 0 < frac < 1:
        n_ref   = int(frac * coords_occ.shape[0])
        choice  = np.random.choice(coords_occ.shape[0], n_ref, replace=False)
        coords_occ = coords_occ[choice]

    # 4) maximum integer‐rounded radius
    r_max = int(np.hypot(H, W))

    # 5) call the fast numba core
    r_vals, denom_counts, num_counts = _corr_core_jit(
        labeled_lattice, coords_occ, coords_all, r_max
    )

    # 6) form C(R) = num_counts / denom_counts, with C(0)=1 by definition
    C = np.zeros_like(num_counts, dtype=np.float64)
    np.divide(num_counts, denom_counts, out=C, where=denom_counts>0)
    C[0] = 1.0

    return r_vals, C


@njit(parallel=True)
def _corr_core_jit(lattice, coords_occ, coords_all, r_max):
    """
    Numba‐JIT’d inner loop.

    For each reference i in coords_occ:
      for each target j in coords_all:
        if j != i:
          compute r = round(dist(i,j))
          denom_counts[r] += 1
          if lattice[i] == lattice[j]:
            num_counts[r] += 1
    """
    N_ref = coords_occ.shape[0]
    N_all = coords_all.shape[0]

    denom_counts = np.zeros(r_max + 1, np.int64)
    num_counts   = np.zeros(r_max + 1, np.int64)

    for ii in range(N_ref):
        xi, yi = coords_occ[ii]
        ci = lattice[xi, yi]   # always >0

        for jj in range(N_all):
            xj, yj = coords_all[jj]
            # skip the same site
            if xi == xj and yi == yj:
                continue

            # integer‐rounded radius
            dx = xi - xj
            dy = yi - yj
            r = int(round((dx*dx + dy*dy)**0.5))
            if r > r_max:
                continue

            # increment denominator
            denom_counts[r] += 1

            # if same cluster label, increment numerator
            if lattice[xj, yj] == ci:
                num_counts[r] += 1

    # distances array 0..r_max
    rs = np.arange(r_max + 1)
    return rs, denom_counts, num_counts


if __name__ == '__main__':
    prob = 0.405
    lx = ly = 100
    my_arr = np.random.choice([0, 1], size=(ly, lx), p=[1 - prob, prob])
    my_arr = binary_fill_holes(my_arr).astype(int)
    my_arr, num_features = label(my_arr)

    dists_jit, cf_jit = old_corr_func_jit(my_arr)
    dists_jit_nopython, cf_jit_nopython = old_corr_func_jit_nopython(my_arr)

    print(cf_jit == cf_jit_nopython)

    # def run_old_jit():
    #     old_corr_func_jit(my_arr)
    #
    # def run_old_jit_nopython():
    #     old_corr_func_jit_nopython(my_arr)
    #
    # # 2) time each one with timeit.repeat
    # reps = 5
    # nloops = 3   # number of repetitions within each timing
    #
    # t_old_jit_nopython = min(timeit.repeat("run_old_jit_nopython()",
    #                setup="from __main__ import run_old_jit_nopython",
    #                repeat=reps, number=nloops)) / nloops
    #
    # t_old_jit = min(timeit.repeat("run_old_jit()",
    #                setup="from __main__ import run_old_jit",
    #                repeat=reps, number=nloops)) / nloops
    #
    #
    # # print(f"Old:     {t_old:.4f} s per call")
    # print(f"Old (jit):    {t_old_jit:.4f} s per call")
    # # print(f"New:  {t_new:.4f} s per call")
    # print(f"Old (jit, nopython): {t_old_jit_nopython:.4f} s per call")

