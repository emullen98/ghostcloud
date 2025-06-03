"""
Created May 31 2025
Updated Jun 02 2025

(IN CLUSTER)
General utility functions for clouds project
"""
import os
import sys
import numpy as np
from numba import set_num_threads, njit, prange
from numba.typed import List
import scipy
from scipy.ndimage import find_objects


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
These functions should NOT be called directly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def _build_offset_distance(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute (dx, dy) offsets & their rounded Euclidean distances to be checked from each lattice point

    Main loop ranges over slightly more values than are truly necessary, but this is just a computational restriction

    Parameters
    ----------
    n_rows : int
    n_cols : int

    Returns
    -------
    np.ndarray
        Offsets from each lattice coordinate to be checked
    np.ndarray
        Euclidean distances corresponding to offsets
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

    return np.array(offsets, dtype=np.int64), np.array(distances, dtype=np.int64)


@njit(parallel=True)
def _corr_func_core(grid: np.ndarray, offsets: np.ndarray, distances: np.ndarray, unique_r: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pair connectivity (correlation) function g(r) for a single cluster

    Parameters
    ----------
    grid : np.ndarray
        Should contain only one cluster
    offsets : np.ndarray
        Nx2 array of (dx, dy) offsets
    distances : np.ndarray
        length-N array of rounded Euclidean distances corresponding to each offset
    unique_r : np.ndarray
        Sorted 1D array of unique distances to evaluate
    frac : float
        Percentage of sites to sample from

    Returns
    -------
    occ_pairs_result : np.ndarray
        Sorted array of # of occupied pairs at each Euclidean distance, starting from trivial case r = 0
    tot_pairs_result : np.ndarray
        Sorted array of # of total pairs at each Euclidean distance
    """
    n_r = unique_r.shape[0]  # number of possible euclidean distances between points
    occ_pairs_result = np.zeros(n_r, dtype=np.float64)
    tot_pairs_result = np.zeros(n_r, dtype=np.float64)

    occ = List()  # Occupied sites as a list of tuples: [(i1, j1), (i2, j2), ...]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                occ.append((i, j))
    n_occ = len(occ)
    if n_occ == 0:
        return occ_pairs_result, tot_pairs_result

    rand_idxs = np.random.choice(a=n_occ, replace=False, size=int(frac * n_occ))

    for ir in prange(n_r):  # Loop over distances in parallel
        r = unique_r[ir]

        count_offsets = 0  # Count how many offsets correspond to this r; for example, for a euclidean distance of 1 there are 8 offsets that yield this distance
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

        occ_pairs = 0
        for p in rand_idxs:
            x, y = occ[p]
            for k in range(offsets.shape[0]):
                if distances[k] == r:
                    dx, dy = offsets[k]
                    tx, ty = int(x + dx), int(y + dy)
                    if 0 <= tx < grid.shape[0] and 0 <= ty < grid.shape[1]:
                        occ_pairs += grid[tx, ty]
        occ_pairs_result[ir] = occ_pairs
        tot_pairs_result[ir] = tot_pairs

    return occ_pairs_result, tot_pairs_result


@njit()
def _get_pa_core_jit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function; see get_perimeters_areas() for description

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    perims : np.ndarray
    areas : np.ndarray
    """
    lx, ly = arr.shape

    cnum = arr.max()  # Each cluster is labeled uniquely, so the largest label is the number of unique clusters.

    areas = np.zeros(cnum)
    perims = np.zeros(cnum)

    for i in range(lx):
        for j in range(ly):
            # If we are at cluster label 5, then we want to add the area and perimeter of this cluster to the 5th entry in the areas/perims lists, which is index 4.
            idx = arr[i, j] - 1
            if idx >= 0:
                # If we are at a border site, set the area and perimeter for this cluster to a nan.
                if i == 0 or j == 0 or i == lx - 1 or j == ly - 1:
                    areas[idx] = np.nan
                    perims[idx] = np.nan
                else:
                    # If we are at an interior site, get the area and perimeter contribution to
                    # this site's cluster.
                    if (not np.isnan(areas[idx])) and (not np.isnan(perims[idx])):
                        areas[idx] = areas[idx] + 1
                        perims[idx] = perims[idx] + int(arr[i + 1, j] == 0) + int(arr[i - 1, j] == 0) + int(arr[i, j + 1] == 0) + int(arr[i, j - 1] == 0)

    return perims, areas


def _get_pa_core(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function; see get_perimeters_areas() for description

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    perims : np.ndarray
    areas : np.ndarray
    """
    lx, ly = arr.shape

    cnum = arr.max()  # Each cluster is labeled uniquely, so the largest label is the number of unique clusters.

    areas = np.zeros(cnum)
    perims = np.zeros(cnum)

    for i in range(lx):
        for j in range(ly):
            # If we are at cluster label 5, then we want to add the area and perimeter of this cluster to the 5th entry in the areas/perims lists, which is index 4.
            idx = arr[i, j] - 1
            if idx >= 0:
                # If we are at a border site, set the area and perimeter for this cluster to a nan.
                if i == 0 or j == 0 or i == lx - 1 or j == ly - 1:
                    areas[idx] = np.nan
                    perims[idx] = np.nan
                else:
                    # If we are at an interior site, get the area and perimeter contribution to
                    # this site's cluster.
                    if (not np.isnan(areas[idx])) and (not np.isnan(perims[idx])):
                        areas[idx] = areas[idx] + 1
                        perims[idx] = perims[idx] + int(arr[i + 1, j] == 0) + int(arr[i - 1, j] == 0) + int(arr[i, j + 1] == 0) + int(arr[i, j - 1] == 0)

    return perims, areas


@njit(parallel=True)
def _logbinning_core(unsorted_x: np.ndarray, unsorted_y: np.ndarray, num_bins: int, error_type: str = 'SEM') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Define outputs
    centers = np.zeros(num_bins)
    errs = np.zeros(num_bins)
    out = np.zeros(num_bins)

    unsorted_y = unsorted_y[unsorted_x > 0]
    unsorted_x = unsorted_x[unsorted_x > 0]

    idxs = np.argsort(unsorted_x)

    # Organize by first index
    x = unsorted_x[idxs]
    y = unsorted_y[idxs]

    logmax = np.log10(x[-1])
    logmin = np.log10(x[0])

    edges = np.logspace(logmin, logmax, num_bins + 1)
    edgeidxs = np.zeros(num_bins + 1)

    for i in range(num_bins + 1):
        tmp = np.abs(x - edges[i])
        # Find minimimum from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        edgeidxs[i] = tmp.argmin()

    # Get centers
    dx = (logmax-logmin)/num_bins
    centers = np.logspace(logmin + dx, logmax - dx, num_bins)

    # Get means
    for i in range(num_bins):
        st = int(edgeidxs[i])
        en = int(edgeidxs[i + 1])
        
        # Add 1 to take into account when start and end are same index
        en = en + int(st == en)
        vals = y[st:en]
        out[i] = np.mean(vals)
        if error_type == 'SEM':
            # SEM = std(X) / sqrt(N). N = en - st.
            errs[i] = np.std(vals) / np.sqrt(en - st)
        else:
            # Standard error = std(X)
            errs[i] = np.std(vals)
        
    return centers, out, errs


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
These functions SHOULD be called directly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def get_pwd() -> str:
    """
    Gets the absolute path to the currently running script
    
    See http://www.faqs.org/docs/diveintopython/regression_path.html for more

    Returns
    -------
    Absolute path (see above description)
    """
    return os.path.abspath(os.path.dirname(sys.argv[0]))


def set_thread_count(threads: int) -> None:
    """
    Tells Numba how many threads to use

    Parameters
    ----------
    threads : int
    """
    set_num_threads(threads)


def get_corr_func(processed_lattice: np.ndarray, num_features: int, max_dist: int, frac: float):
    """
    Wrapper function that takes in a preprocessed lattice and computes its correlation function g(r)

    Parameters
    ----------
    processed_lattice : np.ndarray
        Fully processed lattice
    num_features : int
        Number of clouds
    max_dist : int
        Maximum Euclidean distance between points in the lattice
        This is just the diagonal length rounded to the nearest integer
    frac : float
        Percentage of sites to visit for calculating the correlation function

    Returns
    -------
    corr_func : np.ndarray
        Pair-connectivity function g(r) starting from r = 0 and up to r = max_dist
    """
    slices = find_objects(processed_lattice)

    occ_count = np.zeros(max_dist + 1)
    tot_count = np.zeros(max_dist + 1)
    for i in range(num_features):
        cluster = processed_lattice[slices[i]]
        cluster = np.where(cluster == i + 1, 1, 0)
        if cluster.size == 1:
            continue

        offsets, distances = _build_offset_distance(cluster.shape[0], cluster.shape[1])
        unique_r = np.unique(distances)

        occ_count_temp, tot_count_temp = _corr_func_core(cluster, offsets, distances, unique_r, frac)
        occ_count_temp = np.pad(occ_count_temp, (1, len(occ_count) - (len(occ_count_temp) + 1)), mode='constant', constant_values=(0, 0))
        tot_count_temp = np.pad(tot_count_temp, (1, len(tot_count) - (len(tot_count_temp) + 1)), mode='constant', constant_values=(0, 0))

        occ_count += occ_count_temp
        tot_count += tot_count_temp

    corr_func = np.divide(occ_count, tot_count, out=np.zeros_like(occ_count, dtype=float), where=tot_count != 0)
    corr_func[0] = 1

    return corr_func


def get_perimeters_areas(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function that returns ordered 1-D Numpy arrays of perimeters & areas of clouds in a given image

    When the average linear size is 10**4, the speed up is ~30x with numba applied

    On my machine, using the jit'd version, it takes ~1 second for a 10**5 x 10**5 lattice

    Parameters
    ----------
    arr : np.ndarray
        Cloud image (as an array of integers) to get perimeters & areas for

    Returns
    -------
    perims : np.ndarray
        Array of cloud perimeters
    areas : np.ndarray
        Array of cloud areas
    """
    # Any dtype like np.int8, np.int16, np.int32, np.int64 is fine
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("Input array must contain integers only!")
        
    lx, ly = arr.shape

    if lx * ly > 10**8:
        perims, areas = _get_pa_core_jit(arr=arr)
    else:
        perims, areas = _get_pa_core(arr=arr)

    return perims, areas


def find_nearest_logbin(area_bins: list | np.ndarray, perim_bins: list | np.ndarray, amin: int) -> tuple[int, int]:
    """
    Finds the closest logbinned point in perimeter vs area to some given minimum area.

    Parameters
    ----------
    area_bins : list or np.ndarray
        Logarithmic area bins
    perim_bins : list or np.ndarray
        Logarithmic perimeter bins
    amin : int
        Minimum area to search for logbin it's closest to

    Returns
    -------
    closest_perim_bin : int
        Logarithmic perimeter bin value that lines up with closest_area_bin (see below)
    closest_area_bin : int
        Logarithmic area bin value that's closest to amin
    """
    if amin < 0 or min(area_bins) < 0 or min(perim_bins) < 0:
        raise ValueError('Either a negative amin was passed, or there is a negative area/perimeter bin.')

    closest_area_bin = min(area_bins, key=lambda x: abs(x - amin))
    idxs = np.where(area_bins == closest_area_bin)[0][0]
    closest_perim_bin = perim_bins[idxs]

    return closest_perim_bin, closest_area_bin


def linemaker(slope: float = None, intercept: list = None, xmin: float = None, xmax: float = None, ppd: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns X and Y arrays of a power-law line

    Parameters
    ----------
    slope : float, default=None
        Power law PDF slope
    intercept : list, default=None
        Intercept of the line
        Formatted as [x-val, y-val]
    xmin : float, default=None
        Minimum x-value the line will appear over
    xmax : float, default=None
        Maximum x-value the line will appear over
    ppd : int, default=40
        Number of log-spaced points per decade to evaluate the line at

    Returns
    -------
    x_vals : np.ndarray 
        X-values of the line
    y_vals : np.ndarray 
        Y-values of the line
    """
    if slope is None or intercept is None or xmin is None or xmax is None:
        raise ValueError('Please enter slope, intercept, xmin and xmax.')

    log_x_intercept, log_y_intercept = np.log10(intercept[0]), np.log10(intercept[1])
    log_xmin, log_xmax = np.log10(xmin), np.log10(xmax)

    log_b = log_y_intercept - slope * log_x_intercept  # Calculate the y-intercept of the line on log axes

    x_vals = np.logspace(log_xmin, log_xmax, round(ppd * (log_xmax - log_xmin)))  # Get the x- and y-values of the line as arrays
    y_vals = (10 ** log_b) * (x_vals ** slope)

    return x_vals, y_vals


def logbinning(unsorted_x: np.ndarray, unsorted_y: np.ndarray, num_bins: int, error_type: str = 'SEM', ci: float = 0.68) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Logarithmically bin input data
    
    Generally, prefer to use standard error of the mean (SEM) because the error bar should describe how much the average value would change when binning data together on a logarithmic bin
    
    Report 95% CI using Z-scores
    
    Parameters
    ----------
    unsorted_x : np.ndarray
        X-values of data
    unsorted_y : np.ndarray
        Y-values of data
    num_bins : int
        Number of logarithmic bins to place data into
    error_type : str, default='SEM'
        How to compute the errors in each bin
        Options:
            'SEM' : Standard error of the mean
            {str} : Any other string uses standard deviation
    ci : float, default=0.68
        Confidence interval as a fraction of 100
        For example, a 68% confidence interval is 0.68

    Returns
    -------
    centers : np.ndarray
        Bin X-values
    out : np.ndarray
        Bin Y-values
    errs : np.ndarray
        Errors on bin Y-values multiplied by Z-score
    """    
    centers, out, errs = _logbinning_core(unsorted_x, unsorted_y, num_bins, error_type=error_type)
    z = np.sqrt(2) * scipy.stats.norm.ppf((1 + ci) / 2)

    return centers, out, errs * z
