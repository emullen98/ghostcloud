"""
Created May 16 2025
Updated May 22 2025

Contains utility functions for analyzing high-res PNG images of clouds.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label, find_objects
from numba import njit, prange, set_num_threads
from numba.typed import List


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
These functions should NOT be called directly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def _image_array_to_binary_array(input_arr: np.ndarray, thresh: int) -> np.ndarray:
    """
    Converts a PNG cloud image array into a grayscale & binary array

    Grayscale values range from 0-255

    Parameters
    ----------
    input_arr : np.ndarray
        A PNG image that's been converted to an array
    thresh : int
        Grayscale threshold above which to mark a pixel with a 1

    Returns
    -------
    binary_image : np.ndarray
        A PNG image that has been converted to a binary array where 1s represent pixels above threshold in grayscale value
    """
    binary_image = (input_arr > thresh).astype(int)  # Apply the threshold

    return binary_image


def _load_image_as_array(path: str) -> np.ndarray:
    """
    Takes in a FULL path + file name string and converts it to a grayscale array

    Parameters
    ----------
    path : str
        File name plus its path (i.e., /path/to/file/filename.png)

    Returns
    -------
    image_array : np.ndarray
        Grayscale array corresponding to path
    """
    image = Image.open(path).convert('L')  # Ensure it's in grayscale
    image_array = np.array(image)[267:2933, :]  # Remove the white strips on top & bottom

    return image_array


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

    return (np.array(offsets, dtype=np.int64),
            np.array(distances, dtype=np.int64))


@njit(parallel=True)
def _corr_func_core(grid: np.ndarray, offsets: np.ndarray, distances: np.ndarray, unique_r: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pair connectivity (correlation) function g(r) for a single cloud

    Parameters
    ----------
    grid : np.ndarray
        Input cloud array
        Should contain only one cloud
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


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
These functions should be called directly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def set_thread_count(threads: int) -> None:
    """
    Tells Numba how many threads to use

    Parameters
    ----------
    threads : int
    """
    set_num_threads(threads)


def fill_and_label_image(path: str, thresh: int, rem_border_clouds: bool) -> tuple[np.ndarray, int]:
    """
    Fills and labels an input image

    Parameters
    ----------
    path : str
        Input image file name + path (i.e., '/path/to/image/file_name.png')
    thresh : int
        Grayscale threshold
    rem_border_clouds : bool
        Whether to remove clouds that touch the boundary of the image

    Returns
    -------
    labeled_filled_image : np.ndarray of ints
        Labeled & filled version of image array
    num_features : int
        Number of clusters after filling & labeling the image
    """
    if not 0 <= thresh <= 255:
        raise ValueError(f'Invalid \'thresh\' entry {thresh}. Must be between 0-255 inclusive.')

    my_arr = _load_image_as_array(path=path)
    my_arr = _image_array_to_binary_array(input_arr=my_arr, thresh=thresh)
    my_arr = binary_fill_holes(my_arr)

    if rem_border_clouds:
        labeled_filled_image, _ = label(my_arr)  # Number of features needs to be calculated AFTER removing boundary-touching clusters, if applicable

        top_row = labeled_filled_image[0, :]
        bot_row = labeled_filled_image[-1, :]
        left_col = labeled_filled_image[:, 0]
        right_col = labeled_filled_image[:, -1]

        unique_border_labels = np.unique(np.concatenate((top_row, bot_row, left_col, right_col)))
        unique_border_labels = unique_border_labels[unique_border_labels > 0]

        mask = np.isin(labeled_filled_image, unique_border_labels)
        labeled_filled_image[mask] = 0

        labeled_filled_image = (labeled_filled_image > 0).astype(int)
        labeled_filled_image, num_features = label(labeled_filled_image)
    else:
        labeled_filled_image, num_features = label(my_arr)

    return labeled_filled_image, num_features


def label_image(path: str, thresh: int, rem_border_clouds: bool) -> tuple[np.ndarray, int]:
    """
    Labels (but does not fill) an input image

    Parameters
    ----------
    path : str
        Input image file name + path (i.e., '/path/to/image/file_name.png')
    thresh : int
        Grayscale threshold
    rem_border_clouds : bool
        Whether to remove clouds that touch the boundary of the image

    Returns
    -------
    labeled_binary_image : np.ndarray
        Labeled version of image array
    num_features : int
        Number of clusters after labeling the image
    """
    if not 0 <= thresh <= 255:
        raise ValueError(f'Invalid \'thresh\' entry {thresh}. Must be between 0-255 inclusive.')

    my_arr = _load_image_as_array(path=path)
    my_arr = _image_array_to_binary_array(input_arr=my_arr, thresh=thresh)

    if rem_border_clouds:
        labeled_image, _ = label(my_arr)

        top_row = labeled_image[0, :]
        bot_row = labeled_image[-1, :]
        left_col = labeled_image[:, 0]
        right_col = labeled_image[:, -1]

        unique_border_labels = np.unique(np.concatenate((top_row, bot_row, left_col, right_col)))
        unique_border_labels = unique_border_labels[unique_border_labels > 0]

        mask = np.isin(labeled_image, unique_border_labels)
        labeled_image[mask] = 0

        labeled_image = (labeled_image > 0).astype(int)
        labeled_image, num_features = label(labeled_image)
    else:
        labeled_image, num_features = label(my_arr)

    return labeled_image, num_features


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


@njit()
def get_perimeters_areas(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets lists of perimeters & areas of clouds in a given image.

    When the linear array size is increased to 10**4, the speed up is ~30x with numba applied.

    On my machine, just njit'ing this function makes it take ~1 second for a 10**5 x 10**5 lattice.

    Parameters
    ----------
    arr : np.ndarray
        Cloud image (as an array) to get perimeters & areas for

    Returns
    -------
    perims : np.ndarray
        Array of cloud perimeters
    areas : np.ndarray
        Array of cloud areas
    """
    lx, ly = arr.shape

    # Each cluster is labeled uniquely, so the largest label is the number of unique clusters.
    cnum = arr.max()

    areas = np.zeros(cnum)
    perims = np.zeros(cnum)

    for i in range(lx):
        for j in range(ly):
            # If we are at cluster label 5, then we want to add the area and perimeter of this
            # cluster to the 5th entry in the areas/perims lists, which is index 4.
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


def get_corr_func(processed_lattice: np.ndarray, num_features: int, max_dist: int, frac: float):
    """
    Wrapper function that takes in a preprocessed lattice and computes its correlation function g(r)

    Parameters
    ----------
    processed_lattice : np.ndarray
        Full lattice of clouds
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
        cloud = processed_lattice[slices[i]]
        cloud = np.where(cloud == i + 1, 1, 0)
        if cloud.size == 1:
            continue

        offsets, distances = _build_offset_distance(cloud.shape[0], cloud.shape[1])
        unique_r = np.unique(distances)

        occ_count_temp, tot_count_temp = _corr_func_core(cloud, offsets, distances, unique_r, frac)
        occ_count_temp = np.pad(occ_count_temp, (1, len(occ_count) - (len(occ_count_temp) + 1)), mode='constant', constant_values=(0, 0))
        tot_count_temp = np.pad(tot_count_temp, (1, len(tot_count) - (len(tot_count_temp) + 1)), mode='constant', constant_values=(0, 0))

        occ_count += occ_count_temp
        tot_count += tot_count_temp

    corr_func = np.divide(occ_count, tot_count, out=np.zeros_like(occ_count, dtype=float), where=tot_count != 0)
    corr_func[0] = 1

    return corr_func
