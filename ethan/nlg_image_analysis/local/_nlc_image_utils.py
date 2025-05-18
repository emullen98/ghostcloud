"""
Created May 16 2025
Updated May 18 2025

Contains utility functions for analyzing high-res PNG images of clouds.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label
from numba import njit


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


def fill_and_label_image(path: str, thresh: int) -> tuple[np.ndarray, int]:
    """
    Fills and labels an input image

    Parameters
    ----------
    path : str
        Input image file name + path (i.e., '/path/to/image/file_name.png')
    thresh : int
        Grayscale threshold

    Returns
    -------
    labeled_filled_binary_image : np.ndarray
        Labeled & filled version of image array
    num_features : int
        Number of clusters after filling & labeling the image
    """
    if not 0 <= thresh <= 255:
        raise ValueError(f'Invalid \'thresh\' entry {thresh}. Must be between 0-255 inclusive.')

    my_arr = _load_image_as_array(path=path)
    my_arr = _image_array_to_binary_array(input_arr=my_arr, thresh=thresh)
    my_arr = binary_fill_holes(my_arr)
    labeled_filled_binary_image, num_features = label(my_arr)

    return labeled_filled_binary_image, num_features


def label_image(path: str, thresh: int) -> tuple[np.ndarray, int]:
    """
    Labels (but does not fill) an input image

    Parameters
    ----------
    path : str
        Input image file name + path (i.e., '/path/to/image/file_name.png')
    thresh : int
        Grayscale threshold

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
    labeled_binary_image, num_features = label(my_arr)

    return labeled_binary_image, num_features


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


def get_perimeters_areas(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets lists of perimeters & areas of clouds in a given image.

    When the linear array size is increased to 10**4, the speed up is ~30x with numba applied.

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


@njit(parallel=True)
def corr_func(labeled_lattice: np.ndarray, frac: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the correlation (or pair-connectivity) function for a given LABELED lattice

    Pair connectivity function is defined here as in the sense of percolation (see Kim Christensen's introductory notes on percolation)

    Parameters
    ----------
    labeled_lattice : np.ndarray
        Labeled lattice to compute correlation function of
    frac : float, default=None
        Fraction of sites to use for computing the correlation function

    Returns
    -------
    possible_distances : np.ndarray
        Array of possible integer Euclidean distances in the lattice
    correlation_function : np.ndarray
        Array of correlation function values
    """
    lattice_shape = labeled_lattice.shape
    coords = np.indices(lattice_shape).reshape(len(lattice_shape), -1).T
    max_distance = int(round(np.sqrt(np.sum(np.array(lattice_shape) ** 2))))

    if frac is None:
        pass
    else:
        idxs = np.random.choice(a=np.arange(len(coords)), size=int(frac * len(coords)), replace=False)
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
            inner_val = labeled_lattice[inner_coord[0], inner_coord[1]]
            outer_val = labeled_lattice[outer_coord[0], outer_coord[1]]
            if inner_val == 0 and outer_val == 0:
                pass
            elif (inner_val == 0 and outer_val != 0) or (inner_val != 0 and outer_val == 0):
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 1
            elif (inner_val != 0 and outer_val != 0) and inner_val != outer_val:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 2
            elif (inner_val != 0 and outer_val != 0) and inner_val == outer_val:
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
