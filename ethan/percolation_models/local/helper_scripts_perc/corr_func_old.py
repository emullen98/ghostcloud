"""
Created Aug 14 2024
Updated Aug 14 2024
"""
import numpy as np
from numba import njit, prange
from itertools import combinations_with_replacement
from scipy.ndimage import label
import matplotlib.pyplot as plt

"""
Old CF stuff
"""


@njit()
def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return low


@njit(parallel=True)
def corr_func_calc(array=None, possible_distances=None):
    """
    Calculates the correlation function for "array" at each of "possible_distances."
    :param array: lattice as a numpy array.
    :param possible_distances: possible distances as an array.
    :return: correlation function for the array.
    """
    rows, cols = array.shape
    num_distances = len(possible_distances)
    pair_counts = np.zeros(num_distances, dtype=np.int64)
    same_cluster_counts = np.zeros(num_distances, dtype=np.int64)

    for i1 in prange(rows):
        for j1 in prange(cols):
            for i2 in range(i1, rows):
                for j2 in range(j1 if i1 == i2 else 0, cols):
                    if (i1, j1) != (i2, j2):
                        distance = np.round(np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2), 5)
                        bin_index = binary_search(possible_distances, distance)
                        pair_counts[bin_index] += 1
                        if array[i1, j1] == array[i2, j2] and array[i1, j1] != 0:
                            same_cluster_counts[bin_index] += 1

    correlation = np.zeros(num_distances, dtype=np.float64)
    for k in range(num_distances):
        correlation[k] = same_cluster_counts[k] / pair_counts[k]

    return correlation


def corr_func_old(arr=None):
    """
    Wrapper function that calculates the possible distances and feeds them into the correlation function calculation.
    :param arr: lattice as a numpy array.
    :return: [0] = unique distances in the provided array as a numpy array, [1] = correlation function as a numpy array.
    It is ordered and corresponds to the unique_distances array. Correlation function here is defined as the probability
    that two sites a distance r apart are part of the same cluster.
    """
    if arr is None:
        print('Please input an array!')
        return 'temp'

    size = arr.shape[0]

    combinations = list(combinations_with_replacement(np.arange(size), 2))[1:]
    unique_distances = np.unique([np.sqrt(x ** 2 + y ** 2) for x, y in combinations])
    unique_distances = np.round(unique_distances, 5)

    corr = corr_func_calc(array=arr, possible_distances=unique_distances)

    return unique_distances, corr


"""
New CF stuff
"""


def label_clusters(lattice):
    labeled_lattice, num_features = label(lattice)
    return labeled_lattice


# Updated Sep 11 2024
@njit(parallel=True)
def calculate_corr_func(labeled_lattice):
    """
    :param labeled_lattice:
    :return:
    """
    # Get the coordinates of nonzero entries in the labeled lattice.
    # Each row is a coordinate of one nonzero entry (i.e., np.array([0, 0]))
    coords = np.column_stack(np.nonzero(labeled_lattice))
    # Computes the diagonal distance from one corner of the lattice to the opposite corner
    max_distance = int(np.sqrt(np.sum(np.array(labeled_lattice.shape) ** 2)))

    correlation_function = np.zeros(max_distance + 1)
    counts = np.zeros(max_distance + 1)

    for i in prange(len(coords)):
        for j in range(i + 1, len(coords)):
            r = int(np.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2))
            if labeled_lattice[coords[i][0], coords[i][1]] == labeled_lattice[coords[j][0], coords[j][1]]:
                correlation_function[r] += 1
            counts[r] += 1

    for r in range(max_distance + 1):
        if counts[r] > 0:
            correlation_function[r] /= counts[r]

    possible_distances = np.arange(0, max_distance + 1)

    return possible_distances, correlation_function


def corr_func_new(arr=None):
    labeled_lattice = label_clusters(arr)
    distances, corr = calculate_corr_func(labeled_lattice)
    return distances, corr


test_arr = np.random.choice(a=[0, 1], size=(100, 100))
old_dists, old_cf = corr_func_old(test_arr)
new_dists, new_cf = corr_func_new(test_arr)

print('Old possible distances: ', min(old_dists), max(old_dists))
print('New possible distances: ', min(new_dists), max(new_dists))
