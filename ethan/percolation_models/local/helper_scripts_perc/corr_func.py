import time

import numpy as np
from scipy.ndimage import label
from numba import njit, prange


def label_clusters(lattice):
    labeled_lattice, num_features = label(lattice)
    return labeled_lattice


@njit(parallel=True)
def calculate_correlation_function_numba(labeled_lattice):
    coords = np.column_stack(np.nonzero(labeled_lattice))
    max_distance = int(np.sqrt(np.sum(np.array(labeled_lattice.shape) ** 2)))
    correlation_function = np.zeros(max_distance + 1)
    counts = np.zeros(max_distance + 1)

    for i in prange(len(coords)):
        for j in range(i + 1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            r_squared = dx * dx + dy * dy
            r = int(np.sqrt(r_squared))
            if labeled_lattice[coords[i][0], coords[i][1]] == labeled_lattice[coords[j][0], coords[j][1]]:
                correlation_function[r] += 1
            counts[r] += 1

    for r in range(max_distance + 1):
        if counts[r] > 0:
            correlation_function[r] /= counts[r]

    possible_distances = np.arange(0, max_distance + 1)

    return possible_distances, correlation_function


def corr_func_2(arr=None):
    labeled_lattice = label_clusters(arr)
    distances, corr = calculate_correlation_function_numba(labeled_lattice)
    return distances, corr
