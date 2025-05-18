"""
Created Aug 23 2024
Updated Sep 23 2024

(IN CLUSTER)
-- Calculates the correlation function for a percolation-like 2D lattice.
-- g(r) is defined as the probability that a site a distance r from an occupied site is also occupied AND part of the same
   cluster.
-- Lattice must have each site in a cluster labeled by an integer unique to that cluster.

"""
import numpy as np
from scipy.ndimage import label
from numba import njit


def label_clusters(lattice):
    """
    :param lattice:
    :return:
    """
    labeled_lattice, num_features = label(lattice)

    return labeled_lattice


# Updated Sep 22 2024
@njit(parallel=True)
def calculate_corr_func(labeled_lattice, per):
    """
    :param labeled_lattice: (2-D array of ints or floats, required)
    :param per: (Float, optional) Fraction of sites to randomly draw for calculating g(r).
    :return: [0] Possible integer distances in the lattice; [1] Correlation function at each possible integer distance
    """
    lattice_shape = labeled_lattice.shape
    coords = np.indices(lattice_shape).reshape(len(lattice_shape), -1).T
    max_distance = int(round(np.sqrt(np.sum(np.array(lattice_shape) ** 2))))

    # top_row = labeled_lattice[0, :]
    # bottom_row = labeled_lattice[-1, :]
    # right_col = labeled_lattice[:, -1]
    # left_col = labeled_lattice[:, 0]
    #
    # top_labels = np.unique(top_row[top_row > 0])
    # bottom_labels = np.unique(bottom_row[bottom_row > 0])
    # right_labels = np.unique(right_col[right_col > 0])
    # left_labels = np.unique(left_col[left_col > 0])
    #
    # border_labels = np.unique(np.concatenate((top_labels, bottom_labels, right_labels, left_labels)))
    #
    # for i in range(lattice_shape[0]):
    #     for j in range(lattice_shape[1]):
    #         if labeled_lattice[i, j] in border_labels:
    #             labeled_lattice[i, j] = 0

    if per is None:
        pass
    else:
        new_len = int(per * len(coords))
        idxs = np.random.choice(a=np.arange(len(coords)), size=new_len, replace=False)
        new_coords = np.empty(shape=(len(idxs), coords.shape[1]), dtype=coords.dtype)

        for i in range(len(idxs)):
            new_coords[i] = coords[idxs[i]]
        coords = new_coords

    correlation_function = np.zeros(max_distance + 1)
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


def corr_func(arr, frac=None):
    """
    :param arr:
    :param frac:
    :return:
    """
    labeled_lattice = label_clusters(arr)
    distances, corr = calculate_corr_func(labeled_lattice, frac)

    return distances, corr
