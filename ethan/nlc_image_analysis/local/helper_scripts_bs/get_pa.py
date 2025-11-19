"""
Created Sep 15 2024
Updated Sep 20 2024

"""
import time
import numpy as np
import numba as nb
from numba import prange


# Updated Sep 20 2024
@nb.njit(parallel=True)
def get_pa(arr):
    """
    For lattice of the size of a usual PNG file (2500^2 ish), using numba with parallel = True and prange for the outer
    loop takes about 0.9 seconds with compilation and 0.005 seconds to actually execute, while removing numba makes this
    take 2 seconds for each
    :param arr:
    :return:
    """
    lx, ly = arr.shape

    # Each cluster is labeled uniquely, so the largest label is the number of unique clusters.
    cnum = arr.max()

    areas = np.zeros(cnum)
    perims = np.zeros(cnum)

    for i in prange(lx):
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
