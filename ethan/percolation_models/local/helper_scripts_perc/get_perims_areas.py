"""
Created Aug 14 2024
Updated Apr 10 2025
"""
import numpy as np


def get_perims_areas(arr):
    """
    -- Gets cluster perimeters & areas for a lattice where clusters are labeled uniquely by nonzero integers.
    -- I have checked this function's speed using random numpy arrays of 0s and 1s.
        -- For an array of linear size 10**3, using numba is disadvantageous.
        -- However, when the linear size is increased to 10**4, the speed up is ~30x with numba applied.

    Parameters
    ----------
    arr : arr
        Lattice to find cluster perimeters & areas for

    Returns
    -------
    [0] perims : arr
        cluster perimeters as a numpy array
    [1] areas : arr
        cluster areas as a numpy array
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
