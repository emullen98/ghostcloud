"""
Created xxx xx 2024
Updated Sep 28 2024

(IN CLUSTER)

"""
import numpy as np
import numba as nb


@nb.njit
def get_pa(arr):
    """
    I have checked this function's speed using random numpy arrays of 0s and 1s.
    For an array of linear size 10**3, using numba is disadvantageous.
    However, when the linear size is increased to 10**4, the speed up is ~30x with numba applied.
    :param arr:
    :return:
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
