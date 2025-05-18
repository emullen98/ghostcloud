"""
Created Sep 30 2024
Updated Sep 30 2024

(IN CLUSTER)
"""
import numpy as np
import numba
import scipy  # calculate the amount to multiply error by using normal distribution ppf

arr = np.array


@numba.njit(parallel=True)
def logbinning_core(unsorted_x, unsorted_y, numBins, error_type='SEM'):
    # define outputs
    centers = np.zeros(numBins)
    errs = np.zeros(numBins)
    out = np.zeros(numBins)

    unsorted_y = unsorted_y[unsorted_x > 0]  # get only positive values
    unsorted_x = unsorted_x[unsorted_x > 0]

    idxs = np.argsort(unsorted_x)

    # organize by first index
    x = unsorted_x[idxs]
    y = unsorted_y[idxs]

    logmax = np.log10(x[-1])
    logmin = np.log10(x[0])

    # get edges
    edges = np.logspace(logmin, logmax, numBins + 1)
    # get edge indices
    edgeidxs = np.zeros(numBins + 1)
    for i in range(numBins + 1):
        tmp = np.abs(x - edges[i])
        edgeidxs[
            i] = tmp.argmin()  # find minimimum from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    # get centers
    dx = (logmax - logmin) / numBins
    centers = np.logspace(logmin + dx, logmax - dx, numBins)

    # get means
    for i in range(numBins):
        st = int(edgeidxs[i])
        en = int(edgeidxs[i + 1])

        # add 1 to take into account when start and end are same index
        en = en + int(st == en)
        vals = y[st:en]
        out[i] = np.mean(vals)
        if error_type == 'SEM':
            errs[i] = np.std(vals) / np.sqrt(en - st)  # SEM = std(X)/sqrt(N). N = en-st.
        else:
            errs[i] = np.std(vals)  # standard error = std(X)

    return centers, out, errs


def logbinning(unsorted_x, unsorted_y, numBins, error_type='SEM', ci=0.68):
    centers, out, errs = logbinning_core(unsorted_x, unsorted_y, numBins, error_type=error_type)

    z = np.sqrt(2) * scipy.stats.norm.ppf((1 + ci) / 2)
    return centers, out, errs * z
