"""
Created Jun 09 2024
Updated Jun 09 2024

Gets a fit of the scaling function for the CCDFs
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import moment
from scipy.optimize import curve_fit
from scipy.integrate import quad
import sys
sys.path.append('../twoPlusOne(21)D')
import common_code as cc


kappa = 2


def ccdf(data):
    data = np.array(data)
    if len(data) == 0:
        return np.array([]), np.array([])

    # Take only positive values, non-NaNs, and non-Infs
    data = data[(data > 0) * ~np.isnan(data) * ~np.isinf(data)]

    # Get the unique values and their counts
    vals, counts = np.unique(data, return_counts=True)
    # Sort both the values and their counts the same way
    histx = vals[np.argsort(vals)]
    counts = counts[np.argsort(vals)]
    histx = np.insert(histx, 0, 0)

    # Get cumulative counts for the unique points
    cum_counts = np.cumsum(counts)

    # Get the total number of events
    total_count = cum_counts[-1]

    # Start constructing histy by saying that 100% of the data should be greater than 0
    histy = np.ones(len(counts) + 1)
    histy[1:] = 1 - (cum_counts / total_count)

    return histx, histy


def integrand(t, a, b):
    return b * t ** (-1 * kappa) * np.exp(-1 * a * t)


def integral_func(x, a, b):
    integral, _ = quad(integrand, x, np.inf, args=(2, 1))
    return integral


def model_func(x, a, b):
    return np.array([integral_func(xi, a, b) for xi in x])


# pa = np.load('./lattices/s50000/lattice_pa_s=50000_p=0.381_end=8_job=11587657_task=1.npy')
pa = np.load('../data/lattice_pa_s=50000_p=0.381_end=8_job=11587657_task=1.npy')
perims, areas = pa[0], pa[1]
perims = perims[~np.isnan(perims)]
areas = areas[~np.isnan(areas)]

ahistx, ahisty = ccdf(areas)
area_second_moment = moment(areas, moment=2, nan_policy='omit')
ahistx_rescaled = ahistx / (area_second_moment ** (1 / (3 - kappa)))
ahisty_rescaled = ahisty / (area_second_moment ** ((1 - kappa) / (3 - kappa)))
ahistx_rescaled_bins, ahisty_rescaled_bins = cc.logbinning(ahistx_rescaled, ahisty_rescaled, 50)[:2]

plt.loglog(ahistx_rescaled_bins, ahisty_rescaled_bins, color='silver')

popt, pcov = curve_fit(model_func, ahistx_rescaled_bins, ahisty_rescaled_bins)
print(popt)
curve_xvals = ahistx_rescaled_bins
curve_yvals = model_func(curve_xvals, *popt)
plt.loglog(curve_xvals, curve_yvals, color='blue')

plt.show()
