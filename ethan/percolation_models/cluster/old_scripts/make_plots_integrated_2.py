"""
Created Jun 11 2024
Updated Jun 11 2024

Show the integrated CCDF over bond probabilities and *fit to it.
* Fitting didn't look like it was working too well (possible due to too small of a fitting range).
* Currently shows the fit results from the integrated distribution over timesteps overlayed on top.
* The fitting code is in comments at the bottom.
"""

# import powerlaw as pl
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import os
sys.path.append('/home/emullen2/scratch/DirectedPercolation/')
from helper_scripts.linemaker import linemaker


def ccdf(data, method='scipy'):
    """
    :param data: Input data. TYPE: list or array.
    :param method: Choice between representing CCDF as P(X > x) ('scipy') or P(X >= x) ('dahmen'). TYPE: str.
    :return:
    [0] histx = X-values in CCDF
    [1] histy = Y-values in CCDF
    """
    data = np.array(data)
    if len(data) == 0:
        print('Data array is empty')
        return np.array([]), np.array([])

    if method != 'scipy' and method != 'dahmen':
        print('Please choose between two methods: \'scipy\' or \'dahmen\'')
        return np.array([]), np.array([])

    # Take only positive values, non-NaNs, and non-Infs
    data = data[(data > 0) * ~np.isnan(data) * ~np.isinf(data)]

    # Get the unique values and their counts
    vals, counts = np.unique(data, return_counts=True)
    # Sort both the values and their counts the same way
    histx = vals[np.argsort(vals)]
    counts = counts[np.argsort(vals)]

    # P(X > x)
    if method == 'scipy':
        histx = np.insert(histx, 0, 0)

        # Get cumulative counts for the unique points
        cum_counts = np.cumsum(counts)

        # Get the total number of events
        total_count = cum_counts[-1]

        # Start constructing histy by saying that 100% of the data should be greater than 0
        histy = np.ones(len(counts) + 1)
        histy[1:] = 1 - (cum_counts / total_count)

    # P(X >= x)
    elif method == 'dahmen':
        cum_counts = np.cumsum(counts)
        # Now we insert a 0 at the beginning of cum_counts.
        # Since Pr(X >= x) = 1 - Pr(X < x), we can get the second term from this newly expanded cum_counts
        cum_counts = np.insert(cum_counts, 0, 0)

        total_counts = cum_counts[-1]

        histy = (1 - (cum_counts / total_counts))[:-1]

    return histx, histy


def ccdf2(data_vals, data_counts, method='scipy'):
    data_vals = np.array(data_vals)
    data_counts = np.array(data_counts)
    histx = data_vals[np.argsort(data_vals)]
    counts = data_counts[np.argsort(data_vals)]

    if method == 'scipy':
        histx = np.insert(histx, 0, 0)
        cum_counts = np.cumsum(counts)
        total_count = cum_counts[-1]
        histy = np.ones(len(counts) + 1)
        histy[1:] = 1 - (cum_counts / total_count)

    elif method == 'dahmen':
        cum_counts = np.cumsum(counts)
        cum_counts = np.insert(cum_counts, 0, 0)
        total_counts = cum_counts[-1]
        histy = (1 - (cum_counts / total_counts))[:-1]

    return histx, histy


do_areas = True
do_perims = True
# Impose these here to reduce the size of "all_perims_list" and "all_areas_list"
min_perim, max_perim = 10, 2000
min_area, max_area = 10, 5000

prob_list = [0.32, 0.34, 0.35, 0.36, 0.37, 0.381, 0.384, 0.39, 0.40, 0.41, 0.42]

folder_list = os.listdir(f'./lattices/s50000')

all_perims, all_areas = {}, {}
all_perims_list, all_areas_list = [], []
for prob in prob_list:
    found_it = ''
    for file_name in folder_list:
        found_it_temp = re.findall(f'lattice_pa_s=50000_p={prob}_end=7_job=.*_task=1.npy', file_name)
        if len(found_it_temp) > 0:
            found_it = found_it_temp[0]
    data = np.load(f'./lattices/s50000/{found_it}')
    perims, areas = data[0], data[1]

    if do_areas:
        areas = areas[~np.isnan(areas)]
        area_vals, area_counts = np.unique(areas, return_counts=True)
        area_unique_dict = dict(zip(area_vals, area_counts))
        for area, count in area_unique_dict.items():
            if min_area < area < max_area:
                all_areas_list.extend([area] * count)
            if area in all_areas:
                all_areas[area] += count
            else:
                all_areas[area] = count

    if do_perims:
        perims = perims[~np.isnan(perims)]
        perim_vals, perim_counts = np.unique(perims, return_counts=True)
        perim_unique_dict = dict(zip(perim_vals, perim_counts))
        for perim, count in perim_unique_dict.items():
            if min_perim < perim < max_perim:
                all_perims_list.extend([perim] * count)
            if perim in all_perims:
                all_perims[perim] += count
            else:
                all_perims[perim] = count

if do_areas:
    all_area_histx, all_area_histy = ccdf2(list(all_areas.keys()), list(all_areas.values()))
    area_ccdf_exp = -1.76

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_title('Probability-integrated area CCDF')
    ax1.set_ylabel(r'$Pr(\text{area} > A)$')
    ax1.set_xlabel(r'$A$')
    ax1.loglog(all_area_histx, all_area_histy, color='silver')
    x1, y1 = linemaker(area_ccdf_exp, [10 ** 3, 10 ** (-3)], 100, 800)
    ax1.loglog(x1, y1, color='k', label=rf'$\text{{Time-integrated CCDF}}(A) \sim A^{{{area_ccdf_exp:.2f}}}$')
    ax1.legend(loc='upper right', fancybox=True, shadow=True)
    fig1.tight_layout()
    fig1.savefig('./integrated_area_ccdf_s=50000_t=7_fixedpc_overlay.png', dpi=200)

if do_perims:
    all_perim_histx, all_perim_histy = ccdf2(list(all_perims.keys()), list(all_perims.values()))
    perim_ccdf_exp = -2.79

    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_title('Probability-integrated perimeter CCDF')
    ax2.set_ylabel(r'$Pr(\text{perim} > P)$')
    ax2.set_xlabel(r'$P$')
    ax2.loglog(all_perim_histx, all_perim_histy, color='silver')
    x2, y2 = linemaker(perim_ccdf_exp, [4 * 10 ** 2, 10 ** (-3)], 2 * 10 ** 2, 6 * 10 ** 2)
    ax2.loglog(x2, y2, color='k', label=rf'$\text{{Time-integrated CCDF}}(P) \sim P^{{{perim_ccdf_exp:.2f}}}$')
    ax2.legend(loc='upper right', fancybox=True, shadow=True)
    fig2.tight_layout()
    fig2.savefig('./integrated_perim_ccdf_s=50000_t=7_fixedpc_overlay.png', dpi=200)


# if do_areas:
#     all_area_histx, all_area_histy = ccdf2(list(all_areas.keys()), list(all_areas.values()))
#     area_fit = pl.Fit(all_areas_list, xmin=100, xmax=800)
#     tau = area_fit.power_law.alpha
#     area_ccdf_exp = -1 * (tau - 1)
#
#     fig1, ax1 = plt.subplots(1, 1)
#     ax1.set_title('Integrated area CCDF')
#     ax1.set_ylabel(r'$Pr(\text{area} > A)$')
#     ax1.set_xlabel(r'$A$')
#     ax1.loglog(all_area_histx, all_area_histy, color='silver')
#     ax1.vlines(x=100, ymin=min(all_area_histy), ymax=1, color='r', linestyle='dashed', label=r'My pl fit $A_{min}$')
#     ax1.vlines(x=800, ymin=min(all_area_histy), ymax=1, color='blue', linestyle='dashed', label=r'My pl fit $A_{max}$')
#     x1, y1 = linemaker(area_ccdf_exp, [10 ** 3, 10 ** (-3)], 100, 800)
#     ax1.loglog(x1, y1, color='k', label=rf'$\text{{Integrated CCDF}}(A) \sim A^{{{area_ccdf_exp:.2f}}}$')
#     ax1.legend(loc='upper right', fancybox=True, shadow=True)
#     fig1.tight_layout()
#     fig1.savefig('./integrated_area_ccdf_s=50000_t=7_fit.png', dpi=200)
#
# if do_perims:
#     all_perim_histx, all_perim_histy = ccdf2(list(all_perims.keys()), list(all_perims.values()))
#     perim_fit = pl.Fit(all_perims_list, xmin=200, xmax=600)
#     alpha = perim_fit.power_law.alpha
#     perim_ccdf_exp = -1 * (alpha - 1)
#
#     fig2, ax2 = plt.subplots(1, 1)
#     ax2.set_title('Integrated perimeter CCDF')
#     ax2.set_ylabel(r'$Pr(\text{perim} > P)$')
#     ax2.set_xlabel(r'$P$')
#     ax2.loglog(all_perim_histx, all_perim_histy, color='silver')
#     ax2.vlines(x=200, ymin=min(all_perim_histy), ymax=1, color='r', linestyle='dashed', label=r'My pl fit $P_{min}$')
#     ax2.vlines(x=600, ymin=min(all_perim_histy), ymax=1, color='blue', linestyle='dashed', label=r'My pl fit $P_{max}$')
#     x2, y2 = linemaker(perim_ccdf_exp, [4 * 10 ** 2, 10 ** (-3)], 2 * 10 ** 2, 6 * 10 ** 2)
#     ax2.loglog(x2, y2, color='k', label=rf'$\text{{Integrated CCDF}}(P) \sim P^{{{perim_ccdf_exp:.2f}}}$')
#     ax2.legend(loc='upper right', fancybox=True, shadow=True)
#     fig2.tight_layout()
#     fig2.savefig('./integrated_perim_ccdf_s=50000_t=7_fit.png', dpi=200)
