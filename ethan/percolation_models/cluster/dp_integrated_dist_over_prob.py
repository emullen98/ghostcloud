"""
Created Jun 10 2024
Updated Jun 14 2024

Show the CCDFs of the integrated distributions over different log-spaced bond probabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
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


do_belowpc = True
do_abovepc = False

prob_list_belowpc = np.round(np.logspace(np.log10(0.28), np.log10(0.38), 20), 5)
prob_list_abovepc = np.round(np.logspace(np.log10(0.382), np.log10(0.48), 20), 5)

folder_list = os.listdir(f'./lattices/s50000')

perim_fig, (perim_ax_belowpc, perim_ax_abovepc) = plt.subplots(1, 2, figsize=(10, 6))
area_fig, (area_ax_belowpc, area_ax_abovepc) = plt.subplots(1, 2, figsize=(10, 6))

if do_belowpc:
    all_perims, all_areas = {}, {}
    for prob in prob_list_belowpc:
        found_it = ''
        for file_name in folder_list:
            found_it_temp = re.findall(f'lattice_pa_s=50000_p={prob}_end=7_job=11673316.*.npy', file_name)
            if len(found_it_temp) > 0:
                found_it = found_it_temp[0]
        data = np.load(f'./lattices/s50000/{found_it}')
        perims, areas = data[0], data[1]

        areas = areas[~np.isnan(areas)]
        area_vals, area_counts = np.unique(areas, return_counts=True)
        area_unique_dict = dict(zip(area_vals, area_counts))
        for area, count in area_unique_dict.items():
            if area in all_areas:
                all_areas[area] += count
            else:
                all_areas[area] = count
        area_histx, area_histy = ccdf(areas)
        area_ax_belowpc.plot(area_histx, area_histy, label=f'Bond prob. = {prob}', alpha=0.3)

        perims = perims[~np.isnan(perims)]
        perim_vals, perim_counts = np.unique(perims, return_counts=True)
        perim_unique_dict = dict(zip(perim_vals, perim_counts))
        for perim, count in perim_unique_dict.items():
            if perim in all_perims:
                all_perims[perim] += count
            else:
                all_perims[perim] = count
        perim_histx, perim_histy = ccdf(perims)
        perim_ax_belowpc.plot(perim_histx, perim_histy, label=f'Bond prob = {prob}', alpha=0.3)

    # x_perim, y_perim = linemaker(-2.1, [2*10**2, 10**(-2)], 20, 400)
    all_perim_histx, all_perim_histy = ccdf2(list(all_perims.keys()), list(all_perims.values()))
    perim_ax_belowpc.set_title(r'Integrated perimeter CCDF below $p_c = 0.381$')
    perim_ax_belowpc.set_xlabel(r'$P$')
    perim_ax_belowpc.set_ylabel(r'$Pr(\text{perim} > P)$')
    perim_ax_belowpc.loglog(all_perim_histx, all_perim_histy, linewidth=1.5, label='Integrated')
    # perim_ax_belowpc.loglog(x_perim, y_perim, color='r', linestyle='dashed', label=f'CCDF exp -2.1')
    # perim_fig.legend(loc='upper right', fancybox=True, fontsize='small', shadow=True)
    perim_fig.tight_layout()
    perim_fig.savefig(f'integrated_perim_ccdf_s=50000_t=7_belowpc.png', dpi=200)

    # x_area, y_area = linemaker(-1.4, [10**2, 10**(-2)], 10**1, 10**3)
    all_area_histx, all_area_histy = ccdf2(list(all_areas.keys()), list(all_areas.values()))
    area_ax_belowpc.set_title(f'Integrated area CCDF below $p_c = 0.381$')
    area_ax_belowpc.set_xlabel(r'$A$')
    area_ax_belowpc.set_ylabel(r'$Pr(\text{area} > A)$')
    area_ax_belowpc.loglog(all_area_histx, all_area_histy, linewidth=1.5, label='Integrated')
    # area_ax_belowpc.loglog(x_area, y_area, color='r', linestyle='dashed', label=f'CCDF exp -1.4')
    # area_fig.legend(loc='upper right', fancybox=True, fontsize='small', shadow=True)
    area_fig.tight_layout()
    area_fig.savefig(f'integrated_area_ccdf_s=50000_t=7_belowpc.png', dpi=200)

if do_abovepc:
    all_perims, all_areas = {}, {}
    for prob in prob_list_abovepc:
        found_it = ''
        for file_name in folder_list:
            found_it_temp = re.findall(f'lattice_pa_s=50000_p={prob}_end=7_job=11673316.*.npy', file_name)
            if len(found_it_temp) > 0:
                found_it = found_it_temp[0]
        data = np.load(f'./lattices/s50000/{found_it}')
        perims, areas = data[0], data[1]

        areas = areas[~np.isnan(areas)]
        area_vals, area_counts = np.unique(areas, return_counts=True)
        area_unique_dict = dict(zip(area_vals, area_counts))
        for area, count in area_unique_dict.items():
            if area in all_areas:
                all_areas[area] += count
            else:
                all_areas[area] = count
        area_histx, area_histy = ccdf(areas)
        area_ax_abovepc.plot(area_histx, area_histy, label=f'Bond prob. = {prob}', alpha=0.3)

        perims = perims[~np.isnan(perims)]
        perim_vals, perim_counts = np.unique(perims, return_counts=True)
        perim_unique_dict = dict(zip(perim_vals, perim_counts))
        for perim, count in perim_unique_dict.items():
            if perim in all_perims:
                all_perims[perim] += count
            else:
                all_perims[perim] = count
        perim_histx, perim_histy = ccdf(perims)
        perim_ax_abovepc.plot(perim_histx, perim_histy, label=f'Bond prob = {prob}', alpha=0.3)

    # x_perim, y_perim = linemaker(-2.1, [2 * 10 ** 2, 10 ** (-2)], 20, 400)
    all_perim_histx, all_perim_histy = ccdf2(list(all_perims.keys()), list(all_perims.values()))
    perim_ax_abovepc.set_title(r'Integrated perimeter CCDF above $p_c = 0.381$')
    perim_ax_abovepc.set_xlabel(r'$P$')
    perim_ax_abovepc.set_ylabel(r'$Pr(\text{perim} > P)$')
    perim_ax_abovepc.loglog(all_perim_histx, all_perim_histy, linewidth=1.5, label='Integrated')
    # perim_ax_belowpc.loglog(x_perim, y_perim, color='r', linestyle='dashed', label=f'CCDF exp -2.1')
    # perim_fig.legend(loc='upper right', fancybox=True, fontsize='small', shadow=True)
    perim_fig.tight_layout()
    perim_fig.savefig(f'integrated_perim_ccdf_s=50000_t=7_abovepc.png', dpi=200)

    # x_area, y_area = linemaker(-1.4, [10 ** 2, 10 ** (-2)], 10 ** 1, 10 ** 3)
    all_area_histx, all_area_histy = ccdf2(list(all_areas.keys()), list(all_areas.values()))
    area_ax_abovepc.set_title(f'Integrated area CCDF above $p_c = 0.381$')
    area_ax_abovepc.set_xlabel(r'$A$')
    area_ax_abovepc.set_ylabel(r'$Pr(\text{area} > A)$')
    area_ax_abovepc.loglog(all_area_histx, all_area_histy, linewidth=1.5, label='Integrated')
    # area_ax_abovepc.loglog(x_area, y_area, color='r', linestyle='dashed', label=f'CCDF exp -1.4')
    # area_fig.legend(loc='upper right', fancybox=True, fontsize='small', shadow=True)
    area_fig.tight_layout()
    area_fig.savefig(f'integrated_area_ccdf_s=50000_t=7_abovepc.png', dpi=200)
