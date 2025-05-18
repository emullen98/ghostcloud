"""
Created Jun 11 2024
Updated Jun 11 2024

Does a scaling collapse of perimeter & area CCDFs.
See Nir's thesis, p.17, for the form of the collapse
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import re


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


size = 50000
do_perim = True
do_area = True
do_beforetc = False
do_aftertc = True
# Kappas are the PDF exponents
# Sigmas are the difference between the integrated and non-integrated exponents
kappa_perim = 1.75
sigma_perim = 1.29
kappa_area = 2.0
sigma_area = 0.74

folder_list = os.listdir(f'./lattices/s{size}')

fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(10, 6))
fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(10, 6))
fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(10, 6))
fig4, (ax41, ax42) = plt.subplots(1, 2, figsize=(10, 6))

if do_beforetc:
    for t in range(3, 7):
        candidate_lattices = []
        for file_name in folder_list:
            found_it = re.findall(f".+end={t}.+", file_name)
            if len(found_it) > 0:
                candidate_lattices.append(found_it[0])
        chosen_lattice = random.choice(candidate_lattices)

        data = np.load(f'./lattices/s{size}/{chosen_lattice}')
        perims, areas = data[0], data[1]

        if do_perim:
            phistx, phisty = ccdf(list(perims))
            ax11.loglog(phistx, phisty, '.', label=f'Timestep = {t}')
            phistx_col = phistx / (np.abs((7 - t) / 7) ** (-1 / sigma_perim))
            phisty_col = phisty / (np.abs((7 - t) / 7) ** (-1 * (kappa_perim - 1) / sigma_perim))
            ax12.loglog(phistx_col, phisty_col, '.', label=f'Timestep = {t}')

        if do_area:
            ahistx, ahisty = ccdf(list(areas))
            ax21.loglog(ahistx, ahisty, '.', label=f'Timestep = {t}')
            ahistx_col = ahistx / (np.abs((7 - t) / 7) ** (-1 / sigma_area))
            ahisty_col = ahisty / (np.abs((7 - t) / 7) ** (-1 * (kappa_area - 1) / sigma_area))
            ax22.loglog(ahistx_col, ahisty_col, '.', label=f'Timestep = {t}')

    if do_perim:
        ax11.set_title(f'Perimeter CCDF uncollapsed \n System size = {size}, ' + r'$\kappa_{\text{perim}} = $' + f'{kappa_perim}')
        ax11.set_xlabel(r'$P$')
        ax11.set_ylabel(r'$C(P)$')
        ax11.legend()

        ax12.set_title(f'Perimeter CCDF collapsed \n System size = {size}, ' + r'$\kappa_{\text{perim}} = $' + f'{kappa_perim}')
        ax12.set_xlabel(r'$P / ((t_c - t) / t_c)^{-1 / \sigma_{\text{perim}}}$')
        ax12.set_ylabel(r'$C(P) / ((t_c - t) / t_c)^{-(\kappa_{\text{perim}} - 1) / \sigma_{\text{perim}}}$')
        ax12.legend()

        fig1.tight_layout()
        fig1.savefig(f'./perim_ccdf_collapse_s={size}_kappaperim={kappa_perim}_beforetc.png', dpi=200)

    if do_area:
        ax21.set_title(f'Area CCDF uncollapsed \n System size = {size}, ' + r'$\kappa_{\text{area}} = $' + f'{kappa_area}')
        ax21.set_xlabel(r'$A$')
        ax21.set_ylabel(r'$C(A)$')
        ax21.legend()

        ax22.set_title(f'Area CCDF collapsed \n System size = {size}, ' + r'$\kappa_{\text{area}} = $' + f'{kappa_area}')
        ax22.set_xlabel(r'$A / ((t_c - t) / t_c)^{-1 / \sigma_{\text{area}}}$')
        ax22.set_ylabel(r'$C(A) / ((t_c - t) / t_c)^{-(\kappa_{\text{area}} - 1) / \sigma_{\text{area}}}$')
        ax22.legend()

        fig2.tight_layout()
        fig2.savefig(f'./area_ccdf_collapse_s={size}_kappaarea={kappa_area}_beforetc.png', dpi=200)

if do_aftertc:
    for t in range(8, 14):
        candidate_lattices = []
        for file_name in folder_list:
            found_it = re.findall(f".+end={t}.+", file_name)
            if len(found_it) > 0:
                candidate_lattices.append(found_it[0])
        chosen_lattice = random.choice(candidate_lattices)

        data = np.load(f'./lattices/s{size}/{chosen_lattice}')
        perims, areas = data[0], data[1]

        if do_perim:
            phistx, phisty = ccdf(list(perims))
            ax31.loglog(phistx, phisty, '.', label=f'Timestep = {t}')
            phistx_col = phistx / (np.abs((7 - t) / 7) ** (-1 / sigma_perim))
            phisty_col = phisty / (np.abs((7 - t) / 7) ** (-1 * (kappa_perim - 1) / sigma_perim))
            ax32.loglog(phistx_col, phisty_col, '.', label=f'Timestep = {t}')

        if do_area:
            ahistx, ahisty = ccdf(list(areas))
            ax41.loglog(ahistx, ahisty, '.', label=f'Timestep = {t}')
            ahistx_col = ahistx / (np.abs((7 - t) / 7) ** (-1 / sigma_area))
            ahisty_col = ahisty / (np.abs((7 - t) / 7) ** (-1 * (kappa_area - 1) / sigma_area))
            ax42.loglog(ahistx_col, ahisty_col, '.', label=f'Timestep = {t}')

    if do_perim:
        ax31.set_title(f'Perimeter CCDF uncollapsed \n System size = {size}, ' + r'$\kappa_{\text{perim}} = $' + f'{kappa_perim}')
        ax31.set_xlabel(r'$P$')
        ax31.set_ylabel(r'$C(P)$')
        ax31.legend()

        ax32.set_title(f'Perimeter CCDF collapsed \n System size = {size}, ' + r'$\kappa_{\text{perim}} = $' + f'{kappa_perim}')
        ax32.set_xlabel(r'$P / |(t_c - t) / t_c|^{-1 / \sigma_{\text{perim}}}$')
        ax32.set_ylabel(r'$C(P) / (|t_c - t| / t_c)^{-(\kappa_{\text{perim}} - 1) / \sigma_{\text{perim}}}$')
        ax32.legend()

        fig3.tight_layout()
        fig3.savefig(f'./perim_ccdf_collapse_s={size}_kappaperim={kappa_perim}_aftertc.png', dpi=200)

    if do_area:
        ax41.set_title(f'Area CCDF uncollapsed \n System size = {size}, ' + r'$\kappa_{\text{area}} = $' + f'{kappa_area}')
        ax41.set_xlabel(r'$A$')
        ax41.set_ylabel(r'$C(A)$')
        ax41.legend()

        ax42.set_title(f'Area CCDF collapsed \n System size = {size}, ' + r'$\kappa_{\text{area}} = $' + f'{kappa_area}')
        ax42.set_xlabel(r'$A / |(t_c - t) / t_c|^{-1 / \sigma_{\text{area}}}$')
        ax42.set_ylabel(r'$C(A) / (|t_c - t| / t_c)^{-(\kappa_{\text{area}} - 1) / \sigma_{\text{area}}}$')
        ax42.legend()

        fig4.tight_layout()
        fig4.savefig(f'./area_ccdf_collapse_s={size}_kappaarea={kappa_area}_aftertc.png', dpi=200)
