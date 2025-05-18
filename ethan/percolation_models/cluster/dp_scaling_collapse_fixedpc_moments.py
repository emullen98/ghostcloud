"""
Created May 20 2024
Updated Jun 19 2024

Does a scaling collapse of CCDFs over different times using the distributions' moments.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.stats as stats


do_perim = True
do_area = True
do_before_tc = True
do_after_tc = True
kappa_perim = 2.5
kappa_area = 2
t_c = 7
p_c = 0.381
moment_for_rescaling = 4
folder_list = os.listdir('./lattices/s50000')


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


def rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=None, kind=None):
    """
    Given a moment and the data array, returns the X- and Y-axis rescaling factors using the forms mentioned in Nir's
    thesis.
    :param nth_moment: Moment used in the rescaling; Choose from either the 2nd or 4th moment; INT.
    :param to_rescale: Perimeters or areas to be rescaled; List or Numpy array.
    :param kind: Choose from the two strings 'areas' or 'perims'.
    :return:
    [0] x_rescaling_factor: Rescaling factor for the X-axis.
    [1] y_rescaling_factor: Rescaling factor for the Y-axis.
    """
    data_moment = stats.moment(to_rescale, moment=nth_moment, nan_policy='omit')
    if kind == 'perims':
        x_rescaling_factor = data_moment ** (1 / ((nth_moment + 1) - kappa_perim))
        y_rescaling_factor = data_moment ** ((1 - kappa_perim) / ((nth_moment + 1) - kappa_perim))
    elif kind == 'areas':
        x_rescaling_factor = data_moment ** (1 / ((nth_moment + 1) - kappa_area))
        y_rescaling_factor = data_moment ** ((1 - kappa_area) / ((nth_moment + 1) - kappa_area))

    return x_rescaling_factor, y_rescaling_factor


fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(10, 6))
fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(10, 6))
fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(10, 6))
fig4, (ax41, ax42) = plt.subplots(1, 2, figsize=(10, 6))

if do_before_tc:
    for t in range(3, 7):
        candidate_lattices = []
        for file_name in folder_list:
            found_it = re.findall(f".+end={t}.+", file_name)
            if len(found_it) > 0:
                candidate_lattices.append(found_it[0])
        chosen_lattice = random.choice(candidate_lattices)

        data = np.load(f'./lattices/s50000/{chosen_lattice}')
        perims, areas = data[0], data[1]

        if do_perim:
            phistx, phisty = ccdf(list(perims))
            ax11.loglog(phistx, phisty, label=f'Timestep = {t}')
            x_rescale_perim_beforetc, y_rescale_perim_beforetc = rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=perims, kind='perims')
            phistx_col = phistx / x_rescale_perim_beforetc
            phisty_col = phisty / y_rescale_perim_beforetc
            ax12.loglog(phistx_col, phisty_col, label=f'Timestep = {t}')

        if do_area:
            ahistx, ahisty = ccdf(list(areas))
            ax21.loglog(ahistx, ahisty, label=f'Timestep = {t}')
            x_rescale_area_beforetc, y_rescale_area_beforetc = rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=areas, kind='areas')
            ahistx_col = ahistx / x_rescale_area_beforetc
            ahisty_col = ahisty / y_rescale_area_beforetc
            ax22.loglog(ahistx_col, ahisty_col, label=f'Timestep = {t}')

    if do_perim:
        ax11.set_title(rf'Perimeter CCDF uncollapsed at $p_c = {p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax11.set_xlabel(r'$P$')
        ax11.set_ylabel(r'$C(P)$')
        ax11.legend(fancybox=True, shadow=True)

        ax12.set_title(rf'Perimeter CCDF collapsed at $p_c = {p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax12.set_xlabel(rf'$P / {{\langle P^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}})}}}}$')
        ax12.set_ylabel(rf'$C(P) / {{\langle P^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{perim}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}}}})}}$')
        ax12.legend(fancybox=True, shadow=True)

        fig1.tight_layout()
        fig1.savefig(f'./perim_ccdf_collapse_s=50000_kappaperim={kappa_perim}_beforetc.png', dpi=200)

    if do_area:
        ax21.set_title(rf'Area CCDF uncollapsed at $p_c={p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax21.set_xlabel(r'$A$')
        ax21.set_ylabel(r'$C(A)$')
        ax21.legend(fancybox=True, shadow=True)

        ax22.set_title(rf'Area CCDF collapsed at $p_c={p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax22.set_xlabel(rf'$A / {{\langle A^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax22.set_ylabel(rf'$C(A) / {{\langle A^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{area}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax22.legend(fancybox=True, shadow=True)

        fig2.tight_layout()
        fig2.savefig(f'./area_ccdf_collapse_s=50000_kappaarea={kappa_area}_beforetc.png', dpi=200)

if do_after_tc:
    for t in range(8, 14):
        candidate_lattices = []
        for file_name in folder_list:
            found_it = re.findall(f".+end={t}.+", file_name)
            if len(found_it) > 0:
                candidate_lattices.append(found_it[0])
        chosen_lattice = random.choice(candidate_lattices)

        data = np.load(f'./lattices/s50000/{chosen_lattice}')
        perims, areas = data[0], data[1]

        if do_perim:
            phistx, phisty = ccdf(list(perims))
            ax31.loglog(phistx, phisty, label=f'Timestep = {t}')
            x_rescale_perim_aftertc, y_rescale_perim_aftertc = rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=perims, kind='perims')
            phistx_col = phistx / x_rescale_perim_aftertc
            phisty_col = phisty / y_rescale_perim_aftertc
            ax32.loglog(phistx_col, phisty_col, label=f'Timestep = {t}')

        if do_area:
            ahistx, ahisty = ccdf(list(areas))
            ax41.loglog(ahistx, ahisty, label=f'Timestep = {t}')
            x_rescale_area_aftertc, y_rescale_area_aftertc = rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=areas, kind='areas')
            ahistx_col = ahistx / x_rescale_area_aftertc
            ahisty_col = ahisty / y_rescale_area_aftertc
            ax42.loglog(ahistx_col, ahisty_col, label=f'Timestep = {t}')

    if do_perim:
        ax31.set_title(rf'Perimeter CCDF uncollapsed at $p_c={p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax31.set_xlabel(r'$P$')
        ax31.set_ylabel(r'$C(P)$')
        ax31.legend(fancybox=True, shadow=True)

        ax32.set_title(rf'Perimeter CCDF collapsed at $p_c={p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax32.set_xlabel(rf'$P / {{\langle P^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}}}})}}$')
        ax32.set_ylabel(rf'$C(P) / {{\langle P^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{perim}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}}}})}}$')
        ax32.legend(fancybox=True, shadow=True)

        fig3.tight_layout()
        fig3.savefig(f'./perim_ccdf_collapse_s=50000_kappaperim={kappa_perim}_aftertc.png', dpi=200)

    if do_area:
        ax41.set_title(rf'Area CCDF uncollapsed at $p_c={p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax41.set_xlabel(r'$A$')
        ax41.set_ylabel(r'$C(A)$')
        ax41.legend(fancybox=True, shadow=True)

        ax42.set_title(rf'Area CCDF collapsed at $p_c={p_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax42.set_xlabel(rf'$A / {{\langle A^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax42.set_ylabel(rf'$C(A) / {{\langle A^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{area}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax42.legend(fancybox=True, shadow=True)

        fig4.tight_layout()
        fig4.savefig(f'./area_ccdf_collapse_s=50000_kappaarea={kappa_area}_aftertc.png', dpi=200)
