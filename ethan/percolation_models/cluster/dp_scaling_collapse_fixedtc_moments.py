"""
Created May 22 2024
Updated Jun 19 2024

Does a scaling collapse of CCDFs over different bond probabilities using the distributions' moments.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy.stats as stats

do_perim = True
do_area = True
do_before_pc = True
do_after_pc = True
kappa_perim = 2.5
kappa_area = 2
t_c = 7
p_c = 0.381
# Choose from either the 2nd or 4th moments ?
moment_for_rescaling = 4
prob_list_beforepc = [0.32, 0.34, 0.35, 0.36, 0.37]
prob_list_afterpc = [0.384, 0.39, 0.40, 0.41, 0.42]
folder_list = os.listdir(f'./lattices/s50000')


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

if do_before_pc:
    for prob in prob_list_beforepc:
        found_it = ''
        for file_name in folder_list:
            found_it_temp = re.findall(f'lattice_pa_s=50000_p={prob}_end=7_job=.*_task=1.npy', file_name)
            if len(found_it_temp) > 0:
                found_it = found_it_temp[0]
        data = np.load(f'./lattices/s50000/{found_it}')
        perims, areas = data[0], data[1]

        if do_perim:
            phistx, phisty = ccdf(list(perims))
            ax11.loglog(phistx, phisty, '.', label=f'Bond prob. = {prob}')
            x_rescale_perim_beforepc, y_rescale_perim_beforepc = rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=perims, kind='perims')
            phistx_col = phistx / x_rescale_perim_beforepc
            phisty_col = phisty / y_rescale_perim_beforepc
            ax12.loglog(phistx_col, phisty_col, '.', label=f'Bond prob. = {prob}')

        if do_area:
            ahistx, ahisty = ccdf(list(areas))
            ax21.loglog(ahistx, ahisty, '.', label=f'Bond prob. = {prob}')
            x_rescale_area_beforepc, y_rescale_area_beforepc = rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=areas, kind='areas')
            ahistx_col = ahistx / x_rescale_area_beforepc
            ahisty_col = ahisty / y_rescale_area_beforepc
            ax22.loglog(ahistx_col, ahisty_col, '.', label=f'Bond prob. = {prob}')

    if do_perim:
        ax11.set_title(rf'Perimeter CCDF uncollapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax11.set_xlabel(r'$P$')
        ax11.set_ylabel(r'$C(P)$')
        ax11.legend(fancybox=True, shadow=True)

        ax12.set_title(rf'Perimeter CCDF collapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax12.set_xlabel(rf'$P / {{\langle P^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}})}}}}$')
        ax12.set_ylabel(rf'$C(P) / {{\langle P^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{perim}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}}}})}}$')
        ax12.legend(fancybox=True, shadow=True)

        fig1.tight_layout()
        fig1.savefig(f'./perim_ccdf_collapse_s=50000_kappaperim={kappa_perim}_beforepc.png', dpi=200)

    if do_area:
        ax21.set_title(rf'Area CCDF uncollapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax21.set_xlabel(r'$A$')
        ax21.set_ylabel(r'$C(A)$')
        ax21.legend(fancybox=True, shadow=True)

        ax22.set_title(rf'Area CCDF collapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax22.set_xlabel(rf'$A / {{\langle A^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax22.set_ylabel(rf'$C(A) / {{\langle A^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{area}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax22.legend(fancybox=True, shadow=True)

        fig2.tight_layout()
        fig2.savefig(f'./area_ccdf_collapse_s=50000_kappaarea={kappa_area}_beforepc.png', dpi=200)

if do_after_pc:
    for prob in prob_list_afterpc:
        found_it = ''
        for file_name in folder_list:
            found_it_temp = re.findall(f'lattice_pa_s=50000_p={prob}_end=7_job=.*_task=1.npy', file_name)
            if len(found_it_temp) > 0:
                found_it = found_it_temp[0]
        data = np.load(f'./lattices/s50000/{found_it}')
        perims, areas = data[0], data[1]

        if do_perim:
            phistx, phisty = ccdf(list(perims))
            ax31.loglog(phistx, phisty, '.', label=f'Bond prob. = {prob}')
            x_rescale_perim_afterpc, y_rescale_perim_afterpc = rescaling_factors(nth_moment=moment_for_rescaling, to_rescale=perims, kind='perims')
            phistx_col = phistx / x_rescale_perim_afterpc
            phisty_col = phisty / y_rescale_perim_afterpc
            ax32.loglog(phistx_col, phisty_col, '.', label=f'Bond prob. = {prob}')

        if do_area:
            ahistx, ahisty = ccdf(list(areas))
            ax41.loglog(ahistx, ahisty, '.', label=f'Bond prob. = {prob}')
            x_rescale_area_afterpc, y_rescale_area_afterpc = rescaling_factors(nth_moment=4, to_rescale=areas, kind='areas')
            ahistx_col = ahistx / x_rescale_area_afterpc
            ahisty_col = ahisty / y_rescale_area_afterpc
            ax42.loglog(ahistx_col, ahisty_col, '.', label=f'Bond prob. = {prob}')

    if do_perim:
        ax31.set_title(rf'Perimeter CCDF uncollapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax31.set_xlabel(r'$P$')
        ax31.set_ylabel(r'$C(P)$')
        ax31.legend(fancybox=True, shadow=True)

        ax32.set_title(rf'Perimeter CCDF collapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{perim}}}}}} = {kappa_perim}$')
        ax32.set_xlabel(rf'$P / {{\langle P^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}}}})}}$')
        ax32.set_ylabel(rf'$C(P) / {{\langle P^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{perim}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{perim}}}})}}$')
        ax32.legend(fancybox=True, shadow=True)

        fig3.tight_layout()
        fig3.savefig(f'./perim_ccdf_collapse_s=50000_kappaperim={kappa_perim}_afterpc.png', dpi=200)

    if do_area:
        ax41.set_title(rf'Area CCDF uncollapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax41.set_xlabel(r'$A$')
        ax41.set_ylabel(r'$C(A)$')
        ax41.legend(fancybox=True, shadow=True)

        ax42.set_title(rf'Area CCDF collapsed at $t_c={t_c}$' + '\n System size = 50000, ' + rf'$\kappa_{{{{\text{{area}}}}}} = {kappa_area}$')
        ax42.set_xlabel(rf'$A / {{\langle A^{moment_for_rescaling} \rangle}}^{{1 / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax42.set_ylabel(rf'$C(A) / {{\langle A^{moment_for_rescaling} \rangle}}^{{(1 - \kappa_{{\text{{area}}}}) / ({moment_for_rescaling + 1} - \kappa_{{\text{{area}}}})}}$')
        ax42.legend(fancybox=True, shadow=True)

        fig4.tight_layout()
        fig4.savefig(f'./area_ccdf_collapse_s=50000_kappaarea={kappa_area}_afterpc.png', dpi=200)
