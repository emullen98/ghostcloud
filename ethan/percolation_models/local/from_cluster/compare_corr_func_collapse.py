"""
Created Oct 10 2024
Updated May 13 2025
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import helper_scripts as hs

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Initialization stuff
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def truncate_colormap(colormap, minval=0.1, maxval=0.9, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'trunc({colormap.name},{minval:.2f},{maxval:.2f})', colormap(np.linspace(minval, maxval, n)))
    return new_cmap


def convert_to_reduced_p(val, pc):
    """
    Converts a bond probability to its % difference from p_c

    Parameters
    ----------
    val
    pc

    Returns
    -------

    """
    if val > pc:
        converted = (val - pc) / pc * 100
    else:
        converted = (pc - val) / pc * 100

    return converted


plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.titlesize': 20,
    'axes.linewidth': 1.25,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.25,
    'ytick.major.width': 1.25,
    'savefig.dpi': 300,
    'savefig.format': 'png'
})


# Kappas are the PDF exponents
# Sigmas are the difference between the integrated and non-integrated exponents
kappa_perim = 2.50
sigma_perim = 0.6
kappa_area = 2.0
sigma_area = 0.4
dp_pc = 0.381
sp_pc = 0.405
nu = (kappa_area - 1) / (sigma_area * 2)
d_f = 1 / (sigma_area * nu)

dp_cmap = truncate_colormap(cm.autumn).reversed()
sp_cmap = truncate_colormap(cm.winter).reversed()
dp_norm_abovepc = mcolors.Normalize(vmin=1.48, vmax=25.98)
dp_norm_belowpc = mcolors.Normalize(vmin=1.85, vmax=26.51)
sp_norm_abovepc = mcolors.Normalize(vmin=2.40, vmax=25.93)
sp_norm_belowpc = mcolors.Normalize(vmin=2.72, vmax=25.93)

# Comparing areas
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

dp_data_loc = '/Users/emullen98/Desktop/clouds_and_percolation/data/dp_corr_funcs'
sp_data_loc = '/Users/emullen98/Desktop/clouds_and_percolation/data/sp_corr_funcs'
dp_file_names = os.listdir(dp_data_loc)
sp_file_names = os.listdir(sp_data_loc)
file_names = dp_file_names + sp_file_names
dp_unsorted_file_names = []
dp_unsorted_probs = []
sp_unsorted_file_names = []
sp_unsorted_probs = []
for file_name in file_names:
    if file_name.endswith('.npy') and 'dp_corr_func' in file_name:
        temp_string_1 = file_name.split('_')[5]
        dp_unsorted_probs.append(float(temp_string_1[5:]))
        dp_unsorted_file_names.append(file_name)
    if file_name.endswith('.npy') and 'sp_corr_func' in file_name:
        temp_string_1 = file_name.split('_')[5]
        sp_unsorted_probs.append(float(temp_string_1[5:]))
        sp_unsorted_file_names.append(file_name)

dp_unsorted_probs = np.array(dp_unsorted_probs)
dp_unsorted_file_names = np.array(dp_unsorted_file_names)
dp_sorted_probs = dp_unsorted_probs[np.argsort(dp_unsorted_probs)]
dp_sorted_file_names = dp_unsorted_file_names[np.argsort(dp_unsorted_probs)]
dp_sorted_belowpc_dict = dict(zip(dp_sorted_file_names[:18], dp_sorted_probs[:18]))
dp_sorted_abovepc_dict = dict(zip(dp_sorted_file_names[18:], dp_sorted_probs[18:]))
dp_sorted_abovepc_dict = dict(reversed(list(dp_sorted_abovepc_dict.items())))

sp_unsorted_probs = np.array(sp_unsorted_probs)
sp_unsorted_file_names = np.array(sp_unsorted_file_names)
sp_sorted_probs = sp_unsorted_probs[np.argsort(sp_unsorted_probs)]
sp_sorted_file_names = sp_unsorted_file_names[np.argsort(sp_unsorted_probs)]
sp_sorted_belowpc_dict = dict(zip(sp_sorted_file_names[1:19], sp_sorted_probs[1:19]))
sp_sorted_abovepc_dict = dict(zip(sp_sorted_file_names[19:], sp_sorted_probs[19:]))
sp_sorted_abovepc_dict = dict(reversed(list(sp_sorted_abovepc_dict.items())))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Left side: below p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

dp_shift_factor = 2.5
# Starting at the lowest probability (farthest away from p_c):
for name, prob in dp_sorted_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob, dp_pc)
    dists, cf = np.load(f'{dp_data_loc}/{name}')
    rescaled_dists = dists * (p_hat ** nu)
    rescaled_cf = cf * (dists ** (2 * (2 - d_f)))
    ax1[0].loglog(rescaled_dists * dp_shift_factor, rescaled_cf, '.', color=dp_cmap(dp_norm_belowpc(p_hat)))

dp_sm = cm.ScalarMappable(cmap=dp_cmap, norm=dp_norm_belowpc)
dp_sm.set_array([])  # No data array needed
dp_cbar = fig1.colorbar(dp_sm, ax=ax1[0], orientation='horizontal', fraction=0.05, pad=0.05)
dp_cbar.set_ticks([1.85, 26.51])  # Set ticks at min and max
dp_cbar.ax.set_xticklabels(['1.85', '26.51'])  # Format tick labels

dp_cax = inset_axes(ax1[0], width="15%", height="5%", loc='lower left', borderpad=2)
dp_cbar_inset = fig1.colorbar(cm.ScalarMappable(norm=dp_norm_belowpc, cmap=dp_cmap), cax=dp_cax, orientation='horizontal')
dp_cbar_inset.set_ticks([])
dp_cbar_inset.ax.set_xticklabels([])
dp_cbar_inset.ax.xaxis.set_label_position('top')
dp_cbar_inset.set_label('DP')

for name, prob in sp_sorted_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob, sp_pc)
    dists, cf = np.load(f'{sp_data_loc}/{name}')
    rescaled_dists = dists * (p_hat ** nu)
    rescaled_cf = cf * (dists ** (2 * (2 - d_f)))
    ax1[0].loglog(rescaled_dists, rescaled_cf, '.', color=sp_cmap(sp_norm_belowpc(p_hat)))

sp_sm = cm.ScalarMappable(cmap=sp_cmap, norm=sp_norm_belowpc)
sp_sm.set_array([])  # No data array needed
sp_cbar = fig1.colorbar(sp_sm, ax=ax1[0], orientation='horizontal', fraction=0.05, pad=0.05)
sp_cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p_c - p}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
sp_cbar.set_ticks([2.72, 25.93])  # Set ticks at min and max
sp_cbar.ax.set_xticklabels(['2.72', '25.93'])  # Format tick labels

sp_cax = inset_axes(ax1[0], width="15%", height="5%", loc='upper right', borderpad=2)
sp_cbar_inset = fig1.colorbar(cm.ScalarMappable(norm=sp_norm_belowpc, cmap=sp_cmap), cax=sp_cax, orientation='horizontal')
sp_cbar_inset.set_ticks([])
sp_cbar_inset.ax.set_xticklabels([])
sp_cbar_inset.ax.xaxis.set_label_position('top')
sp_cbar_inset.set_label('SP')

ax1[0].set_xlim(right=10**4)
ax1[0].set_xlabel('$|\\textbf{r}|\\hat{p}^{\\nu}$', fontsize=20)
ax1[0].set_ylabel('$g(|\\textbf{r}|)|\\textbf{r}|^{2(d-d_f)}$', fontsize=20)
ax1[0].set_title('Collapsed $\\textbf{below} ~ p_c$')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Right side: above p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

fig1.suptitle(f'Percolation correlation function comparison \n System size = 5000, $\\kappa_{{\\text{{area}}}} = {kappa_area}, \\sigma_{{\\text{{area}}}} = {sigma_area}$')
plt.show()
