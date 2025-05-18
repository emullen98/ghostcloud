"""
Created Oct 05 2024
Updated Oct 06 2024

Compare area collapses of SP & DP lattices across probabilities.
Uses the CCDF files.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    :param val:
    :return:
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
    'savefig.format': 'pdf'
})


# Kappas are the PDF exponents
# Sigmas are the difference between the integrated and non-integrated exponents
kappa_perim = 2.50
sigma_perim = 0.6
kappa_area = 2.0
sigma_area = 0.4
dp_pc = 0.381
sp_pc = 0.405
t_c = 7

# cmap = truncate_colormap(cmocean.cm.thermal).reversed()
dp_cmap = truncate_colormap(cm.autumn).reversed()
sp_cmap = truncate_colormap(cm.winter).reversed()
dp_norm_abovepc = mcolors.Normalize(vmin=1.48, vmax=25.98)
dp_norm_belowpc = mcolors.Normalize(vmin=1.85, vmax=26.51)
sp_norm_abovepc = mcolors.Normalize(vmin=2.40, vmax=25.93)
sp_norm_belowpc = mcolors.Normalize(vmin=2.72, vmax=25.93)

# Comparing areas
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

file_names = os.listdir('../')
dp_unsorted_area_file_names = []
dp_unsorted_area_probs = []
sp_unsorted_area_file_names = []
sp_unsorted_area_probs = []
for file_name in file_names:
    if file_name.endswith('.npy') and 'dp_area_ccdf' in file_name:
        temp_string_1 = file_name.split('_')[4]
        dp_unsorted_area_probs.append(float(temp_string_1[2:]))
        dp_unsorted_area_file_names.append(file_name)
    elif file_name.endswith('.npy') and 'sp_area_ccdf' in file_name:
        temp_string_1 = file_name.split('_')[4]
        sp_unsorted_area_probs.append(float(temp_string_1[2:]))
        sp_unsorted_area_file_names.append(file_name)

dp_unsorted_area_probs = np.array(dp_unsorted_area_probs)
dp_unsorted_area_file_names = np.array(dp_unsorted_area_file_names)
dp_sorted_area_probs = dp_unsorted_area_probs[np.argsort(dp_unsorted_area_probs)]
dp_sorted_area_file_names = dp_unsorted_area_file_names[np.argsort(dp_unsorted_area_probs)]
dp_sorted_area_belowpc_dict = dict(zip(dp_sorted_area_file_names[:19], dp_sorted_area_probs[:19]))
dp_sorted_area_abovepc_dict = dict(zip(dp_sorted_area_file_names[19:], dp_sorted_area_probs[19:]))
dp_sorted_area_abovepc_dict = dict(reversed(list(dp_sorted_area_abovepc_dict.items())))

sp_unsorted_area_probs = np.array(sp_unsorted_area_probs)
sp_unsorted_area_file_names = np.array(sp_unsorted_area_file_names)
sp_sorted_area_probs = sp_unsorted_area_probs[np.argsort(sp_unsorted_area_probs)]
sp_sorted_area_file_names = sp_unsorted_area_file_names[np.argsort(sp_unsorted_area_probs)]
sp_sorted_area_belowpc_dict = dict(zip(sp_sorted_area_file_names[:19], sp_sorted_area_probs[:19]))
sp_sorted_area_abovepc_dict = dict(zip(sp_sorted_area_file_names[19:], sp_sorted_area_probs[19:]))
sp_sorted_area_abovepc_dict = dict(reversed(list(sp_sorted_area_abovepc_dict.items())))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Left side: below p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Place DP areas below pc on left axes
for name, prob in dp_sorted_area_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob, dp_pc)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ahistx_col = area_ccdf_x * (np.abs(p_hat / 100) ** (1 / sigma_area))
    ahisty_col = area_ccdf_y * (np.abs(p_hat / 100) ** (-1 * (kappa_area - 1) / sigma_area))
    ax1[0].loglog(ahistx_col, ahisty_col, '.', color=dp_cmap(dp_norm_belowpc(p_hat)))

dp_sm = cm.ScalarMappable(cmap=dp_cmap, norm=dp_norm_belowpc)
dp_sm.set_array([])
dp_cbar = fig1.colorbar(dp_sm, ax=ax1[0], orientation='horizontal', fraction=0.05, pad=0.05)
dp_cbar.set_ticks([1.85, 26.51])
dp_cbar.ax.set_xticklabels(['1.85', '26.51'])

dp_cax = inset_axes(ax1[0], width="15%", height="5%", loc='lower left', borderpad=2)
dp_cbar_inset = fig1.colorbar(cm.ScalarMappable(norm=dp_norm_belowpc, cmap=dp_cmap), cax=dp_cax, orientation='horizontal')
dp_cbar_inset.set_ticks([])
dp_cbar_inset.ax.set_xticklabels([])
dp_cbar_inset.ax.xaxis.set_label_position('top')
dp_cbar_inset.set_label('DP')

# Place SP areas below pc on left axes
for name, prob in sp_sorted_area_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob, sp_pc)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ahistx_col = area_ccdf_x * (np.abs(p_hat / 100) ** (1 / sigma_area))
    ahisty_col = area_ccdf_y * (np.abs(p_hat / 100) ** (-1 * (kappa_area - 1) / sigma_area))
    ax1[0].loglog(ahistx_col, ahisty_col, '.', color=sp_cmap(sp_norm_belowpc(p_hat)))

sp_sm = cm.ScalarMappable(cmap=sp_cmap, norm=sp_norm_belowpc)
sp_sm.set_array([])
sp_cbar = fig1.colorbar(sp_sm, ax=ax1[0], orientation='horizontal', fraction=0.05, pad=0.05)
sp_cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p_c - p}{p_c} \times 100\%$', labelpad=0, fontsize=16)
sp_cbar.set_ticks([2.72, 25.93])
sp_cbar.ax.set_xticklabels(['2.72', '25.93'])

sp_cax = inset_axes(ax1[0], width="15%", height="5%", loc='upper right', borderpad=2)
sp_cbar_inset = fig1.colorbar(cm.ScalarMappable(norm=sp_norm_belowpc, cmap=sp_cmap), cax=sp_cax, orientation='horizontal')
sp_cbar_inset.set_ticks([])
sp_cbar_inset.ax.set_xticklabels([])
sp_cbar_inset.ax.xaxis.set_label_position('top')
sp_cbar_inset.set_label('SP')

ax1[0].set_xlabel('$A \\cdot \\hat{p}^{1 / \\sigma_{\\text{area}}}$', fontsize=20)
ax1[0].set_ylabel('$C(A) \\cdot \\hat{p}^{-(\\kappa_{\\text{area}} - 1) / \\sigma_{\\text{area}}}$', fontsize=20)
ax1[0].set_title('$\\textbf{Below} ~ p_c$')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Right side: above p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Place DP areas above pc on right axes
for name, prob in dp_sorted_area_abovepc_dict.items():
    p_hat = convert_to_reduced_p(prob, dp_pc)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ahistx_col = area_ccdf_x * (np.abs(p_hat / 100) ** (1 / sigma_area))
    ahisty_col = area_ccdf_y * (np.abs(p_hat / 100) ** (-1 * (kappa_area - 1) / sigma_area))
    ax1[1].loglog(ahistx_col, ahisty_col, '.', color=dp_cmap(dp_norm_abovepc(p_hat)))

dp_sm = cm.ScalarMappable(cmap=dp_cmap, norm=dp_norm_abovepc)
dp_sm.set_array([])
dp_cbar = fig1.colorbar(dp_sm, ax=ax1[1], orientation='horizontal', fraction=0.05, pad=0.05)
dp_cbar.set_ticks([1.48, 25.98])
dp_cbar.ax.set_xticklabels(['1.48', '25.98'])

dp_cax = inset_axes(ax1[1], width="15%", height="5%", loc='lower left', borderpad=2)
dp_cbar_inset = fig1.colorbar(cm.ScalarMappable(norm=dp_norm_abovepc, cmap=dp_cmap), cax=dp_cax, orientation='horizontal')
dp_cbar_inset.set_ticks([])
dp_cbar_inset.ax.set_xticklabels([])
dp_cbar_inset.ax.xaxis.set_label_position('top')
dp_cbar_inset.set_label('DP')

# Place SP areas above pc on right axes
for name, prob in sp_sorted_area_abovepc_dict.items():
    p_hat = convert_to_reduced_p(prob, sp_pc)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ahistx_col = area_ccdf_x * (np.abs(p_hat / 100) ** (1 / sigma_area))
    ahisty_col = area_ccdf_y * (np.abs(p_hat / 100) ** (-1 * (kappa_area - 1) / sigma_area))
    ax1[1].loglog(ahistx_col, ahisty_col, '.', color=sp_cmap(sp_norm_abovepc(p_hat)))

sp_sm = cm.ScalarMappable(cmap=sp_cmap, norm=sp_norm_abovepc)
sp_sm.set_array([])
sp_cbar = fig1.colorbar(sp_sm, ax=ax1[1], orientation='horizontal', fraction=0.05, pad=0.05)
sp_cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p - p_c}{p_c} \times 100\%$', labelpad=0, fontsize=16)
sp_cbar.set_ticks([2.40, 25.93])
sp_cbar.ax.set_xticklabels(['2.40', '25.93'])

sp_cax = inset_axes(ax1[1], width="15%", height="5%", loc='upper right', borderpad=2)
sp_cbar_inset = fig1.colorbar(cm.ScalarMappable(norm=sp_norm_abovepc, cmap=sp_cmap), cax=sp_cax, orientation='horizontal')
sp_cbar_inset.set_ticks([])
sp_cbar_inset.ax.set_xticklabels([])
sp_cbar_inset.ax.xaxis.set_label_position('top')
sp_cbar_inset.set_label('SP')

ax1[1].set_xlabel('$A \\cdot \\hat{p}^{1 / \\sigma_{\\text{area}}}$', fontsize=20)
ax1[1].set_ylabel('$C(A) \\cdot \\hat{p}^{-(\\kappa_{\\text{area}} - 1) / \\sigma_{\\text{area}}}$', fontsize=20)
ax1[1].set_title('$\\textbf{Above} ~ p_c$')
fig1.suptitle(f'SP vs. DP area CCDF collapse \n System size = 50000, $\\kappa_{{\\text{{area}}}} = {kappa_area}, \\sigma_{{\\text{{area}}}} = {sigma_area}$')
fig1.savefig(f'./area_ccdf_collapse_compare.pdf')

