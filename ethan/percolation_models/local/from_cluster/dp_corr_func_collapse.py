"""
Created Oct 10 2024
Updated Oct 10 2024


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean
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
    'savefig.format': 'png'
})


# Kappas are the PDF exponents
# Sigmas are the difference between the integrated and non-integrated exponents
kappa_perim = 2.50
sigma_perim = 0.6
kappa_area = 2.0
sigma_area = 0.4
dp_pc = 0.381
nu = (kappa_area - 1) / (sigma_area * 2)
d_f = 1 / (sigma_area * nu)

# Darkest colors first
cmap = truncate_colormap(cmocean.cm.thermal).reversed()
dp_norm_abovepc = mcolors.Normalize(vmin=1.48, vmax=25.98)
dp_norm_belowpc = mcolors.Normalize(vmin=1.85, vmax=26.51)

# Comparing areas
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

file_names = os.listdir('../')
dp_unsorted_file_names = []
dp_unsorted_probs = []
for file_name in file_names:
    if file_name.endswith('.npy') and 'dp_corr_func' in file_name:
        temp_string_1 = file_name.split('_')[5]
        dp_unsorted_probs.append(float(temp_string_1[5:]))
        dp_unsorted_file_names.append(file_name)

dp_unsorted_probs = np.array(dp_unsorted_probs)
dp_unsorted_file_names = np.array(dp_unsorted_file_names)

dp_sorted_probs = dp_unsorted_probs[np.argsort(dp_unsorted_probs)]
dp_sorted_file_names = dp_unsorted_file_names[np.argsort(dp_unsorted_probs)]

dp_sorted_belowpc_dict = dict(zip(dp_sorted_file_names[:18], dp_sorted_probs[:18]))

dp_sorted_abovepc_dict = dict(zip(dp_sorted_file_names[18:], dp_sorted_probs[18:]))
dp_sorted_abovepc_dict = dict(reversed(list(dp_sorted_abovepc_dict.items())))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 1: below p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Starting at the lowest probability (farthest away from p_c):
for name, prob in dp_sorted_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob, dp_pc)
    dists, cf = np.load(f'../{name}')
    ax1[0].loglog(dists, cf, '.', color=cmap(dp_norm_belowpc(p_hat)))
    rescaled_dists = dists * (p_hat ** nu)
    rescaled_cf = cf * (dists ** (2 * (2 - d_f)))
    ax1[1].loglog(rescaled_dists, rescaled_cf, '.', color=cmap(dp_norm_belowpc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=dp_norm_belowpc)
sm.set_array([])  # No data array needed
cbar = fig1.colorbar(sm, ax=ax1.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p_c - p}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([1.85, 26.51])  # Set ticks at min and max
cbar.ax.set_xticklabels(['1.85', '26.51'])  # Format tick labels

ax1[0].set_title('Uncollapsed')
ax1[0].set_xlabel('$|\\textbf{r}|$', fontsize=20)
ax1[0].set_ylabel('$g(|\\textbf{r}|)$', fontsize=20)
ax1[0].set_xlim(right=10**3)
ax1[1].set_title('Collapsed')
ax1[1].set_xlabel('$|\\textbf{r}|\\hat{p}^{\\nu}$', fontsize=20)
ax1[1].set_ylabel('$g(|\\textbf{r}|)|\\textbf{r}|^{2(d-d_f)}$', fontsize=20)
ax1[1].set_xlim(right=10**4)
fig1.suptitle(f'Directed percolation correlation function collapse $\\mathbf{{below}} ~ p_c$ \n System size = 5000, t=7, $\\kappa_{{\\text{{area}}}} = {kappa_area}, \\sigma_{{\\text{{area}}}} = {sigma_area}$')
fig1.savefig(f'./dp_cf_collapse_belowpc_s=50000_t=7_kappa={kappa_area}_sigma={sigma_area}.png')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 2: above p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Starting at the highest probability (farthest away from p_c):
for name, prob in dp_sorted_abovepc_dict.items():
    p_hat = convert_to_reduced_p(prob, dp_pc)
    dists, cf = np.load(f'../{name}')
    ax2[0].loglog(dists, cf, '.', color=cmap(dp_norm_abovepc(p_hat)))
    rescaled_dists = dists * (p_hat ** nu)
    rescaled_cf = cf * (dists ** (2 * (2 - d_f)))
    ax2[1].loglog(rescaled_dists, rescaled_cf, '.', color=cmap(dp_norm_abovepc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=dp_norm_abovepc)
sm.set_array([])  # No data array needed
cbar = fig2.colorbar(sm, ax=ax2.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p - p_c}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([1.48, 25.98])  # Set ticks at min and max
cbar.ax.set_xticklabels(['1.48', '25.98'])  # Format tick labels

ax2[0].set_title('Uncollapsed')
ax2[0].set_xlabel('$|\\textbf{r}|$', fontsize=20)
ax2[0].set_ylabel('$g(|\\textbf{r}|)$', fontsize=20)
# ax2[0].set_xlim(right=10**3)
ax2[1].set_title('Collapsed')
ax2[1].set_xlabel('$|\\textbf{r}|\\hat{p}^{\\nu}$', fontsize=20)
ax2[1].set_ylabel('$g(|\\textbf{r}|)|\\textbf{r}|^{2(d-d_f)}$', fontsize=20)
# ax2[1].set_xlim(right=10**4)
fig2.suptitle(f'Directed percolation correlation function collapse $\\mathbf{{above}} ~ p_c$ \n System size = 5000, t=7, $\\kappa_{{\\text{{area}}}} = {kappa_area}, \\sigma_{{\\text{{area}}}} = {sigma_area}$')
fig2.savefig(f'./dp_cf_collapse_abovepc_s=50000_t=7_kappa={kappa_area}_sigma={sigma_area}.png')
