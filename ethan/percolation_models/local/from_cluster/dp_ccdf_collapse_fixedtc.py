"""
Created Oct 04 2024
Updated Oct 04 2024

Take CCDFs generated in the cluster and collapse them.
All CCDFs should be at t_c = 7.
Bond probabilities were logarithmically spaced, 19 below p_c and 19 above.
Note that I removed the CCDFs for lattices with bond probability to close to p_c since those don't collapse

**TO USE**
1. Ensure that this script is in the "local" directory.
2. Ensure that the CCDF .npy files are in the main directory of the percolation project
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cmocean

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Initialization stuff
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

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


def truncate_colormap(colormap, minval=0.1, maxval=0.9, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'trunc({colormap.name},{minval:.2f},{maxval:.2f})', colormap(np.linspace(minval, maxval, n)))
    return new_cmap


# Kappas are the PDF exponents
# Sigmas are the difference between the integrated and non-integrated exponents
kappa_perim = 2.50
sigma_perim = 0.6
kappa_area = 2.0
sigma_area = 0.4
p_c = 0.381
t_c = 7

cmap = truncate_colormap(cmocean.cm.thermal).reversed()
norm_abovepc = mcolors.Normalize(vmin=1.48, vmax=25.98)
norm_belowpc = mcolors.Normalize(vmin=1.85, vmax=26.51)

# Perimeter collapse below p_c
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
# Perimeter collapse above p_c
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
# Perimeter collapse below p_c
fig3, ax3 = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
# Perimeter collapse above p_c
fig4, ax4 = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)


def convert_to_reduced_p(val):
    """
    Converts a bond probability to its % difference from p_c
    :param val:
    :return:
    """
    if val > p_c:
        converted = (val - p_c) / p_c * 100
    else:
        converted = (p_c - val) / p_c * 100

    return converted


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sort the files by probability for better plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

file_names = os.listdir('../')
unsorted_area_file_names = []
unsorted_area_probs = []
unsorted_perim_file_names = []
unsorted_perim_probs = []
for file_name in file_names:
    if file_name.endswith('.npy') and 'dp_area_ccdf' in file_name:
        temp_string_1 = file_name.split('_')[4]
        unsorted_area_probs.append(float((temp_string_1[2:])))
        unsorted_area_file_names.append(file_name)
    if file_name.endswith('.npy') and 'dp_perim_ccdf' in file_name:
        temp_string_1 = file_name.split('_')[4]
        unsorted_perim_probs.append(float((temp_string_1[2:])))
        unsorted_perim_file_names.append(file_name)

unsorted_area_probs = np.array(unsorted_area_probs)
unsorted_area_file_names = np.array(unsorted_area_file_names)
sorted_area_probs = unsorted_area_probs[np.argsort(unsorted_area_probs)]
sorted_area_file_names = unsorted_area_file_names[np.argsort(unsorted_area_probs)]
sorted_area_belowpc_dict = dict(zip(sorted_area_file_names[:19], sorted_area_probs[:19]))
sorted_area_abovepc_dict = dict(zip(sorted_area_file_names[19:], sorted_area_probs[19:]))
sorted_area_abovepc_dict = dict(reversed(list(sorted_area_abovepc_dict.items())))

unsorted_perim_probs = np.array(unsorted_perim_probs)
unsorted_perim_file_names = np.array(unsorted_perim_file_names)
sorted_perim_probs = unsorted_perim_probs[np.argsort(unsorted_perim_probs)]
sorted_perim_file_names = unsorted_perim_file_names[np.argsort(unsorted_perim_probs)]
sorted_perim_belowpc_dict = dict(zip(sorted_perim_file_names[:19], sorted_perim_probs[:19]))
sorted_perim_abovepc_dict = dict(zip(sorted_perim_file_names[19:], sorted_perim_probs[19:]))
sorted_perim_abovepc_dict = dict(reversed(list(sorted_perim_abovepc_dict.items())))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 1: Perimeters below p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


for name, prob in sorted_perim_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob)
    perim_ccdf_x, perim_ccdf_y = np.load(f'../{name}')
    ax1[0].loglog(perim_ccdf_x, perim_ccdf_y, '.', color=cmap(norm_belowpc(p_hat)))
    phistx_col = perim_ccdf_x * (np.abs((p_c - prob) / p_c) ** (1 / sigma_perim))
    phisty_col = perim_ccdf_y * (np.abs((p_c - prob) / p_c) ** (-1 * (kappa_perim - 1) / sigma_perim))
    ax1[1].loglog(phistx_col, phisty_col, '.', color=cmap(norm_belowpc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_belowpc)
sm.set_array([])  # No data array needed
cbar = fig1.colorbar(sm, ax=ax1.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p_c - p}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([1.85, 26.51])  # Set ticks at min and max
cbar.ax.set_xticklabels(['1.85', '26.51'])  # Format tick labels

ax1[0].set_title('Uncollapsed')
ax1[0].set_xlabel('$P$', fontsize=20)
ax1[0].set_ylabel('$C(P)$', fontsize=20)
ax1[1].set_title('Collapsed')
ax1[1].set_xlabel('$P \\cdot \\hat{p}^{1 / \\sigma_{\\text{perim}}}$', fontsize=20)
ax1[1].set_ylabel('$C(P) \\cdot \\hat{p}^{-(\\kappa_{\\text{perim}} - 1) / \\sigma_{\\text{perim}}}$', fontsize=20)
fig1.suptitle(f'Directed percolation perim CCDF collapse $\\mathbf{{below}} ~ p_c$ \n System size = 50000, $t_c = {t_c}, \\kappa_{{\\text{{perim}}}} = {kappa_perim}, \\sigma_{{\\text{{perim}}}} = {sigma_perim}$')
fig1.savefig(f'./dp_perim_ccdf_collapse_belowpc_s=50000_t={t_c}_kappa={kappa_perim}_sigma={sigma_perim}.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 2: Perimeters above p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for name, prob in sorted_perim_abovepc_dict.items():
    p_hat = convert_to_reduced_p(prob)
    perim_ccdf_x, perim_ccdf_y = np.load(f'../{name}')
    ax2[0].loglog(perim_ccdf_x, perim_ccdf_y, '.', color=cmap(norm_abovepc(p_hat)))
    phistx_col = perim_ccdf_x * (np.abs((prob - p_c) / p_c) ** (1 / sigma_perim))
    phisty_col = perim_ccdf_y * (np.abs((prob - p_c) / p_c) ** (-1 * (kappa_perim - 1) / sigma_perim))
    ax2[1].loglog(phistx_col, phisty_col, '.', color=cmap(norm_abovepc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_abovepc)
sm.set_array([])  # No data array needed
cbar = fig2.colorbar(sm, ax=ax2.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p - p_c}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([1.48, 25.98])  # Set ticks at min and max
cbar.ax.set_xticklabels(['1.48', '25.98'])  # Format tick labels

ax2[0].set_title('Uncollapsed')
ax2[0].set_xlabel('$P$')
ax2[0].set_ylabel('$C(P)$')
ax2[1].set_title('Collapsed')
ax2[1].set_xlabel('$P \\cdot \\hat{p}^{1 / \\sigma_{\\text{perim}}}$', fontsize=20)
ax2[1].set_ylabel('$C(P) \\cdot \\hat{p}^{-(\\kappa_{\\text{perim}} - 1) / \\sigma_{\\text{perim}}}$', fontsize=20)
fig2.suptitle(f'Directed percolation perim CCDF collapse $\\mathbf{{above}} ~ p_c$ \n System size = 50000, $t_c = {t_c}, \\kappa_{{\\text{{perim}}}} = {kappa_perim}, \\sigma_{{\\text{{perim}}}} = {sigma_perim}$')
fig2.savefig(f'./dp_perim_ccdf_collapse_abovepc_s=50000_t={t_c}_kappa={kappa_perim}_sigma={sigma_perim}.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 3: Areas below p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for name, prob in sorted_area_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ax3[0].loglog(area_ccdf_x, area_ccdf_y, '.', color=cmap(norm_belowpc(p_hat)))
    ahistx_col = area_ccdf_x * (np.abs((p_c - prob) / p_c) ** (1 / sigma_area))
    ahisty_col = area_ccdf_y * (np.abs((p_c - prob) / p_c) ** (-1 * (kappa_area - 1) / sigma_area))
    ax3[1].loglog(ahistx_col, ahisty_col, '.', color=cmap(norm_belowpc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_belowpc)
sm.set_array([])  # No data array needed
cbar = fig3.colorbar(sm, ax=ax3.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p_c - p}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([1.85, 26.51])  # Set ticks at min and max
cbar.ax.set_xticklabels(['1.85', '26.51'])  # Format tick labels

ax3[0].set_title('Uncollapsed')
ax3[0].set_xlabel('$A$')
ax3[0].set_ylabel('$C(A)$')
ax3[1].set_title('Collapsed')
ax3[1].set_xlabel('$A \\cdot \\hat{p}^{1 / \\sigma_{\\text{area}}}$', fontsize=20)
ax3[1].set_ylabel('$C(A) \\cdot \\hat{p}^{-(\\kappa_{\\text{area}} - 1) / \\sigma_{\\text{area}}}$', fontsize=20)
fig3.suptitle(f'Directed percolation area CCDF collapse $\\mathbf{{below}} ~ p_c$ \n System size = 50000, $t_c = {t_c}, \\kappa_{{\\text{{area}}}} = {kappa_area}, \\sigma_{{\\text{{area}}}} = {sigma_area}$')
fig3.savefig(f'./dp_area_ccdf_collapse_belowpc_s=50000_t={t_c}_kappa={kappa_area}_sigma={sigma_area}.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 4: Areas above p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for name, prob in sorted_area_abovepc_dict.items():
    p_hat = convert_to_reduced_p(prob)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ax4[0].loglog(area_ccdf_x, area_ccdf_y, '.', color=cmap(norm_abovepc(p_hat)))
    ahistx_col = area_ccdf_x * (np.abs((prob - p_c) / p_c) ** (1 / sigma_area))
    ahisty_col = area_ccdf_y * (np.abs((prob - p_c) / p_c) ** (-1 * (kappa_area - 1) / sigma_area))
    ax4[1].loglog(ahistx_col, ahisty_col, '.', color=cmap(norm_abovepc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_abovepc)
sm.set_array([])  # No data array needed
cbar = fig4.colorbar(sm, ax=ax4.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\hat{p} \times 100\% ~ = ~ \frac{p - p_c}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([1.48, 25.98])  # Set ticks at min and max
cbar.ax.set_xticklabels(['1.48', '25.98'])  # Format tick labels

ax4[0].set_title('Uncollapsed')
ax4[0].set_xlabel('$A$')
ax4[0].set_ylabel('$C(A)$')
ax4[1].set_title('Collapsed')
ax4[1].set_xlabel('$A \\cdot \\hat{p}^{1 / \\sigma_{\\text{area}}}$', fontsize=20)
ax4[1].set_ylabel('$C(A) \\cdot \\hat{p}^{-(\\kappa_{\\text{area}} - 1) / \\sigma_{\\text{area}}}$', fontsize=20)
fig4.suptitle(f'Directed percolation area CCDF collapse $\\mathbf{{above}} ~ p_c$ \n System size = 50000, $t_c = {t_c}, \\kappa_{{\\text{{area}}}} = {kappa_area}, \\sigma_{{\\text{{area}}}} = {sigma_area}$')
fig4.savefig(f'./dp_area_ccdf_collapse_abovepc_s=50000_t={t_c}_kappa={kappa_area}_sigma={sigma_area}.pdf')
