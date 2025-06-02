"""
Created Oct 05 2024
Updated Oct 05 2024

Take site percolation CCDFs generated in the in_cluster and collapse them using their moments.
Bond probabilities were logarithmically spaced, 20 below p_c and 20 above.
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
from helper_scripts_perc.ccdf_rescaling_factors_2 import ccdf_rescaling_factors_2 as rescaler

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
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'trunc({colormap.save_dir},{minval:.2f},{maxval:.2f})', colormap(np.linspace(minval, maxval, n)))
    return new_cmap


# Kappas are the PDF exponents
# Sigmas are the difference between the integrated and non-integrated exponents
kappa_perim = 2.50
kappa_area = 2.0
p_c = 0.405
moment_for_rescaling = 4

cmap = truncate_colormap(cmocean.cm.thermal).reversed()
norm_abovepc = mcolors.Normalize(vmin=2.40, vmax=25.93)
norm_belowpc = mcolors.Normalize(vmin=2.72, vmax=25.93)

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
    if file_name.endswith('.npy') and 'sp_area_ccdf' in file_name:
        temp_string_1 = file_name.split('_')[4]
        unsorted_area_probs.append(float((temp_string_1[2:])))
        unsorted_area_file_names.append(file_name)
    if file_name.endswith('.npy') and 'sp_perim_ccdf' in file_name:
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
    x_rescale_factor, y_rescale_factor = rescaler(ccdf_x=perim_ccdf_x, ccdf_y=perim_ccdf_y, nth_moment=moment_for_rescaling, pdf_exp=kappa_perim)
    phistx_col = perim_ccdf_x / x_rescale_factor
    phisty_col = perim_ccdf_y / y_rescale_factor
    ax1[1].loglog(phistx_col, phisty_col, '.', color=cmap(norm_belowpc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_belowpc)
sm.set_array([])  # No data array needed
cbar = fig1.colorbar(sm, ax=ax1.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\frac{p_c - p}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([2.72, 25.93])  # Set ticks at min and max
cbar.ax.set_xticklabels(['2.72', '25.93'])  # Format tick labels

ax1[0].set_title('Uncollapsed')
ax1[0].set_xlabel('$P$', fontsize=20)
ax1[0].set_ylabel('$C(P)$', fontsize=20)
ax1[1].set_title('Collapsed')
ax1[1].set_xlabel(f'$P \\cdot \\langle P^{moment_for_rescaling} \\rangle ^{{-1 / (5 - \\kappa_{{\\text{{perim}}}})}}$', fontsize=20)
ax1[1].set_ylabel(f'$C(P) \\cdot \\langle P^{moment_for_rescaling} \\rangle ^{{-(1 - \\kappa_{{\\text{{perim}}}}) / ({moment_for_rescaling + 1} - \\kappa_{{\\text{{perim}}}})}}$', fontsize=20)
fig1.suptitle(f'Site percolation perim CCDF collapse $\\mathbf{{below}} ~ p_c$ \n System size = 50000, $\\kappa_{{\\text{{perim}}}} = {kappa_perim}$')
fig1.savefig(f'./sp_perim_ccdf_collapse_belowpc_moment_s=50000_kappa={kappa_perim}.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 2: Perimeters above p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for name, prob in sorted_perim_abovepc_dict.items():
    p_hat = convert_to_reduced_p(prob)
    perim_ccdf_x, perim_ccdf_y = np.load(f'../{name}')
    ax2[0].loglog(perim_ccdf_x, perim_ccdf_y, '.', color=cmap(norm_abovepc(p_hat)))
    x_rescale_factor, y_rescale_factor = rescaler(ccdf_x=perim_ccdf_x, ccdf_y=perim_ccdf_y, nth_moment=moment_for_rescaling, pdf_exp=kappa_perim)
    phistx_col = perim_ccdf_x / x_rescale_factor
    phisty_col = perim_ccdf_y / y_rescale_factor
    ax2[1].loglog(phistx_col, phisty_col, '.', color=cmap((norm_abovepc(p_hat))))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_abovepc)
sm.set_array([])  # No data array needed
cbar = fig2.colorbar(sm, ax=ax2.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\frac{p - p_c}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([2.40, 25.93])  # Set ticks at min and max
cbar.ax.set_xticklabels(['2.40', '25.93'])  # Format tick labels

ax2[0].set_title('Uncollapsed')
ax2[0].set_xlabel('$P$', fontsize=20)
ax2[0].set_ylabel('$C(P)$', fontsize=20)
ax2[1].set_title('Collapsed')
ax2[1].set_xlabel(f'$P \\cdot \\langle P^{moment_for_rescaling} \\rangle ^{{-1 / (5 - \\kappa_{{\\text{{perim}}}})}}$', fontsize=20)
ax2[1].set_ylabel(f'$C(P) \\cdot \\langle P^{moment_for_rescaling} \\rangle ^{{-(1 - \\kappa_{{\\text{{perim}}}}) / ({moment_for_rescaling + 1} - \\kappa_{{\\text{{perim}}}})}}$', fontsize=20)
fig2.suptitle(f'Site percolation perim CCDF collapse $\\mathbf{{above}} ~ p_c$ \n System size = 50000, $\\kappa_{{\\text{{perim}}}} = {kappa_perim}$')
fig2.savefig(f'./sp_perim_ccdf_collapse_abovepc_moment_s=50000_kappa={kappa_perim}.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 3: Areas below p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for name, prob in sorted_area_belowpc_dict.items():
    p_hat = convert_to_reduced_p(prob)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ax3[0].loglog(area_ccdf_x, area_ccdf_y, '.', color=cmap(norm_belowpc(p_hat)))
    x_rescale_factor, y_rescale_factor = rescaler(ccdf_x=area_ccdf_x, ccdf_y=area_ccdf_y, nth_moment=moment_for_rescaling, pdf_exp=kappa_area)
    ahistx_col = area_ccdf_x / x_rescale_factor
    ahisty_col = area_ccdf_y / y_rescale_factor
    ax3[1].loglog(ahistx_col, ahisty_col, '.', color=cmap(norm_belowpc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_belowpc)
sm.set_array([])  # No data array needed
cbar = fig3.colorbar(sm, ax=ax3.ravel().tolist(), orientation='horizontal', fraction=0.05)  # Shared colorbar for both plots
cbar.set_label(r'$\frac{p_c - p}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([2.72, 25.93])  # Set ticks at min and max
cbar.ax.set_xticklabels(['2.72', '25.93'])  # Format tick labels

ax3[0].set_title('Uncollapsed')
ax3[0].set_xlabel('$A$', fontsize=20)
ax3[0].set_ylabel('$C(A)$', fontsize=20)
ax3[1].set_title('Collapsed')
ax3[1].set_xlabel(f'$A \\cdot \\langle A^{moment_for_rescaling} \\rangle ^{{-1 / (5 - \\kappa_{{\\text{{area}}}})}}$', fontsize=20)
ax3[1].set_ylabel(f'$C(A) \\cdot \\langle A^{moment_for_rescaling} \\rangle ^{{-(1 - \\kappa_{{\\text{{area}}}}) / ({moment_for_rescaling + 1} - \\kappa_{{\\text{{area}}}})}}$', fontsize=20)
fig3.suptitle(f'Site percolation area CCDF collapse $\\mathbf{{below}} ~ p_c$ \n System size = 50000, $\\kappa_{{\\text{{area}}}} = {kappa_area}$')
fig3.savefig(f'./sp_area_ccdf_collapse_belowpc_moment_s=50000_kappa={kappa_area}.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 4: Areas above p_c
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for name, prob in sorted_area_abovepc_dict.items():
    p_hat = convert_to_reduced_p(prob)
    area_ccdf_x, area_ccdf_y = np.load(f'../{name}')
    ax4[0].loglog(area_ccdf_x, area_ccdf_y, '.', color=cmap(norm_abovepc(p_hat)))
    x_rescale_factor, y_rescale_factor = rescaler(ccdf_x=area_ccdf_x, ccdf_y=area_ccdf_y, nth_moment=moment_for_rescaling, pdf_exp=kappa_area)
    ahistx_col = area_ccdf_x / x_rescale_factor
    ahisty_col = area_ccdf_y / y_rescale_factor
    ax4[1].loglog(ahistx_col, ahisty_col, '.', color=cmap(norm_abovepc(p_hat)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm_abovepc)
sm.set_array([])  # No data array needed
cbar = fig4.colorbar(sm, ax=ax4.ravel().tolist(), orientation='horizontal', fraction=0.05)
cbar.set_label(r'$\frac{p - p_c}{p_c} \times 100\%$', labelpad=0.1, fontsize=16)
cbar.set_ticks([2.40, 25.93])  # Set ticks at min and max
cbar.ax.set_xticklabels(['2.40', '25.93'])  # Format tick labels

ax4[0].set_title('Uncollapsed')
ax4[0].set_xlabel('$A$', fontsize=20)
ax4[0].set_ylabel('$C(A)$', fontsize=20)
ax4[1].set_title('Collapsed')
ax4[1].set_xlabel(f'$A \\cdot \\langle A^{moment_for_rescaling} \\rangle ^{{-1 / (5 - \\kappa_{{\\text{{area}}}})}}$', fontsize=20)
ax4[1].set_ylabel(f'$C(A) \\cdot \\langle A^{moment_for_rescaling} \\rangle ^{{-(1 - \\kappa_{{\\text{{area}}}}) / ({moment_for_rescaling + 1} - \\kappa_{{\\text{{area}}}})}}$', fontsize=20)
fig4.suptitle(f'Site percolation area CCDF collapse $\\mathbf{{above}} ~ p_c$ \n System size = 50000, $\\kappa_{{\\text{{area}}}} = {kappa_area}$')
fig4.savefig(f'./sp_area_ccdf_collapse_abovepc_moment_s=50000_kappa={kappa_area}.pdf')

