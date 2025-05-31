"""
Created Nov 18 2024
Updated Nov 19 2024
"""
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emullen98/Desktop/')
sys.path.append('/nlc_image_analysis/local')
import common_code as cc
import helper_scripts as hs

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{gensymb}',
    'axes.titlesize': 20,
    'axes.linewidth': 1.25,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.25,
    'ytick.major.width': 1.25,
    'xtick.major.pad': 10,
    'savefig.dpi': 300,
    'savefig.format': 'png'
})


filename = f'../../../Datasets/NLC_data/csv_files/Areas_perims/40G_v4.csv'
areas, perims, area_x, area_y, perim_x, perim_y = hs.load_pa(filename)
xmin, xmax, exp = cc.find_pl_montecarlo(data=areas, stepsize=1)

fig, ax = plt.subplots(1, 1, constrained_layout=True)

x, y = cc.linemaker(1 - exp, [100, 0.8], 30, 1000)

ax.loglog(area_x, area_y, '.', color='grey',
          label='$\\text{CCDF}(A) \\sim A^{-(\\kappa_{\\text{area}} - 1)}$')
ax.loglog(x, y, color='r', linestyle='dashed',
          label=f'MC fit $\\kappa_{{\\text{{area}}}} = {exp:.2f}$')
ax.vlines(x=xmin, ymin=min(area_y), ymax=ax.get_ybound()[1], color='blue',
          label=f'MC $x_{{min}} = {xmin:.0f}$')
ax.set_title(f'Area CCDF for 40G threshold')
ax.set_ylabel('$\\text{Pr}(\\text{area} \\geq A)$')
ax.set_xlabel('$A  /  25\\text{ km}^2$')
ax.set_xlim(left=min(area_x))
ax.legend(loc='upper right', fancybox=True, shadow=True)
fig.savefig(f'../../../Plots/Fractal_scaling/with_fits/area_ccdf_40G_mcfit.png')
plt.close('all')

