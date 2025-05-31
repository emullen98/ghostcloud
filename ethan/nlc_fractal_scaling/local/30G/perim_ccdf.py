"""
Created Nov 18 2024
Updated Nov 18 2024


"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
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


def plot_ccdf():
    """
    Plots the perimeter CCDF with Monte Carlo-estimated xmin, xmax, and exponent.

    Returns
    -------

    """
    filename = f'../../../Datasets/NLC_data/csv_files/Areas_perims/30G_v4.csv'
    areas, perims, area_x, area_y, perim_x, perim_y = hs.load_pa(filename)
    xmin, xmax, exp = cc.find_pl_montecarlo(data=perims, stepsize=1)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    x, y = cc.linemaker(1 - exp, [100, 0.8], xmin, max(perims))

    ax.loglog(perim_x, perim_y, '.', color='grey', label='$\\text{CCDF}(P) \\sim P^{-(\\kappa_{\\text{perim}} - 1)}$')
    ax.loglog(x, y, color='r', linestyle='dashed', label=f'MC fit $\\kappa_{{\\text{{perim}}}} = {exp:.2f}$')
    ax.vlines(x=xmin, ymin=min(perim_y), ymax=ax.get_ybound()[1], color='blue',
              label=f'MC $x_{{min}} = {xmin:.0f}$')
    ax.set_title(f'Perimeter CCDF for 30G threshold')
    ax.set_ylabel('$\\text{Pr}(\\text{perim} \\geq P)$')
    ax.set_xlabel('$P ~ / ~ 5\\text{ km}$')
    ax.set_xlim(left=min(perim_x))
    ax.legend(loc='lower left', fancybox=True, shadow=True)
    fig.savefig(f'../../../Plots/Fractal_scaling/with_fits/perim_ccdf_30G_mcfit.png')
    plt.close('all')

    return 'temp'


plot_ccdf()
