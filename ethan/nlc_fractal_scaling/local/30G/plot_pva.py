"""
Created Nov 18 2024
Updated Nov 18 2024


"""
import numpy as np
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


def plot_pva():
    """

    Returns
    -------

    """
    loc = f'../../../Datasets/NLC_data/csv_files/Areas_perims/30G_v4.csv'

    areas, perims = hs.load_pa(loc)[:2]

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    bx, by, _ = cc.logbinning(np.array(areas), np.array(perims), 50)

    xmin = 100
    closest_perim_bin, closest_area_bin = hs.find_nearest(bx, by, xmin)

    pva_exp, pva_std_err, pva_rsq = cc.fit(np.log10(bx), np.log10(by), xmin=np.log10(xmin))
    pva_exp = np.round(pva_exp[0], 2)
    pva_std_err = np.round(pva_std_err[0], 2)
    pva_rsq = np.round(pva_rsq, 2)

    x, y = cc.linemaker(pva_exp, [10**2, 2*10**2], xmin, max(bx))

    ax.set_title('$P \\sim A^{D_f / 2}$, 30G threshold')
    ax.scatter(areas, perims, s=1, color='silver')
    ax.loglog(bx, by, '.', color='k')
    ax.loglog(x, y, color='r', linestyle='dashed', label=f'LSQ fit $D_{{\\text{{f}}}} = {pva_exp * 2} \\pm {pva_std_err}, R^2 = {pva_rsq}$')
    ax.vlines(x=closest_area_bin, ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1], color='blue', label=f'$A_{{\\text{{min}}}} = {closest_area_bin:.0f}$')
    ax.hlines(y=closest_perim_bin, xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1], color='green', label=f'$P_{{\\text{{min}}}} = {closest_perim_bin:.0f}$')
    ax.set_ylabel('$P ~ / ~ 5\\text{ km}$')
    ax.set_xlabel('$A ~ / ~ 25\\text{ km}^2$')
    ax.set_xlim(left=10)
    ax.set_ylim(bottom=10)
    ax.legend(loc='lower right', fancybox=True, shadow=True)
    fig.savefig(f'../../../Plots/Fractal_scaling/with_fits/pva_30G_fit_xmin={xmin:.0f}.png')

    return 'temp'


plot_pva()
