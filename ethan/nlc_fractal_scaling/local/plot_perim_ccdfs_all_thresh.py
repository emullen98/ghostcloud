"""
Created Nov 18 2024
Updated Nov 19 2024


"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean
import sys
import itertools
sys.path.append('/nlc_image_analysis/local')
import helper_scripts as hs

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{gensymb}',
    'lines.linewidth': 4,
    'legend.fontsize': 20,
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

normalized_thresh = mcolors.Normalize(vmin=30, vmax=60)
cmap = hs.truncate_cm(cmocean.cm.thermal)
colors = itertools.cycle(cmap(normalized_thresh((30, 40, 50, 60))))

loc_30G = f'../../Datasets/NLC_data/csv_files/Areas_perims/30G_v4.csv'
loc_40G = f'../../Datasets/NLC_data/csv_files/Areas_perims/40G_v4.csv'
loc_50G = f'../../Datasets/NLC_data/csv_files/Areas_perims/50G_v4.csv'
loc_60G = f'../../Datasets/NLC_data/csv_files/Areas_perims/60G_v4.csv'
ccdfx_30G, ccdfy_30G = hs.load_pa(loc_30G)[4:]
ccdfx_40G, ccdfy_40G = hs.load_pa(loc_40G)[4:]
ccdfx_50G, ccdfy_50G = hs.load_pa(loc_50G)[4:]
ccdfx_60G, ccdfy_60G = hs.load_pa(loc_60G)[4:]

fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.loglog(ccdfx_30G, ccdfy_30G, color=next(colors), label='30G')
ax.loglog(ccdfx_40G, ccdfy_40G, color=next(colors), label='40G')
ax.loglog(ccdfx_50G, ccdfy_50G, color=next(colors), label='50G')
ax.loglog(ccdfx_60G, ccdfy_60G, color=next(colors), label='60G')
ax.set_title('Perimeter CCDF for all thresholds')
ax.set_ylabel('$\\text{Pr}(\\text{perim} \\geq P)$')
ax.set_xlabel('$P  /  5\\text{ km}$')
ax.legend(loc='lower left', fancybox=True, shadow=True)

fig.savefig('../../Plots/Fractal_scaling/no_fits/perim_ccdf_all_thresh.png')
