"""
Created Oct 19 2025
Updated Oct 19 2025

(LOCAL)
-- Plot CCDF of perimeters for 40G threshold
"""
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emullen98/Desktop/')
sys.path.append('/Users/emullen98/Desktop/ghostcloud/ethan/nlc_fractal_scaling/local/')
import helper_scripts as hs
from _get_data import get_data

fig, ax = plt.subplots(1, 1, constrained_layout=True)
threshold = 40
min_area = 15
mc_exp = -1.803  # From fit_mc_area.py results
mc_pmin = 46  # From fit_mc_area.py results
mc_pmax = 5649  # From fit_mc_area.py results

plot_save_loc = hs.get_pwd()
filename = f'/Users/emullen98/Downloads/og_clouds/{threshold}G_v4.csv'
area, perims = get_data(csvloc=filename, min_area=min_area)
ccdf_x, ccdf_y = hs.ccdf(perims)

# x, y = hs.linemaker(mc_exp + 1, [100, 0.4], 30, 700)

ax.loglog(ccdf_x, ccdf_y, '.', color='grey', label='$\\text{CCDF}(P) \\sim P^{-(\\kappa_{\\text{perim}} - 1)}$')
# ax.loglog(x, y, color='r', linestyle='dashed', label=f'MC fit $\\kappa_{{\\text{{perim}}}} = {mc_exp}$')
# ax.axvline(x=mc_pmin, color='black', linestyle='dotted', label=f'$P_{{\\min}} = {mc_pmin} \\text{{ pixels}}$')
# ax.axvline(x=mc_pmax, color='black', linestyle='dashdot', label=f'$P_{{\\max}} = {mc_pmax} \\text{{ pixels}}$')
# ax.set_xlim(10, 1e4)
# ax.set_ylim(bottom=1e-4)
ax.set_title(f'Perimeter CCDF for {threshold}G threshold \n using min area {min_area} pixels')
ax.set_ylabel('$\\text{Pr}(\\text{perim} \\geq P)$')
ax.set_xlabel('$P  /  5\\text{ km}$')
ax.legend()
fig.savefig(f'{plot_save_loc}/perim_ccdf_{threshold}G.png')
