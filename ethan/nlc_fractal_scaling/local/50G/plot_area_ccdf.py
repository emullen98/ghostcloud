"""
Created Oct 18 2025
Updated Oct 18 2025

(LOCAL)
-- Plot CCDF of areas for 50G threshold
-- From Monte Carlo power-law fitting, area exponent is 1.735...
"""
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emullen98/Desktop/')
sys.path.append('/Users/emullen98/Desktop/ghostcloud/ethan/nlc_fractal_scaling/local/')
import helper_scripts as hs
from _get_data import get_data

fig, ax = plt.subplots(1, 1, constrained_layout=True)
threshold = 50
filename = f'/Users/emullen98/Downloads/og_clouds/{threshold}G_v4.csv'
mc_exp = -1.73  # From fit_mc_area.py results
mc_amin = 27  # From fit_mc_area.py results
mc_amax = 1482  # From fit_mc_area.py results
plot_save_loc = hs.get_pwd()

area, perims = get_data(csvloc=filename)
ccdf_x, ccdf_y = hs.ccdf(area)

x, y = hs.linemaker(mc_exp + 1, [100, 0.4], 30, 700)

ax.loglog(ccdf_x, ccdf_y, '.', color='grey', label='$\\text{CCDF}(A) \\sim A^{-(\\kappa_{\\text{area}} - 1)}$')
ax.loglog(x, y, color='r', linestyle='dashed', label=f'MC fit $\\kappa_{{\\text{{area}}}} = {mc_exp}$')
ax.axvline(x=mc_amin, color='black', linestyle='dotted', label=f'$A_{{\\min}} = {mc_amin} \\text{{ pixels}}$')
ax.axvline(x=mc_amax, color='black', linestyle='dashdot', label=f'$A_{{\\max}} = {mc_amax} \\text{{ pixels}}$')
ax.set_xlim(10, 1e4)
ax.set_ylim(bottom=1e-4)
ax.set_title(f'Area CCDF for {threshold}G threshold')
ax.set_ylabel('$\\text{Pr}(\\text{area} \\geq A)$')
ax.set_xlabel('$A  /  25\\text{ km}^2$')
ax.legend()
fig.savefig(f'{plot_save_loc}/area_ccdf_{threshold}G.png')
