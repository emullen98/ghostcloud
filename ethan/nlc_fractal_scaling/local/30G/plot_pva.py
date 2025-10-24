"""
Created Nov 18 2024
Updated Oct 19 2025
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emullen98/Desktop/')
sys.path.append('/Users/emullen98/Desktop/ghostcloud/ethan/nlc_fractal_scaling/local/')
import helper_scripts as hs
from _get_data import get_data


def find_nearest(area_bins=None, perim_bins=None, area_val=None):
    """
    :param area_bins:
    :param perim_bins:
    :param area_val:
    :return:
    """
    if area_bins is None or perim_bins is None or area_val is None:
        print('One of the parameters for find_nearest() is missing!')
        return 'temp'

    closest_area_bin = min(area_bins, key=lambda x: abs(x - area_val))
    idxs = np.where(area_bins == closest_area_bin)[0][0]
    closest_perim_bin = perim_bins[idxs]

    return closest_perim_bin, closest_area_bin


fig, ax = plt.subplots(1, 1, constrained_layout=True)

thresh = 30
num_bins = 50
min_area = 100
plot_save_loc = hs.get_pwd()
loc = f'/Users/emullen98/Downloads/og_clouds/{thresh}G_v4.csv'
areas, perims = get_data(csvloc=loc)

bx, by, _ = hs.logbinning(np.array(areas), np.array(perims), num_bins)
closest_perim_bin, closest_area_bin = find_nearest(bx, by, min_area)

pva_exp, pva_std_err, pva_rsq = hs.fit(np.log10(bx), np.log10(by), xmin=np.log10(min_area))
pva_exp = np.round(pva_exp[0], 2)
pva_std_err = np.round(pva_std_err[0], 2)
pva_rsq = np.round(pva_rsq, 2)

x, y = hs.linemaker(pva_exp, [10**2, 2*10**2], min_area, max(bx))

# ax.set_title('$P \\sim A^{D_f / 2}$, 30G threshold')
ax.scatter(areas, perims, s=1, color='silver')
ax.loglog(bx, by, '.', color='k')
ax.loglog(x, y, color='r', linestyle='dashed', label=f'LSQ fit $D_{{\\text{{f}}}} = {pva_exp * 2} \\pm {pva_std_err}, R^2 = {pva_rsq}$')
ax.vlines(x=closest_area_bin, ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1], color='blue', label=f'$A_{{\\text{{min}}}} = {closest_area_bin:.0f}$')
ax.hlines(y=closest_perim_bin, xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1], color='green', label=f'$P_{{\\text{{min}}}} = {closest_perim_bin:.0f}$')
ax.set_ylabel('$P ~ / ~ 5\\text{ km}$')
ax.set_xlabel('$A ~ / ~ 25\\text{ km}^2$')
ax.set_xlim(left=10)
ax.set_ylim(bottom=10)
ax.legend(loc='lower right')
fig.savefig(f'{plot_save_loc}/pva_30G_fit_xmin={min_area:.0f}.png')

