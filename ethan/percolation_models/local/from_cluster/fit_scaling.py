"""
Created Jun 03 2024
Updated Jun 05 2024

Do fits to different scaling things for a given lattice's perimeter-area file
"""

import powerlaw as pl
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../twoPlusOne(21)D/')
import common_code as cc
from helper_scripts_perc.linemaker import linemaker

# 'a' = area
# 'p' = perimeter
# 'pva' = perimeter vs area
to_do = 'pva'

if to_do == 'a':
    pa = np.load('../data/lattice_pa_s=50000_p=0.381_end=7_job=11548394_task=1.npy')
    areas = pa[1]
    areas = areas[~np.isnan(areas)]

    # xmin based on initial testing
    fit = pl.Fit(areas, xmin=200)
    print(fit.xmin)
    fig = fit.plot_ccdf(linestyle='None', marker='.')
    fit.power_law.plot_ccdf(color='r', linewidth=1, linestyle='--', alpha=1, ax=fig, label=rf'$|\tau|$ = {fit.power_law.alpha :.2f}')

    plt.legend()
    plt.show()

elif to_do == 'p':
    pa = np.load('../data/lattice_pa_s=50000_p=0.381_end=7_job=11548394_task=1.npy')
    perims = pa[0]
    perims = perims[~np.isnan(perims)]

    # xmin based on initial testing
    fit = pl.Fit(perims, xmin=280)
    print(fit.xmin)
    fig = fit.plot_ccdf(linestyle='None', marker='.')
    fit.power_law.plot_ccdf(color='r', linewidth=1, linestyle='--', alpha=1, ax=fig,
                            label=rf'$|\alpha|$ = {fit.power_law.alpha :.2f}')

    plt.legend()
    plt.show()

elif to_do == 'pva':
    pa = np.load('../data/lattice_pa_s=50000_p=0.381_end=7_job=11548394_task=1.npy')
    perims, areas = pa[0], pa[1]
    perims = perims[~np.isnan(perims)]
    areas = areas[~np.isnan(areas)]

    area_bins, perim_bins = cc.logbinning(areas, perims, 50)[:2]
    # I think these are safe limits from looking at the log binned values
    exp = cc.fit(np.log10(area_bins), np.log10(perim_bins), xmin=np.log10(3*10**2), xmax=np.log10(10**5))[0][0]
    print(exp)
    x, y = linemaker(exp, [10**3, 10**4], 2*10**2, 10**5)

    plt.axvline(x=200)
    plt.scatter(areas, perims)
    plt.plot(area_bins, perim_bins, '.', color='r')
    plt.plot(x, y, color='r', linestyle='dashed')
    plt.loglog()
    plt.show()
