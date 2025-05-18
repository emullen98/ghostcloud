"""
Created Jun 05 2024
Updated Jun 05 2024

Show a lattice's CCDF
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../twoPlusOne(21)D/')
import common_code as cc

# 'a' = area
# 'p' = perimeter
# 'pva' = perimeter vs area
to_do = 'p'

if to_do == 'a':
    pa = np.load('../data/lattice_pa_s=50000_p=0.381_end=7_job=11548394_task=1.npy')
    areas = pa[1]
    areas = areas[~np.isnan(areas)]

    histx, histy = cc.ccdf(list(areas))
    plt.loglog(histx, histy)
    plt.show()

elif to_do == 'p':
    pa = np.load('../data/lattice_pa_s=50000_p=0.381_end=7_job=11548394_task=1.npy')
    perims = pa[0]
    perims = perims[~np.isnan(perims)]

    histx, histy = cc.ccdf(list(perims))
    plt.loglog(histx, histy)
    plt.show()

elif to_do == 'pva':
    pa = np.load('../data/lattice_pa_s=50000_p=0.381_end=7_job=11548394_task=1.npy')
    perims, areas = pa[0], pa[1]
    perims = perims[~np.isnan(perims)]
    areas = areas[~np.isnan(areas)]

    plt.scatter(areas, perims)
    plt.loglog()
    plt.show()