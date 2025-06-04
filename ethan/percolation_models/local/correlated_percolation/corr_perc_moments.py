"""
Created Jun 04 2025
Updated Jun 04 2025

Test divergence of second moment of cluster size distribution for correlated percolation model
"""
from clouds_helpers import generate_2d_correlated_field, get_perimeters_areas, linemaker, logbinning, fill_and_label_lattice
from helper_scripts import ccdf
from scipy.ndimage import binary_fill_holes, label
import scipy.stats as stats
import matplotlib.pyplot as plt 

L = 1024              
gamma = 0.2      
# Critical percolation threshold appears to be around 0.495 for linear system size of 1024
thresholds = [0.49, 0.492, 0.494, 0.496, 0.498]
seed = 42

moments = []
for p in thresholds:
    field = generate_2d_correlated_field(L, gamma, unit_normalize=True, seed=42) 
    perc_map = (field < p).astype(int)
    perc_map, num_features = fill_and_label_lattice(perc_map, rem_border_clusters=True)

    perims, areas = get_perimeters_areas(perc_map)   
    areas = areas[areas != areas.max()]  # Remove the largest cluster

    moment = stats.moment(areas, moment=2, nan_policy='omit', center=0)
    moments.append(moment)

plt.plot(thresholds, moments, marker='o')
plt.title(f'Gamma = {gamma}, L={L}')
plt.xlabel('Percolation Threshold (p)')
plt.ylabel('<A^2>')
plt.show()
