"""
Created Jun 02 2025
Updated Jun 02 2025

Test perimeter-area scaling for correlated percolation model
"""
from clouds_helpers import generate_2d_correlated_field, get_perimeters_areas, linemaker, logbinning
from scipy.ndimage import binary_fill_holes, label
import numpy as np
import matplotlib.pyplot as plt 

L = 1024              
gamma_val = 0.2      
p_val = 0.5    
seed = 42   

field = generate_2d_correlated_field(L, gamma_val, unit_normalize=True, seed=seed)
perc_map = (field < p_val).astype(int)
perc_map_filled = binary_fill_holes(perc_map)  
perc_map_labeled, num_features = label(perc_map_filled)

perims, areas = get_perimeters_areas(perc_map_labeled)   

perims_binned, areas_binned = logbinning(perims, areas, num_bins=30)[:2]

plt.scatter(areas, perims)
plt.scatter(areas_binned, perims_binned, color='red')

d_f = 1.40
x, y = linemaker(d_f / 2, [100, 200], 20, 3000)
plt.plot(x, y, color='red', linestyle='dashed', label=f'D_f = {d_f:.2f}')

plt.legend()
plt.loglog()
plt.show()