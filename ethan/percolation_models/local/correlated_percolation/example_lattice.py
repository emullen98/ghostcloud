"""
Created Jun 01 2025
Updated Jun 02 2025

Shows an example of generating a 2D correlated percolation map with and without holes filled
"""
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from clouds_helpers import generate_2d_correlated_field

L = 1024              
gamma_val = 0.2      
p_val = 0.5927    
seed = 42            

field = generate_2d_correlated_field(L, gamma_val, unit_normalize=True, seed=seed)
perc_map = field < p_val
perc_map_filled = binary_fill_holes(perc_map)  # Fill holes in the percolation map

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(perc_map, cmap='Greys_r', origin='lower')
ax[0].set_title(f"2D Correlated Percolation (γ={gamma_val}, p={p_val})")
ax[0].axis('off')

ax[1].imshow(perc_map_filled, cmap='Greys_r', origin='lower')
ax[1].set_title(f"2D Correlated Percolation (filled) (γ={gamma_val}, p={p_val})")
ax[1].axis('off')

plt.show()