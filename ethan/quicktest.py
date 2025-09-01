import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label
import clouds_helpers
from clouds_helpers._general_utils import get_corr_func_new

# lattice = np.zeros((2, 4))
# lattice[0, 0:] = 1 
# lattice[1, :2] = 1
# print(get_corr_func_new(lattice.astype(int), num_features=1, min_cluster_size=1))


L = 128              
gamma_val = 0.2      
p_val = 0.5927    
seed = 42            
field = clouds_helpers.generate_2d_correlated_field(L, gamma_val, unit_normalize=True, seed=seed)
perc_map = field < p_val
perc_map_filled = binary_fill_holes(perc_map)  # Fill holes in the percolation map
perc_map_labeled, num_features = label(perc_map_filled)
get_corr_func_new(perc_map_labeled, num_features, min_cluster_size=1)