"""
Created Jun 04 2025
Updated Jun 04 2025

For the purposes of demoing parallelization for use in the cluster

For a fixed gamma value, hone in the percolation threshold by computing the first few moments of the percolation cluster size distribution
"""
# Pseudocode
#
# Gamma will be fixed to 0.2
# p will be varied to find the percolation threshold
# Estimate of perc threshold is around 0.49 for linear system size of 1024
#
# 1. For each of N=10 thresholds (p values), do the following:
# 2. For each of n=100 iterations, do the following:
#   a. Initialize a list to store the second moments
#   b. Generate a 2D correlated percolation map
#   c. Fill the holes 
#   d. Remove the border clusters and the single largest cluster
#   e. Compute the second moment of the cluster size distribution
#   f. Save this second moment to the list
# 3. Save the second moments for each p value
import numpy as np
import matplotlib.pyplot as plt 
import time
from clouds_helpers import generate_2d_correlated_field, fill_and_label_lattice, get_perimeters_areas
from joblib import Parallel, delayed

# thresholds = np.array([0.480, 0.482, 0.484, 0.486, 0.488, 0.490, 0.492, 0.494, 0.496, 0.498, 0.500, 0.502, 0.504, 0.506, 0.508, 0.510])
thresholds = np.array([0.480, 0.482, 0.484, 0.486, 0.488, 0.490, 0.492, 0.494])
L = 1024
gamma = 0.2
seed = 42


def compute_second_moment_for_threshold(i):
    field = generate_2d_correlated_field(L, gamma, unit_normalize=True, seed=seed+i)
    perc_map = (field < thresholds[i]).astype(np.int32)
    perc_map, num_features = fill_and_label_lattice(perc_map, rem_border_clusters=True)
    perims, areas = get_perimeters_areas(perc_map)
    if areas.size > 0:
        areas = areas[areas != areas.max()]
        if areas.size > 0:
            return np.mean(areas.astype(np.float64)**2)
    return np.nan


start = time.time()
# moments = [compute_second_moment_for_threshold(i) for i in range(len(thresholds))]
moments = Parallel(n_jobs=8)(
    delayed(compute_second_moment_for_threshold)(i) for i in range(len(thresholds))
)
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")


plt.plot(thresholds, moments, marker='o')
plt.show()
