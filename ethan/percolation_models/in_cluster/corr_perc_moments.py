"""
Created Jun 04 2025
Updated Jun 04 2025

(IN CLUSTER)
For a fixed gamma value, hone in the percolation threshold by computing the first few moments of the percolation cluster size distribution
"""
import sys
import numpy as np
from clouds_helpers import generate_2d_correlated_field, fill_and_label_lattice, get_perimeters_areas
from joblib import Parallel, delayed

task_id = int(sys.argv[1]) 
L = int(sys.argv[2])
gamma = float(sys.argv[3])
threads = int(sys.argv[4])

save_loc = f'/projects/illinois/eng/physics/dahmen/mullen/Clouds/correlated_percolation/perc_thresh_estimates/gamma={gamma:.1f}'

thresholds = np.array([0.470, 0.472, 0.474, 0.476, 0.478, 0.480, 0.482, 0.484, 0.486, 0.488, 0.490, 0.492, 0.494, 0.496, 0.498, 0.500, 0.502, 0.504, 0.506, 0.508, 0.510])


def compute_second_moment_for_threshold(i):
    field = generate_2d_correlated_field(L, gamma, unit_normalize=True)
    perc_map = (field < thresholds[i]).astype(np.int32)
    perc_map, num_features = fill_and_label_lattice(perc_map, rem_border_clusters=True)
    perims, areas = get_perimeters_areas(perc_map)
    if areas.size > 0:
        areas = areas[areas != areas.max()]
        if areas.size > 0:
            return np.mean(areas.astype(np.float64)**2)
    return np.nan


moments = Parallel(n_jobs=threads)(
    delayed(compute_second_moment_for_threshold)(i) for i in range(len(thresholds))
)

moments = np.array(moments)

np.save(f'{save_loc}/moments_{L}_{gamma:.1f}_{min(thresholds):.3f}_{max(thresholds):.3f}_task{task_id}.npy', np.array([thresholds, moments]))
