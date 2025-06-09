"""
Created Jun 04 2025
Updated Jun 08 2025

(IN CLUSTER)
For a fixed gamma value, hone in the percolation threshold 
"""
import sys
import numpy as np
from clouds_helpers import generate_2d_correlated_field, fill_and_label_lattice, get_perimeters_areas
from joblib import Parallel, delayed

task_id = int(sys.argv[1]) 
L = int(sys.argv[2])
gamma = float(sys.argv[3])
threads = int(sys.argv[4])
thresh_min = float(sys.argv[5])
thresh_max = float(sys.argv[6])     
thresh_step = float(sys.argv[7])
sub_runs = int(sys.argv[8])

save_loc = f'/projects/illinois/eng/physics/dahmen/mullen/Clouds/correlated_percolation/perc_thresh_estimates/gamma={gamma:.1f}'

thresholds = np.linspace(start=thresh_min, stop=thresh_max, num=round((thresh_max - thresh_min) / thresh_step) + 1, endpoint=True)


def compute_second_moment_for_threshold(i):
    second_moments = np.zeros(sub_runs)  # Don't set dtype to int here since it's possible to have NaN values
    for j in range(sub_runs):
        field = generate_2d_correlated_field(L, gamma, unit_normalize=True)
        perc_map = (field < thresholds[i]).astype(np.int32)  # Use index i here to access the threshold
        perc_map = fill_and_label_lattice(perc_map, rem_border_clusters=True)[0]
        areas = get_perimeters_areas(perc_map)[1]
        if areas.size > 0:
            areas = areas[areas != areas.max()]
            if areas.size > 0:
                second_moments[j] = np.mean(areas.astype(np.float64)**2)
            else:
                second_moments[j] = np.nan
        else:
            second_moments[j] = np.nan
    
    return second_moments


# Returns a list of arrays, where each array contains (sub_runs) second moments for a specific threshold
# I.e., in this case, it returns a list of 31 arrays, each containing 10 second moment values
moments = Parallel(n_jobs=threads)(
    delayed(compute_second_moment_for_threshold)(i) for i in range(len(thresholds))
)

np.save(f'{save_loc}/moments_{L}_{sub_runs}_{gamma:.1f}_{thresh_min:.3f}_{thresh_max:.3f}_task{task_id}.npy', np.vstack([thresholds, np.array(moments).T]))
