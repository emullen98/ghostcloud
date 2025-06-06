"""
Created Jun 05 2025
Updated Jun 05 2025

(IN CLUSTER)
Combine the output .npy files from corr_perc_moments.py into a single .npy file.
"""
import numpy as np
import os

loc = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/correlated_percolation/perc_thresh_estimates/gamma=2.0'
file_names = [name for name in os.listdir(loc) if name.endswith('.npy')]

thresholds = np.linspace(start=0.46, stop=0.51, num=round((0.51 - 0.46) / 0.002) + 1)

# One row per run (plus an initial row for thresholds)
# One column per threshold
data_arr = np.zeros(shape=(len(file_names) + 1, len(thresholds)))
data_arr[0, :] = thresholds
for i in range(len(file_names)):
    data = np.load(f'{loc}/{file_names[i]}')
    data_arr[i + 1, :] = data[1, :]

np.save(f'{loc}/combined_moments.npy', data_arr)
