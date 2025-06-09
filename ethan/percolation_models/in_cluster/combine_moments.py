"""
Created Jun 05 2025
Updated Jun 08 2025

(IN CLUSTER)
Combine the output .npy files from corr_perc_moments.py into a single .npy file.
"""
import numpy as np
import os

loc = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/correlated_percolation/perc_thresh_estimates/gamma=2.0'
file_names = [name for name in os.listdir(loc) if name.endswith('.npy')]

thresholds = np.linspace(start=0.45, stop=0.51, num=round((0.51 - 0.45) / 0.002) + 1)

# Number of rows = (number of runs within a script x number of scripts being run) + an initial row for thresholds
# One column per threshold
data_arr = np.zeros(shape=(10 * len(file_names) + 1, len(thresholds)))
data_arr[0, :] = thresholds
for i in range(len(file_names)):
    data = np.load(f'{loc}/{file_names[i]}')
    # Skip the first row since it is just thresholds
    # Each file contributes 10 rows
    # First file contributes rows 1-10, second file 11-20, etc.
    data_arr[10 * i + 1:10 * (i + 1), :] = data[1:, :]

np.save(f'{loc}/combined_moments.npy', data_arr)
