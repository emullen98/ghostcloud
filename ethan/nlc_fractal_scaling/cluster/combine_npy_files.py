"""
Created Nov 20 2024
Updated Nov 20 2024

(IN CLUSTER)
Loads in .npy files that contain the xmin, xmax, and exponent from find_pl_montecarlo fits to area/perimeter CCDFs and
combines them into one file.
"""
import numpy as np
import os
import sys

thresh = sys.argv[1]

area_master = np.zeros((500, 3))
perim_master = np.zeros((500, 3))

files = [file for file in os.listdir('.') if file.endswith('.npy') and thresh in file]

for i in range(len(files)):
    data_arr = np.load(f'./{files[i]}')
    area_arr = data_arr[0]
    perim_arr = data_arr[1]
    area_master[i] = area_arr
    perim_master[i] = perim_arr

np.save(f'./find_pl_mc_{thresh}_areas_combined.npy', area_master)
np.save(f'./find_pl_mc_{thresh}_perims_combined.npy', perim_master)
