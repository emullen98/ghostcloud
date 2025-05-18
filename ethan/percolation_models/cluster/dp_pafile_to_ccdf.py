"""
Created Jun 11 2024
Updated Oct 04 2024

(IN CLUSTER)
Generate perimeter & area CCDFs at t_c=7 for different bond probabilities.
Use these CCDFs to do collapses.
"""
import numpy as np
import os
import sys
from helper_scripts.ccdf import ccdf

task_id = sys.argv[1]

load_loc = './results/dp/s=50000'
list_of_files = os.listdir(load_loc)

for file_name in list_of_files:
    if f'task={task_id}' in file_name:
        file_info_string = file_name[6:]
        data = np.load(f'{load_loc}/{file_name}')
        perims, areas = data[0], data[1]
        perims = perims[~np.isnan(perims)]
        areas = areas[~np.isnan(areas)]
        perim_ccdf_x, perim_ccdf_y = ccdf(perims)
        area_ccdf_x, area_ccdf_y = ccdf(areas)
        np.save(f'./dp_area_ccdf_{file_info_string}', np.array([area_ccdf_x, area_ccdf_y]))
        np.save(f'./dp_perim_ccdf_{file_info_string}', np.array([perim_ccdf_x, perim_ccdf_y]))
