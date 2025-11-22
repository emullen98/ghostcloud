"""
Created Nov 22 2025
Updated Nov 22 2025

(IN CLUSTER)
-- Conducts bootstrap fits to NLC PNG cloud areas & both perimeter definitions to extract exponents

"""
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/projects/illinois/eng/physics/dahmen/mullen')
import helper_scripts as hs

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Load & set parameters (see docstring for details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

jobid = int(sys.argv[1])
taskid = int(sys.argv[2])
bootstrap_runs = int(sys.argv[3])
thresh_type = str(sys.argv[4]) 
hours = str(sys.argv[5])
load_path = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/clouds_paper_data/png_csvs'
save_path = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/clouds_paper_data/png_distribution_fits'

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for file_name in os.listdir(load_path):
    if file_name.endswith('.csv') and f'{thresh_type}_{hours}' in file_name:
        cur_file_name = file_name.split('.csv')[0]  # Use this when saving later
        data_file = os.path.join(load_path, file_name)
        break

df = pd.read_csv(data_file)

areas = df['area_px'].to_numpy()
perims_hull = df['perim_hull_edge'].to_numpy()
perims_acce = df['perim_accessible_edge'].to_numpy()

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Do bootstrapping to extract fractal dimensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

amins, amaxs, areas_exps = np.zeros(bootstrap_runs), np.zeros(bootstrap_runs), np.zeros(bootstrap_runs)
hull_pmins, hull_pmaxs, hull_perim_exps = np.zeros(bootstrap_runs), np.zeros(bootstrap_runs), np.zeros(bootstrap_runs)
acce_pmins, acce_pmaxs, acce_perim_exps = np.zeros(bootstrap_runs), np.zeros(bootstrap_runs), np.zeros(bootstrap_runs)

for run in range(bootstrap_runs):
    boot_areas = np.random.choice(areas, len(areas))
    boot_perims_hull = np.random.choice(perims_hull, len(perims_hull))
    boot_perims_acce = np.random.choice(perims_acce, len(perims_acce))

    amins[run], amaxs[run], areas_exps[run] = hs.find_pl_montecarlo(data=boot_areas)
    hull_pmins[run], hull_pmaxs[run], hull_perim_exps[run] = hs.find_pl_montecarlo(data=boot_perims_hull)
    acce_pmins[run], acce_pmaxs[run], acce_perim_exps[run] = hs.find_pl_montecarlo(data=boot_perims_acce)

final_arr = np.array([amins, amaxs, areas_exps,
                      hull_pmins, hull_pmaxs, hull_perim_exps,
                      acce_pmins, acce_pmaxs, acce_perim_exps])

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Save data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

np.save(file=f'{save_path}/png_bootstrap_distributions_{thresh_type}_{hours}_taskid={taskid}.npy', arr=final_arr)
