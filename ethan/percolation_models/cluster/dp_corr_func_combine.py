"""
Created Oct 06 2024
Updated Oct 06 2024

Combines all 100 DP correlation functions generated at a particular bond probability into an averaged file
"""
import sys
import os
import numpy as np

prob = sys.argv[1]
jobids = []
cf_master = []
dists_master = []

for file_name in os.listdir('./'):
    if f'prob={prob}' in file_name and file_name.endswith('.npy'):
        jobids.append(file_name.split('jobid')[1][1:9])
        dists, cf = np.load(f'./{file_name}')
        cf_master.append(cf)
        dists_master.append(dists)

cf_master = np.array(cf_master)
cf_master_mean = np.mean(cf_master, axis=0)
np.save(f'./dp_corr_func_fill_averaged_prob={prob}_timestep=7_jobid={jobids[0]}', np.array([dists_master[0], cf_master_mean]))
