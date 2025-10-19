"""
Created Nov 19 2024
Updated Nov 19 2024

(IN CLUSTER)
-- Uses default parameters / discrete setting in find_pl_montecarlo to extract best-fit xmin, xmax, and power-law exponents to a particular threshold's area & perimeter data.

"""
import numpy as np
import sys
import helper_scripts as hs

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
thresh = str(sys.argv[3])

filename = f'./data/{thresh}_v4.csv'

areas, perims = hs.load_pa(filename)[0:2]
area_xmin, area_xmax, area_exp = hs.find_pl_montecarlo(data=areas, stepsize=1)
perim_xmin, perim_xmax, perim_exp = hs.find_pl_montecarlo(data=perims, stepsize=1)

results_arr = np.array([[area_xmin, area_xmax, area_exp], [perim_xmin, perim_xmax, perim_exp]])

np.save(f'./find_pl_mc_{thresh}_jobid={job_id}_taskid={task_id}.npy', results_arr)
