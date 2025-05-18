"""
Created Aug 23 2024
Updated Sep 23 2024

(IN CLUSTER)
-- Computes and saves a directed percolation (DP) lattice given some parameters.

"""
from helper_scripts.corr_func import corr_func
from helper_scripts.make_lattice import make_lattice
import sys
import numpy as np

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
size = int(sys.argv[3])
prob = float(sys.argv[4])
end_time = int(sys.argv[5])
fill_holes = int(sys.argv[6])
frac = float(sys.argv[7])

dp_lattice = make_lattice(size=size, p=prob, endTime=end_time, fillHoles=fill_holes)[1]

if frac == 1:
    distances, cf = corr_func(arr=dp_lattice)
else:
    distances, cf = corr_func(arr=dp_lattice, frac=frac)

if fill_holes:
    np.save(f'./dp_corr_func_fill_prob={prob}_timestep={end_time}_jobid={job_id}_taskid={task_id}.npy', np.array([distances, cf]))
else:
    np.save(f'./dp_corr_func_nofill_prob={prob}_timestep={end_time}_jobid={job_id}_taskid={task_id}.npy', np.array([distances, cf]))
