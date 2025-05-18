"""
Created Aug 23 2024
Updated Sep 23 2024

(IN CLUSTER)
--

"""
import numpy as np
from scipy.ndimage import binary_fill_holes
import sys
from helper_scripts.corr_func import corr_func

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
size = int(sys.argv[3])
prob = float(sys.argv[4])
fill_holes = int(sys.argv[5])
frac = float(sys.argv[6])

sp_lattice = np.random.choice([0, 1], size=(size, size), p=[1 - prob, prob]).astype('int8')
if fill_holes:
    sp_lattice = binary_fill_holes(sp_lattice).astype('int8')

if frac == 1:
    distances, cf = corr_func(arr=sp_lattice)
else:
    distances, cf = corr_func(arr=sp_lattice, frac=frac)

if fill_holes:
    np.save(f'./lattices/s{size}/sp_corr_func_fill_prob={prob}_jobid={job_id}_taskid={task_id}.npy', np.array([distances, cf]))
else:
    np.save(f'./lattices/s{size}/sp_corr_func_nofill_prob={prob}_jobid={job_id}_taskid={task_id}.npy', np.array([distances, cf]))
