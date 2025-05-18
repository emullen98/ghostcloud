"""
Created Oct 08 2024
Updated Oct 08 2024

(IN CLUSTER)
Calculates the correlation function for a lattice at a series of pre-defined probabilities.
"""
from helper_scripts.corr_func import corr_func
from scipy.ndimage import binary_fill_holes
import sys
import numpy as np

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
size = int(sys.argv[3])
fill_holes = int(sys.argv[4])
frac = float(sys.argv[5])

# These are the probabilities below p_c
probs = np.round(np.logspace(np.log10(0.30), np.log10(0.40), 20), 5)[:10]
# These are the probabilites above p_c
# probs = np.round(np.logspace(np.log10(0.41), np.log10(0.51), 20), 5)

for prob in probs:
    sp_lattice = np.random.choice([0, 1], size=(size, size), p=[1 - prob, prob]).astype('int8')
    if fill_holes:
        sp_lattice = binary_fill_holes(sp_lattice).astype('int8')

    if frac == 1:
        distances, cf = corr_func(arr=sp_lattice)
    else:
        distances, cf = corr_func(arr=sp_lattice, frac=frac)

    if fill_holes:
        np.save(f'./sp_corr_func_fill_prob={prob}_jobid={job_id}_taskid={task_id}.npy',
                np.array([distances, cf]))
    else:
        np.save(f'./sp_corr_func_nofill_prob={prob}_jobid={job_id}_taskid={task_id}.npy',
                np.array([distances, cf]))
