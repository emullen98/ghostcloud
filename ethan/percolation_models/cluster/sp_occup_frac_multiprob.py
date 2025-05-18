"""
Created Oct 10 2024
Updated Oct 10 2024

(IN CLUSTER)
Get the occupied fraction of a SP lattice after hole filling for each some set of 40 probabilities, 20 below and 20
above p_c.
"""
from scipy.ndimage import binary_fill_holes
import sys
import numpy as np

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
size = int(sys.argv[3])
fill_holes = int(sys.argv[4])

# These are the probabilities below p_c
probs_below = np.round(np.logspace(np.log10(0.30), np.log10(0.40), 20), 5)
# These are the probabilites above p_c
probs_above = np.round(np.logspace(np.log10(0.41), np.log10(0.51), 20), 5)
probs = np.concatenate((probs_below, probs_above))

occup_fracs = np.zeros(shape=(2, 40))
occup_fracs[0] = probs

for i in range(len(probs)):
    prob = probs[i]
    sp_lattice = np.random.choice([0, 1], size=(size, size), p=[1 - prob, prob]).astype('int8')
    if fill_holes:
        sp_lattice = binary_fill_holes(sp_lattice).astype('int8')

    occup_frac = np.sum(sp_lattice) / size ** 2
    occup_fracs[1][i] = occup_frac

np.save(f'./sp_occup_frac_multiprob_size={size}_jobid={job_id}_taskid={task_id}.npy', occup_fracs)
