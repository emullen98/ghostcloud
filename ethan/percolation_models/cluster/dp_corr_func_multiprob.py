"""
Created Oct 06 2024
Updated Oct 06 2024

(IN CLUSTER)
Calculates the correlation function for a lattice at a series of pre-defined probabilities.
"""
from helper_scripts.corr_func import corr_func
from helper_scripts.make_lattice import make_lattice
import sys
import numpy as np

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
size = int(sys.argv[3])
end_time = int(sys.argv[4])
fill_holes = int(sys.argv[5])
frac = float(sys.argv[6])

# These are the probabilities below p_c
# probs = np.round(np.logspace(np.log10(0.28), np.log10(0.38), 20), 5)[:-1]
# These are the probabilites above p_c
probs = np.round(np.logspace(np.log10(0.382), np.log10(0.48), 20), 5)[-6:]

# Loop through the probabilities and make the lattice, calculate its correlation function, and save it
for prob in probs:
    dp_lattice = make_lattice(size=size, p=prob, endTime=end_time, fillHoles=fill_holes)[1]

    if frac == 1:
        distances, cf = corr_func(arr=dp_lattice)
    else:
        distances, cf = corr_func(arr=dp_lattice, frac=frac)

    if fill_holes:
        np.save(f'./dp_corr_func_fill_prob={prob}_timestep={end_time}_jobid={job_id}_taskid={task_id}.npy', np.array([distances, cf]))
    else:
        np.save(f'./dp_corr_func_nofill_prob={prob}_timestep={end_time}_jobid={job_id}_taskid={task_id}.npy', np.array([distances, cf]))
