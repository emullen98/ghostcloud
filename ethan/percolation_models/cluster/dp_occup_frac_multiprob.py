"""
Created Oct 10 2024
Updated Oct 10 2024

(IN CLUSTER)
Get the occupied fraction of a DP lattice after hole filling for each some set of 40 probabilities, 20 below and 20
above p_c.
"""
import sys
from helper_scripts.make_lattice import *

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
size = int(sys.argv[3])
fill_holes = int(sys.argv[4])
end_time = int(sys.argv[5])

# These are the probabilities below p_c
probs_below = np.round(np.logspace(np.log10(0.28), np.log10(0.38), 20), 5)
# These are the probabilites above p_c
probs_above = np.round(np.logspace(np.log10(0.382), np.log10(0.48), 20), 5)
probs = np.concatenate((probs_below, probs_above))

occup_fracs = np.zeros(shape=(2, 40))
occup_fracs[0] = probs

for i in range(len(probs)):
    prob = probs[i]
    dp_lattice = make_lattice(size=size, p=prob, endTime=end_time, fillHoles=fill_holes)[1]

    occup_frac = np.sum(dp_lattice) / size ** 2
    occup_fracs[1][i] = occup_frac

np.save(f'./dp_occup_frac_multiprob_size={size}_endtime={end_time}_jobid={job_id}_taskid={task_id}.npy', occup_fracs)
