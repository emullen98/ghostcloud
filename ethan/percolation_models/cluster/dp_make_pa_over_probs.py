"""
Created Jun 14 2024
Updated Oct 02 2024

(IN CLUSTER)
Saves the perimeters & area files for a lattice at a particular timestep and bond probability.
Designed to work with sbatch file "dp_make_pa_over_probs.sbatch"
Uses the slurm array task ID to select the bond probability.
"""
import sys
from helper_scripts.make_lattice import *
from helper_scripts.get_pa import *

job_id = sys.argv[1]
task_id = int(sys.argv[2])
size = int(sys.argv[3])
end_time = int(sys.argv[4])

probs_beforetc = np.round(np.logspace(np.log10(0.28), np.log10(0.38), 20), 5)
probs_aftertc = np.round(np.logspace(np.log10(0.382), np.log10(0.48), 20), 5)
probs = np.concatenate([probs_beforetc, probs_aftertc])
prob = probs[task_id - 1]

save_dir = f'./results/dp/s={size}'

lattice, _, _ = make_lattice(size=size, p=prob, endTime=end_time, fillHoles=True, includeDiags=False)

p, a = get_pa(lattice)
pa = np.array([p, a])
np.save(f'{save_dir}/dp_pa_s={size}_p={prob}_end={end_time}_job={job_id}_task={task_id}.npy', pa)
