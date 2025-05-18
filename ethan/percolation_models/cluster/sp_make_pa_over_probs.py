"""
Created Oct 02 2024
Updated Oct 02 2024

(IN CLUSTER)
Creates a SP lattice at a given occupation probability and saves its perimeter/area information.
Designed to work with sbatch file "sp_make_pa_over_probs.sbatch".
Uses the slurm array task ID to select the occupation probability.
"""
import sys
import numpy as np
from helper_scripts.get_pa import get_pa
from scipy.ndimage import binary_fill_holes, label

job_id = sys.argv[1]
task_id = int(sys.argv[2])
size = int(sys.argv[3])

probs_beforetc = np.round(np.logspace(np.log10(0.30), np.log10(0.40), 20), 5)
probs_aftertc = np.round(np.logspace(np.log10(0.41), np.log10(0.51), 20), 5)
probs = np.concatenate([probs_beforetc, probs_aftertc])
prob = probs[task_id - 1]

save_dir = f'./results/sp/s={size}'

lattice = np.random.choice([0, 1], size=(size, size), p=[1 - prob, prob])

filled_lattice = binary_fill_holes(lattice).astype('int8')
labeled_array, _ = label(filled_lattice)

p, a = get_pa(labeled_array)
pa = np.array([p, a])
np.save(f'{save_dir}/sp_pa_s={size}_p={prob}_job={job_id}_task={task_id}.npy', pa)
