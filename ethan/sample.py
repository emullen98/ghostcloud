from clouds_helpers import get_corr_func
import numpy as np
import sys
import os

ethan_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(ethan_dir)  # Location of both utils and ethan directories
sys.path.append(parent_dir)
from utils.cloud_utils import generate_site_percolation_lattice, flood_fill_and_label_features

task_id = int(sys.argv[1])

save_path = '/projects/illinois/eng/physics/dahmen/mullen/Clouds'

rng = np.random.default_rng(42)  
seeds = rng.integers(low=0, high=2**32, size=100, dtype=np.uint32)
cur_seed = seeds[task_id - 1]

unprocessed_lattice = generate_site_percolation_lattice(2000, 2000, 0.4074, seed=cur_seed)
labeled_lattice, num_features = flood_fill_and_label_features(unprocessed_lattice)

corr_func = get_corr_func(labeled_lattice, num_features, int(round(np.sqrt(2000**2 + 2000**2))), 1)
x = np.arange(0, corr_func.shape[0])

np.save(f'{save_path}/corr_func_data_seed={cur_seed}.npy', np.array([x, corr_func]))
