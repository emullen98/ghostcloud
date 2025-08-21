import math
import numpy as np
import matplotlib.pyplot as plt
import clouds.utils.autocorr_utils as autocorr_utils
import clouds.utils.cloud_utils as cloud_utils
# from clouds.utils.test_array import test_cloud, test_lattice, test_square, test_lattice_large
from clouds.utils.autocorr_utils import xp

# test_cloud = test_lattice_large.astype(np.uint8)

LATTICE_SIZE = 400
FILL_PROB = 0.405
SEED = 1001

raw_lattice = cloud_utils.generate_site_percolation_lattice(LATTICE_SIZE, LATTICE_SIZE, FILL_PROB, seed = SEED)
# raw_lattice = test_cloud.copy().astype(bool)

flood_filled_lattice, _ = cloud_utils.flood_fill_and_label_features(raw_lattice)
cropped_clouds = cloud_utils.extract_cropped_clouds_by_size(flood_filled_lattice, min_area=10)

total_num = xp.zeros(0, dtype=float)
total_denom = xp.zeros(0, dtype=float)

for cloud in cropped_clouds:
    cloud = xp.asarray(cloud, dtype=xp.uint8)
    h,w = cloud.shape
    max_radius = int(math.floor(math.sqrt(h**2 + w**2)))
    padded_cloud = autocorr_utils.pad_image(cloud, pad=max_radius + 2)

    annulus_stack = autocorr_utils.generate_annulus_stack(
        padded_cloud.shape, radii=range(1, max_radius)
    )

    num_temp, denom_temp = autocorr_utils.compute_radial_autocorr(
        padded_cloud, annulus_stack
    )

    num_temp = xp.asarray(num_temp)
    denom_temp = xp.asarray(denom_temp)
    # print(num_temp, denom_temp)

    total_num = autocorr_utils.extend_and_add(total_num, num_temp)
    total_denom = autocorr_utils.extend_and_add(total_denom, denom_temp)

C_r = total_num / total_denom

with open('scratch/annulus_'+str(LATTICE_SIZE)+'_seed_'+str(SEED)+'.txt', 'w') as f:
    for val in C_r:
        f.write(f"{val}\n")

with open('scratch/annulus_'+str(LATTICE_SIZE)+'_seed_'+str(SEED)+'_numden.txt', 'w') as f:
    for num, denom in zip(total_num, total_denom):
        f.write(f"{num},{denom}\n")
