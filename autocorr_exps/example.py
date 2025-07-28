import utils.autocorr_utils
import utils.cloud_utils

import numpy as xp

LATTICE_SIZE = 100
FILL_PROB = 0.405

raw_lattice = utils.cloud_utils.generate_site_percolation_lattice(LATTICE_SIZE, LATTICE_SIZE, FILL_PROB)
flood_filled_lattice, _ = utils.cloud_utils.flood_fill_and_label_features(raw_lattice)
cropped_clouds = utils.cloud_utils.extract_cropped_clouds_by_size(flood_filled_lattice, min_area=10)

total_num = xp.zeros(0, dtype=float)
total_denom = xp.zeros(0, dtype=float)

for cloud in cropped_clouds:
    padded, cloud_mask = utils.autocorr_utils.pad_image(cloud, max(cloud.shape[0], cloud.shape[1]))
    mask_stack = utils.autocorr_utils.generate_annulus_stack(
        padded.shape, radii=range(1, max(cloud.shape[0], cloud.shape[1]) + 1)
    )
    num_temp, denom_temp = utils.autocorr_utils.compute_radial_autocorr(
        padded, mask_stack, cloud_mask
    )
    num_temp = xp.asarray(num_temp)
    denom_temp = xp.asarray(denom_temp)

    total_num = utils.autocorr_utils.extend_and_add(total_num, num_temp)
    total_denom = utils.autocorr_utils.extend_and_add(total_denom, denom_temp)

# Compute the final autocorrelation
if total_denom.size > 0:
    C_r = total_num / total_denom

#graph C_r as a curve with r as the x-axis
import matplotlib.pyplot as plt
radii = xp.arange(1, len(C_r) + 1)
plt.plot(radii, C_r)
plt.xlabel('Radius (r)')
plt.ylabel('C(r)')
plt.title('Radial Autocorrelation')
plt.grid()
plt.show()