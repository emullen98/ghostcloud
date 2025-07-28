import utils.autocorr_utils
import utils.cloud_utils
import math

import numpy as xp

LATTICE_SIZE = 100
FILL_PROB = 0.405

raw_lattice = utils.cloud_utils.generate_site_percolation_lattice(LATTICE_SIZE, LATTICE_SIZE, FILL_PROB)
flood_filled_lattice, _ = utils.cloud_utils.flood_fill_and_label_features(raw_lattice)
cropped_clouds = utils.cloud_utils.extract_cropped_clouds_by_size(flood_filled_lattice, min_area=50)

total_num = xp.zeros(0, dtype=float)
total_denom = xp.zeros(0, dtype=float)

utils.autocorr_utils.print_lattice(flood_filled_lattice)

count = 0

for cloud in cropped_clouds:
    h, w = cloud.shape
    max_radius = int(math.floor(h**2 + w**2)**0.5)
    padded, cloud_mask = utils.autocorr_utils.pad_image(cloud, max_radius + 2)
    mask_stack = utils.autocorr_utils.generate_annulus_stack(
        padded.shape, radii=range(0, max_radius)
    )
    utils.autocorr_utils.print_lattice(padded)
    #create an empty array of the same shape as the annuli in the stack, not the stack itself
    final_disk = xp.zeros(mask_stack.shape[1:], dtype=bool)
    for mask in mask_stack:
        utils.autocorr_utils.print_lattice(mask)
    #since the final_disk is the same shape as every mask, we simply add the masks to it as a simple numpy operation
        final_disk += mask
    utils.autocorr_utils.print_lattice(final_disk)

    count += 1
    if count > 1:  
        break

    num_temp, denom_temp = utils.autocorr_utils.compute_radial_autocorr(
        padded, mask_stack, cloud_mask
    )
    num_temp = xp.asarray(num_temp)
    denom_temp = xp.asarray(denom_temp)
    print(num_temp, denom_temp)

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