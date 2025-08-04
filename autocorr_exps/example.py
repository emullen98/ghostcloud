import math
import numpy as np
import matplotlib.pyplot as plt
import utils.autocorr_utils as autocorr_utils
import utils.cloud_utils as cloud_utils
from utils.test_array import test_cloud
from utils.autocorr_utils import xp

test_cloud = test_cloud.astype(np.uint8)

LATTICE_SIZE = 100
FILL_PROB = 0.405

raw_lattice = cloud_utils.generate_site_percolation_lattice(LATTICE_SIZE, LATTICE_SIZE, FILL_PROB)
flood_filled_lattice, _ = cloud_utils.flood_fill_and_label_features(raw_lattice)
cropped_clouds = cloud_utils.extract_cropped_clouds_by_size(flood_filled_lattice, min_area=100)

total_num = xp.zeros(0, dtype=float)
total_denom = xp.zeros(0, dtype=float)

for cloud in cropped_clouds:
    cloud = cloud.astype(np.uint8)
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

#graph C_r as a curve with r as the x-axis as a log log plot
import matplotlib.pyplot as plt
radii = xp.arange(1, len(C_r) + 1)
plt.loglog(radii, C_r)
plt.xlabel('Radius (r)')
plt.ylabel('C(r)')
plt.title('Radial Autocorrelation')
plt.grid()
plt.show()


# h, w = test_cloud.shape
# cloud_area = np.sum(test_cloud)
# max_radius = int(np.floor(np.sqrt(h**2 + w**2)))
# padded_cloud = autocorr_utils.pad_image(test_cloud, pad=max_radius + 2)

# annulus_stack = autocorr_utils.generate_annulus_stack(
#     padded_cloud.shape, radii=range(1, max_radius)
# )

# num, denom = autocorr_utils.compute_radial_autocorr(
#     padded_cloud, annulus_stack
# )

# print(num, denom)
