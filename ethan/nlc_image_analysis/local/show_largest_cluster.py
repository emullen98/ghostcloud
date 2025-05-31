"""
Created Sep 20 2024
Updated Sep 20 2024

"""
from helper_scripts.image_to_binary_array import *
from helper_scripts.get_pa import *
import matplotlib.pyplot as plt

threshold = 120
image_loc = '/Users/emullen98/Desktop/atmturb3/Datasets/2013-01-09--08-22-05--578_p0_flatfield_corr.png'
og_image, binary_image, filled_binary_image, labeled_filled_image = image_to_binary_array(path=image_loc, thresh=threshold)

perims, areas = get_pa(labeled_filled_image)

# Identify the largest cluster's label in labeled_filled_image variable
max_area_cluster_label = np.nanargmax(areas) + 1

# Make a lattice to store only sites belonging to the largest filled cluster
new_lattice = np.zeros(shape=labeled_filled_image.shape, dtype=bool)
new_lattice[np.where(labeled_filled_image == max_area_cluster_label)] = 1

plt.imshow(~new_lattice, cmap='binary')
plt.title(f'Largest cluster size for threshold = {threshold}')
plt.show()
