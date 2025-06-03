"""
Created Jun 21 2024
Updated Apr 13 2025

-- Creates a site percolation square lattice
-- Makes two figures using this lattice:
    (1) A figure showing the flood filling procedure, highlighting the largest clusters
    (2) A figure showing the perimeter vs area scatter plot on log scales
"""
import os
import sys
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_fill_holes
from collections import Counter
import clouds_helpers

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Initialize things
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

fig_loc = os.path.abspath(os.path.dirname(sys.argv[0]))

lx = ly = 1000
prob = 0.405
lattice = np.random.choice([0, 1], size=(ly, lx), p=[1 - prob, prob])


def get_perims_areas(arr):
    """
    -- Gets in_cluster perimeters & areas for a lattice where clusters are labeled uniquely by nonzero integers.
    -- I have checked this function's speed using site percolation lattices.
        -- For an array of linear size 10**3, using numba is disadvantageous.
        -- However, when the linear size is increased to 10**4, the speed up is ~30x with numba applied.

    Parameters
    ----------
    arr : arr
        Lattice to find in_cluster perimeters & areas for

    Returns
    -------
    [0] perims : arr
        cluster perimeters as a numpy array
    [1] areas : arr
        cluster areas as a numpy array
    """
    lx, ly = arr.shape

    cnum = arr.max()  # each in_cluster is labeled uniquely, so the largest label is the number of unique clusters.

    areas = np.zeros(cnum)
    perims = np.zeros(cnum)

    for i in range(lx):
        for j in range(ly):
            idx = arr[i, j] - 1  # If we are at in_cluster label 5, then we want to add the area and perimeter of this in_cluster to the 5th entry in the areas/perims lists, which is index 4.

            if idx >= 0:
                if i == 0 or j == 0 or i == lx - 1 or j == ly - 1:  # If we are at a border site, set the area and perimeter for this in_cluster to a nan.
                    areas[idx] = np.nan
                    perims[idx] = np.nan
                else:
                    if (not np.isnan(areas[idx])) and (not np.isnan(perims[idx])):  # If we are at an interior site, get the area and perimeter contribution to this site's in_cluster.
                        areas[idx] = areas[idx] + 1
                        perims[idx] = perims[idx] + int(arr[i + 1, j] == 0) + int(arr[i - 1, j] == 0) + int(arr[i, j + 1] == 0) + int(arr[i, j - 1] == 0)

    return perims, areas


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 1: Flood filling example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

fig1, (ax11, ax12, ax13) = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)

cmap_default = plt.get_cmap('viridis')  # colormap stuff - there is probably an easier way to do this...
color_0 = cmap_default(0.0)
color_1 = cmap_default(0.33)
color_2 = cmap_default(0.50)
color_3 = cmap_default(0.65)
color_4 = cmap_default(0.90)
color_5 = cmap_default(1.0)
colors = [color_0, color_1, color_2, color_3, color_4, color_5]
cmap = mcolors.ListedColormap(colors)
bounds = [0, 1, 2, 3, 4, 5, 6]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

ax11.pcolormesh(lattice, cmap=cmap, norm=norm)  # Show the lattice before holes have been filled
ax11.set_aspect('equal')
ax11.invert_yaxis()
ax11.set_title('Before flood fill')

lattice_filled = np.array(binary_fill_holes(lattice), dtype='uint8')  # Show the lattice after holes have been filled
ax12.pcolormesh(lattice_filled, cmap=cmap, norm=norm)
ax12.set_aspect('equal')
ax12.invert_yaxis()
ax12.set_title('After flood fill')

labeled_array, _ = label(lattice_filled)  # Show the three four largest clusters on the lattice after holes have been filled
labels_all = labeled_array.flatten()
labels_all = labels_all[np.nonzero(labels_all)].tolist()
temp_list = Counter(labels_all)  # this is a counter object
commons = temp_list.most_common(4)  # [(first value, count), (second value, count), ...]
lattice_filled_labeled = np.copy(lattice_filled)
for i in range(2, len(commons) + 2):  # 2, 3, 4, 5
    lattice_filled_labeled[np.where(labeled_array == commons[i - 2][0])] = i

ax13.pcolormesh(lattice_filled_labeled, cmap=cmap, norm=norm)
ax13.set_aspect('equal')
ax13.invert_yaxis()
ax13.set_title('4 largest clusters \n after flood fill')
fig1.suptitle(f'Lattice of random 0s and 1s, Pr ~ (1) = {prob}', fontsize='xx-large')
# fig1.savefig(f'{fig_loc}/random_sites_p={prob}.png')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Figure 2: Perimeter vs. area scatter plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

p, a = get_perims_areas(labeled_array)
# print(np.nanmin(a))

# fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
# ax2.scatter(a, p)
#
# ax2.set_xlabel('Area')
# ax2.set_ylabel('Perimeter')
# ax2.set_title(f'Lattice of random 0s and 1s, Pr ~ (1) = {prob}')
# ax2.legend()
# ax2.loglog()
# fig2.savefig(f'{fig_loc}/random_sites_pva_p={prob}.png')
