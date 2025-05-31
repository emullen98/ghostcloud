"""
Created Jun 26 2024
Updated Apr 30 2025

Analyze a PNG file of NLCs.
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import binary_fill_holes, label


def getPA(arr):
    """
    I have checked this function's speed using random numpy arrays of 0s and 1s.
    For an array of linear size 10**3, using numba is disadvantageous.
    However, when the linear size is increased to 10**4, the speed up is ~30x with numba applied.
    :param arr:
    :return:
    """
    lx, ly = arr.shape

    # Each cluster is labeled uniquely, so the largest label is the number of unique clusters.
    cnum = arr.max()

    areas = np.zeros(cnum)
    perims = np.zeros(cnum)

    for i in range(lx):
        for j in range(ly):
            # If we are at cluster label 5, then we want to add the area and perimeter of this
            # cluster to the 5th entry in the areas/perims lists, which is index 4.
            idx = arr[i, j] - 1

            if idx >= 0:
                # If we are at a border site, set the area and perimeter for this cluster to a nan.
                if i == 0 or j == 0 or i == lx - 1 or j == ly - 1:
                    areas[idx] = np.nan
                    perims[idx] = np.nan
                else:
                    # If we are at an interior site, get the area and perimeter contribution to
                    # this site's cluster.
                    if (not np.isnan(areas[idx])) and (not np.isnan(perims[idx])):
                        areas[idx] = areas[idx] + 1
                        perims[idx] = perims[idx] + int(arr[i + 1, j] == 0) + int(arr[i - 1, j] == 0) + int(arr[i, j + 1] == 0) + int(arr[i, j - 1] == 0)

    return perims, areas


def image_to_binary_array(path, thresh=100):
    # Open the image using Pillow
    original_image = Image.open(path).convert('L')  # Ensure it's in grayscale

    # Convert the image to a numpy array
    image_array = np.array(original_image)

    # Remove the white strips on top & bottom 266 pixel rows
    image_array = image_array[267:2933, :]

    # Apply the threshold
    binary_image = (image_array > thresh).astype(int)

    return image_array, binary_image


threshold = 100
og_image, bin_image = image_to_binary_array('/Users/emullen98/Downloads/2013-01-09--08-22-05--578_p0_flatfield_corr.png', thresh=threshold)
filled_image = binary_fill_holes(bin_image)
labeled_image, _ = label(filled_image)

p, a = getPA(labeled_image)
print('Max perimeter: ', np.nanmax(p))
print('Max area: ', np.nanmax(a))
print('Min area: ', np.nanmin(a))
print('Number of clusters not touching boundary (using perims): ', len(p[~np.isnan(p)]))
print('Number of clusters touching boundary (using perims): ', len(p[np.isnan(p)]))
print('Number of clusters not touching boundary (using areas): ', len(a[~np.isnan(a)]))
print('Number of clusters touching boundary (using areas): ', len(a[np.isnan(a)]))


fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

ax1.imshow(og_image)
ax2.imshow(bin_image)
ax3.imshow(filled_image)

ax1.set_title('New NLC raw image \n 2013-01-09--08-22-05--578_p0_flatfield_corr')
fig1.tight_layout()
fig1.savefig(f'grayscale_demo_raw_thresh={threshold}.png', dpi=200)

ax2.set_title(f'New NLC image grayscale threshold = {threshold} \n 2013-01-09--08-22-05--578_p0_flatfield_corr')
fig2.tight_layout()
fig2.savefig(f'grayscale_demo_thresh={threshold}.png', dpi=200)

ax3.set_title(f'New NLC image holes filled \n 2013-01-09--08-22-05--578_p0_flatfield_corr')
fig3.tight_layout()
fig3.savefig(f'grayscale_demo_filled_thresh={threshold}.png', dpi=200)
