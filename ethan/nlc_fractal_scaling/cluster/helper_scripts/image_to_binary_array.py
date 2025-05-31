"""
Created Sep 15 2024
Updated Sep 16 2024

"""
from PIL import Image
import numpy as np
from scipy.ndimage import binary_fill_holes, label


# Updated Sep 16 2024
def image_to_binary_array(path, thresh=100, fill_holes=True, label_clusters=True):
    """
    Takes in a PNG file and converts it into a grayscale and binary array.
    Grayscale values range from 0 -> 255.
    :param path: (String, required) Path to .png file.
    :param thresh: (Integer, optional) Grayscale threshold (0-255) for binary conversion. If a pixel grayscale value is
    > threshold, it is assigned a value of 1.
    :param fill_holes: (Boolean, optional) Whether to fill holes after grayscale conversion. Default is True.
    :param label_clusters: (Boolean, optional) Whether to label clusters in hole-filled binary image. Default is True.
    :return: [0] Numpy array of image converted to grayscale; [1] Numpy array of image converted to binary;
    [2] Numpy array of binary image with holes filled; [3] Numpy array of binary image with holes filled & clusters
    labeled
    """
    original_image = Image.open(path).convert('L')  # Ensure it's in grayscale

    image_array = np.array(original_image)
    # Remove the white strips on top & bottom
    image_array = image_array[267:2933, :]

    # Apply the threshold
    binary_image = (image_array > thresh).astype(int)

    if fill_holes:
        filled_binary_image = binary_fill_holes(binary_image)
        if label_clusters:
            labeled_filled_binary_image, _ = label(filled_binary_image)
        else:
            labeled_filled_binary_image = filled_binary_image
    else:
        filled_binary_image = labeled_filled_binary_image = binary_image

    return image_array, binary_image, filled_binary_image, labeled_filled_binary_image
