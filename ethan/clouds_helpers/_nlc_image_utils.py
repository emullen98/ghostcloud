"""
Created May 16 2025
Updated May 31 2025

(IN CLUSTER)
Contains utility functions for analyzing high-res PNG images of clouds.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
These functions should NOT be called directly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def _image_array_to_binary_array(input_arr: np.ndarray, thresh: int) -> np.ndarray:
    """
    Converts a PNG cloud image array into a grayscale & binary array

    Grayscale values range from 0-255

    Parameters
    ----------
    input_arr : np.ndarray
        A PNG image that's been converted to an array
    thresh : int
        Grayscale threshold above which to mark a pixel with a 1

    Returns
    -------
    binary_image : np.ndarray
        A PNG image that has been converted to a binary array where 1s represent pixels above threshold in grayscale value
    """
    binary_image = (input_arr > thresh).astype(int)  # Apply the threshold

    return binary_image


def _load_image_as_array(path: str) -> np.ndarray:
    """
    Takes in a FULL path + file name string and converts it to a grayscale array

    Parameters
    ----------
    path : str
        File name plus its path (i.e., /path/to/file/filename.png)

    Returns
    -------
    image_array : np.ndarray
        Grayscale array corresponding to path
    """
    image = Image.open(path).convert('L')  # Ensure it's in grayscale
    image_array = np.array(image)[267:2933, :]  # Remove the white strips on top & bottom

    return image_array


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
These functions should be called directly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def fill_and_label_image(path: str, thresh: int, rem_border_clouds: bool) -> tuple[np.ndarray, int]:
    """
    Fills and labels an input image

    Parameters
    ----------
    path : str
        Input image file name + path (i.e., '/path/to/image/file_name.png')
    thresh : int
        Grayscale threshold
    rem_border_clouds : bool
        Whether to remove clouds that touch the boundary of the image

    Returns
    -------
    labeled_filled_image : np.ndarray of ints
        Labeled & filled version of image array
    num_features : int
        Number of clusters after filling & labeling the image
    """
    if not 0 <= thresh <= 255:
        raise ValueError(f'Invalid \'thresh\' entry {thresh}. Must be between 0-255 inclusive.')

    my_arr = _load_image_as_array(path=path)
    my_arr = _image_array_to_binary_array(input_arr=my_arr, thresh=thresh)
    my_arr = binary_fill_holes(my_arr)

    if rem_border_clouds:
        labeled_filled_image, _ = label(my_arr)  # Number of features needs to be calculated AFTER removing boundary-touching clusters, if applicable

        top_row = labeled_filled_image[0, :]
        bot_row = labeled_filled_image[-1, :]
        left_col = labeled_filled_image[:, 0]
        right_col = labeled_filled_image[:, -1]

        unique_border_labels = np.unique(np.concatenate((top_row, bot_row, left_col, right_col)))
        unique_border_labels = unique_border_labels[unique_border_labels > 0]

        mask = np.isin(labeled_filled_image, unique_border_labels)
        labeled_filled_image[mask] = 0

        labeled_filled_image = (labeled_filled_image > 0).astype(int)
        labeled_filled_image, num_features = label(labeled_filled_image)
    else:
        labeled_filled_image, num_features = label(my_arr)

    return labeled_filled_image, num_features


def label_image(path: str, thresh: int, rem_border_clouds: bool) -> tuple[np.ndarray, int]:
    """
    Labels (but does not fill) an input image

    Parameters
    ----------
    path : str
        Input image file name + path (i.e., '/path/to/image/file_name.png')
    thresh : int
        Grayscale threshold
    rem_border_clouds : bool
        Whether to remove clouds that touch the boundary of the image

    Returns
    -------
    labeled_binary_image : np.ndarray
        Labeled version of image array
    num_features : int
        Number of clusters after labeling the image
    """
    if not 0 <= thresh <= 255:
        raise ValueError(f'Invalid \'thresh\' entry {thresh}. Must be between 0-255 inclusive.')

    my_arr = _load_image_as_array(path=path)
    my_arr = _image_array_to_binary_array(input_arr=my_arr, thresh=thresh)

    if rem_border_clouds:
        labeled_image, _ = label(my_arr)

        top_row = labeled_image[0, :]
        bot_row = labeled_image[-1, :]
        left_col = labeled_image[:, 0]
        right_col = labeled_image[:, -1]

        unique_border_labels = np.unique(np.concatenate((top_row, bot_row, left_col, right_col)))
        unique_border_labels = unique_border_labels[unique_border_labels > 0]

        mask = np.isin(labeled_image, unique_border_labels)
        labeled_image[mask] = 0

        labeled_image = (labeled_image > 0).astype(int)
        labeled_image, num_features = label(labeled_image)
    else:
        labeled_image, num_features = label(my_arr)

    return labeled_image, num_features
