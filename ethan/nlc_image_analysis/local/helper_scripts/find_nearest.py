"""
Created Nov 18 2024
Updated Nov 18 2024


"""
import numpy as np


def find_nearest(area_bins=None, perim_bins=None, area_val=None):
    """
    :param area_bins:
    :param perim_bins:
    :param area_val:
    :return:
    """
    if area_bins is None or perim_bins is None or area_val is None:
        print('One of the parameters for find_nearest() is missing!')
        return 'temp'

    closest_area_bin = min(area_bins, key=lambda x: abs(x - area_val))
    idxs = np.where(area_bins == closest_area_bin)[0][0]
    closest_perim_bin = perim_bins[idxs]

    return closest_perim_bin, closest_area_bin
