"""
Created Sep 28 2024
Updated Sep 28 2024

(IN CLUSTER)

"""
import numpy as np


# Credit to Sid for writing this
# Created Nov 30 2023
# Updated May 08 2024
def timestep(arr, prob, lx, ly):
    """
    Evolves a DP lattice by one timestep.
    :param arr: (Numpy array of ints) Current lattice
    :param prob: (Float) Bond probability; critical value for DP is 0.381
    :param lx: (Int) Width of lattice
    :param ly: (Int) Height of lattice
    :return: [0] Numpy array of lattice after one timestep
    """
    prob1 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob2 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob3 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    slice2 = np.roll(arr, shift=(0, -1), axis=(0, 1)).astype('int8')
    slice3 = np.roll(arr, shift=(-1, 0), axis=(0, 1)).astype('int8')
    slice_new = prob1 * arr + prob2 * slice2 + prob3 * slice3
    slice_new = (slice_new > 0).astype('int8')

    return slice_new
