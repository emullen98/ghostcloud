"""
Created on Aug 14 2024
Updated on Aug 14 2024

Original code written by Sid & Ethan Nov 30 2023
"""
import numpy as np


def timestep(slices, prob, lx, ly):
    """
    Return a square 2D lattice after evolving according to DP rules.
    :param slices: the lattice as a numpy array
    :param prob: bond probability
    :param lx: number of rows in lattice
    :param ly: number of columns in lattice
    :return: updated lattice
    """
    prob1 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob2 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob3 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    slice2 = np.roll(slices, shift=(0, -1), axis=(0, 1)).astype('int8')
    slice3 = np.roll(slices, shift=(-1, 0), axis=(0, 1)).astype('int8')
    slice_new = prob1 * slices + prob2 * slice2 + prob3 * slice3
    slice_new = (slice_new > 0).astype('int8')

    return slice_new
