"""
Created Aug 14 2024
Updated Aug 14 2024
"""
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from timestep import timestep


def make_lattice(size=100, p=0.381, end_time=7, fill_holes=True, include_diags=False):
    """
    Creates a DP lattice with the given parameters.
    :param size: linear lattice size
    :param p: bond probability
    :param end_time: how many timesteps to evolve the system by
    :param fill_holes: whether to flood fill the clusters
    :param include_diags: whether to include diagonal sites in a cluster
    :return: [0] = lattice with uniquely labeled clusters, [1] = lattice that has been filled (returns just the evolved
    lattice if fill_holes is set to False), [2] = lattice after evolving without filling holes.
    """
    lx = ly = size

    if include_diags:
        m = np.ones((3, 3))
    else:
        m = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    # Generate the lattice
    lattice = np.ones((ly, lx), dtype='int8')
    for i in range(end_time):
        # Timestep returns the lattice as an array of 8-bit (1-byte) integers
        lattice = timestep(lattice, p, lx, ly)

    # Fill the holes or don't
    # Not explicity assigning a type to 'labeledArray'.
    # The largest label will dictate the type of the array, so for very large systems this will likely be an array of 64-bit integers.
    if fill_holes:
        filledLattice = binary_fill_holes(lattice).astype('int8')
        labeledArray, numFeatures = label(filledLattice, structure=m)
    else:
        filledLattice = lattice
        labeledArray, numFeatures = label(filledLattice, structure=m)

    return labeledArray, filledLattice, lattice
