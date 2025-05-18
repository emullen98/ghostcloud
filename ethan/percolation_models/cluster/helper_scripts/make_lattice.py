"""
Created xxx xx 2024
Updated Sep 29 2024

(IN CLUSTER)

"""
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from helper_scripts.timestep import timestep


def make_lattice(size=100, p=0.381, endTime=7, fillHoles=True, includeDiags=False):
    lx = ly = size

    if includeDiags:
        m = np.ones((3, 3))
    else:
        m = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    # Generate the lattice
    lattice = np.ones((ly, lx), dtype='int8')
    for i in range(endTime):
        # Timestep returns the lattice as an array of 8-bit (1-byte) integers
        lattice = timestep(lattice, p, lx, ly)

    # Fill the holes or don't
    # Not explicity assigning a type to 'labeledArray'.
    # The largest label will dictate the type of the array, so for very large systems this will likely be an array of 64-bit integers.
    if fillHoles:
        filledLattice = binary_fill_holes(lattice).astype('int8')
        labeledArray, numFeatures = label(filledLattice, structure=m)
    else:
        filledLattice = lattice
        labeledArray, numFeatures = label(filledLattice, structure=m)

    return labeledArray, filledLattice, lattice
