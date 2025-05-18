"""
Created Aug 14 2024
Updated Aug 14 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from timestep import timestep

default_path = os.path.dirname(sys.argv[0])
default_path = os.path.abspath(default_path)


def frame_generator(times=np.arange(1, 101, 2), size=500, prob=0.381, occup='full', path=default_path):
    """
    Creates images at timesteps "times" for a DP lattice with the given parameters.
    :param times: array of times to generate snapshots at.
    :param size: linear size of the system.
    :param prob: bond probability. Default was determined by tweaking known values slightly.
    :param occup: initial state of lattice. Options are 'full' for a fully occupied initial state and any other string
    for a single site to be initially occupied.
    :param path: path to the directory where images should be stored. Default is absolute path of CWD.
    seed)
    :return: nothing.
    """
    times = times.astype('int')

    lx = ly = size

    if occup == 'full':
        curLattice = np.ones((ly, lx))
    else:
        curLattice = np.zeros((ly, lx))
        yco, xco = np.random.choice(size, 2)
        curLattice[yco][xco] = 1

    for i in range(1, max(times) + 1):
        curLattice = timestep(curLattice, prob, lx, ly)
        print(f'Timestep {i} / {max(times)}')
        if i in times:
            plt.imshow(curLattice)
            plt.title(f'Timestep {i} \n size={size}, prob.={prob}')
            plt.savefig(f'{path}/frame_{i}.png')

    return 'temp'
