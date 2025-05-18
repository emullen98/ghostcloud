"""
Written 3/30/23

This script has one goal:
Run the simulation some number of times for each system size in some specified range. For each system size, calculate
the average values of both the maximum cluster & gap sizes. As an example, imagine we start from system size N = 100. We
run m = 200 simulations for this system size, recording the maximum cluster size and the maximum gap size at some time T
for each of these 200 simulations. We then take the average of this value such that in the end, we have an avg. max
sizes for both clusters and gaps at each system size.

The two variants (so far) that I have done are to fix the time T to be either...
1) T = 0.75*N (i.e., the time that we calculate the max cluster & gap sizes at increases with the system size)
2) T = 100 (time is fixed)

The results for the two cases are as follows:
1) T = 0.75*N: cluster sizes increase logarithmically, but gap sizes increase like a power law with exponent < 1
2) T = 100: Both sizes increase logarithmically
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import mpltex
from numba import njit
from numba import vectorize
from joblib import Parallel, delayed


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simulation functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


@vectorize(nopython=True)
def logic2(a, b, r1, p):
    if a == 1:
        if b == 1:  # CASE A A
            if r1 < (1 - (1 - p) ** 2):
                return 1
            else:
                return 0
        else:  # CASE A I
            if r1 < p:
                return 1
            else:
                return 0
    else:
        if b == 1:  # CASE A I
            if r1 < p:
                return 1
            else:
                return 0
        else:  # CASE I I
            return 0


@njit
def simula_DP(p, L0, T):
    L = np.copy(L0)
    N = len(L0)  # Number of columns (system width)
    H = np.zeros((T + 1, N))  # T+1 is the number of rows (number of timesteps)
    H[0] = np.copy(L)
    d = 1
    cont = 1
    for t in range(T):  # Loop through the timesteps
        L2 = np.zeros(len(L))  # Initialize a new row
        if d == 1:  # MOVE TO THE RIGHT
            d = 2
            x = L[:-1]  # The first row (L0) without the rightmost element
            y = L[1:]  # The first row (L0) without the leftmost element
            r = np.random.random(len(x))  # A numpy array of length (columns - 1) that contains random floats
            L2[:-1] = logic2(x, y, r, p)
            if L[-1] == 1:
                if np.random.random() < p:
                    L2[-1] = 1
            else:
                L2[-1] = L[-1]
            L = np.copy(L2)
        else:  # MOVE TO THE LEFT
            d = 1
            x = L[:-1]
            y = L[1:]
            r = np.random.random(len(x))
            L2[1:] = logic2(x, y, r, p)
            if L[1] == 1:
                if np.random.random() < p:
                    L2[1] = 1
            else:
                L2[1] = L[1]
            L = np.copy(L2)
        H[cont] = np.copy(L)
        cont += 1

    return H


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Utility & plotting functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def groups(lst):
    zero_groups = []
    ones_groups = []

    start = None
    for i, num in enumerate(lst):
        if num == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                zero_groups.append(lst[start:i])
                start = None
    if start is not None:
        zero_groups.append(lst[start:])

    start = None
    for i, num in enumerate(lst):
        if num == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                ones_groups.append(lst[start:i])
                start = None
    if start is not None:
        ones_groups.append(lst[start:])

    len_zero_list = [len(subgroup) for subgroup in zero_groups]
    len_ones_list = [len(subgroup) for subgroup in ones_groups]

    return len_zero_list, len_ones_list


def iter_loops(sz, p):
    L0 = np.ones(sz)
    p = p

    # Run the simulation and
    a = simula_DP(p, L0, int(sz * 0.75))  # third argument is int(size * 0.75)

    # Get the statistics on either the cluster widths or the gap widths
    zero_group_lens, ones_group_lens = groups(list(a[-1]))  # Get only the last row

    return max(zero_group_lens), max(ones_group_lens)


def sys_loops(sz, runs, p):
    L0 = np.ones(sz)
    runs = runs
    p = p

    # Do 'runs' runs for each system size, retrieving the max cluster and gap sizes each time. Take the average for both.
    res = Parallel(n_jobs=-1)(delayed(iter_loops)(sz, p) for run in range(runs))
    gap_group_lens_mega_list, cluster_group_lens_mega_list = zip(*res)
    gap_group_lens_mega_list, cluster_group_lens_mega_list = list(gap_group_lens_mega_list), list(cluster_group_lens_mega_list)

    # Calculate the average max cluster & gap size for the system size:
    cluster_max_mean = np.mean(cluster_group_lens_mega_list)
    gap_max_mean = np.mean(gap_group_lens_mega_list)

    return cluster_max_mean, gap_max_mean


@mpltex.aps_decorator
def plot_maxes(sys_sizes, step_size, mean_cluster_maxes, mean_gap_maxes):
    # Generate and plot the CCDFs
    for option in ['clusters', 'gaps']:
        if option == 'clusters':
            plt.close('all')
            plt.plot(sys_sizes, mean_cluster_maxes)
            plt.xscale('log')
            plt.ylabel('Avg. max cluster size')
            plt.xlabel('System size')
            plt.title(rf'Max cluster size for sys widths {str(sys_sizes[0])} - {str(sys_sizes[-1])}, $T=0.75 \times N$')
            plt.tight_layout(pad=0.6)
            plt.savefig(f'plots/Max Sizes/{option}_max_sizes_{str(sys_sizes[0])}-{str(sys_sizes[-1])}-{str(step_size)}.png')
        else:
            plt.close('all')
            plt.loglog(sys_sizes, mean_gap_maxes)
            plt.ylabel('Avg. max gap size')
            plt.xlabel('System size')
            plt.title(rf'Max gap size for sys widths {str(sys_sizes[0])} - {str(sys_sizes[-1])}, $T=0.75 \times N$')
            plt.tight_layout(pad=0.6)
            plt.savefig(f'plots/Max Sizes/{option}_max_sizes_{str(sys_sizes[0])}-{str(sys_sizes[-1])}-{str(step_size)}.png')

    return 'temp'


def main(**kwargs):
    # Default values of all the parameters
    p = 0.6447  # Critical value is p = 0.6447
    step = 50
    sys_size_range = [i for i in range(100, 1000, step)]
    runs = 10

    # Unpack the parameters
    for key, value in kwargs.items():
        if key == 'max_size_range':
            sys_size_range = value
        elif key == 'bond_prob':
            p = value
        elif key == 'max_size_runs':
            runs = value
        elif key == 'max_size_step':
            step = value

    # # Loop through all the system sizes and get the average max cluster & gap sizes for each system size at T = 0.75 * N
    # res = Parallel(n_jobs=-1)(delayed(sys_loops)(size, runs, p) for size in sys_size_range)
    # cluster_maxes, gap_maxes = zip(*res)

    cluster_maxes = []
    gap_maxes = []
    for size in sys_size_range:
        cluster_max, gap_max = sys_loops(size, runs, p)
        cluster_maxes.append(cluster_max)
        gap_maxes.append(gap_max)

    _ = plot_maxes(sys_size_range, step, cluster_maxes, gap_maxes)

    return 'temp'


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Control section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

if __name__ == '__main__':
    bond_prob = 0.6447
    max_size_lower = 100
    max_size_upper = 5025
    max_size_step = 25
    max_size_range = [i for i in range(max_size_lower, max_size_upper, max_size_step)]
    max_size_runs = 200

    start = time.time()
    main(bond_prob=bond_prob, max_size_range=max_size_range, max_size_runs=max_size_runs, max_size_step=max_size_step)
    end = time.time()
    print("Time taken [s]: ", end-start)
