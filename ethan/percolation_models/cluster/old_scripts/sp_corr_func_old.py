"""
Created Jul 16 2024
Updated Aug 14 2024

(IN CLUSTER)

"""
import numpy as np
from numba import njit, prange
from scipy.ndimage import binary_fill_holes, label
from itertools import combinations_with_replacement
import sys

job_id = int(sys.argv[1])
task_id = int(sys.argv[2])
size = int(sys.argv[3])
prob = float(sys.argv[4])
fill_holes = int(sys.argv[5])


@njit
def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return low


@njit(parallel=True)
def correlation_function(array, possible_distances):
    rows, cols = array.shape
    num_distances = len(possible_distances)
    pair_counts = np.zeros(num_distances, dtype=np.int64)
    same_cluster_counts = np.zeros(num_distances, dtype=np.int64)

    for i1 in prange(rows):
        for j1 in prange(cols):
            for i2 in range(i1, rows):
                for j2 in range(j1 if i1 == i2 else 0, cols):
                    if (i1, j1) != (i2, j2):
                        distance = np.round(np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2), 5)
                        bin_index = binary_search(possible_distances, distance)
                        pair_counts[bin_index] += 1
                        if array[i1, j1] == array[i2, j2] and array[i1, j1] != 0:
                            same_cluster_counts[bin_index] += 1

    correlation = np.zeros(num_distances, dtype=np.float64)
    for k in range(num_distances):
        correlation[k] = same_cluster_counts[k] / pair_counts[k]

    return correlation


sp_lattice = np.random.choice([0, 1], size=(size, size), p=[1 - prob, prob]).astype('int8')
if fill_holes:
    sp_lattice = binary_fill_holes(sp_lattice).astype('int8')
sp_lattice, _ = label(sp_lattice)

combinations = list(combinations_with_replacement(np.arange(size), 2))[1:]
unique_distances = np.unique([np.sqrt(x ** 2 + y ** 2) for x, y in combinations])
unique_distances = np.round(unique_distances, 5)

corr = correlation_function(sp_lattice, unique_distances)

if fill_holes:
    np.save(f'./lattices/s{size}/sp_corr_func_fill_jobid={job_id}_taskid={task_id}.npy', corr)
else:
    np.save(f'./lattices/s{size}/sp_corr_func_nofill_jobid={job_id}_taskid={task_id}.npy', corr)
