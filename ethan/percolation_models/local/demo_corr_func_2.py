import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.ndimage import binary_fill_holes, label, find_objects
import timeit


# @njit(parallel=True)
def new_corr_func(cloud, cloud_shape, max_distance, frac=None):
    coords = np.indices(cloud_shape).reshape(len(cloud_shape), -1).T

    print(cloud)

    if frac is None:
        pass
    else:
        idxs = np.random.choice(a=np.arange(len(coords)), size=int(frac * len(coords)), replace=False)
        new_coords = np.empty(shape=(len(idxs), coords.shape[1]), dtype=coords.dtype)

        for i in range(len(idxs)):
            new_coords[i] = coords[idxs[i]]
        coords = new_coords

    same_count = np.zeros(max_distance + 1)
    total_count = np.zeros(max_distance + 1)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            inner_coord = coords[i]
            outer_coord = coords[j]
            inner_val = cloud[inner_coord[0], inner_coord[1]]
            outer_val = cloud[outer_coord[0], outer_coord[1]]
            if inner_val == 0 and outer_val == 0:
                pass
            elif (inner_val == 0 and outer_val != 0) or (inner_val != 0 and outer_val == 0):
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                total_count[r] += 1
            elif inner_val != 0 and outer_val != 0:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                total_count[r] += 2
                same_count[r] += 2

    print(same_count)
    print(total_count)

    return same_count, total_count


# @njit(parallel=True)
def old_corr_func(labeled_lattice, frac=None):
    lattice_shape = labeled_lattice.shape
    coords = np.indices(lattice_shape).reshape(len(lattice_shape), -1).T
    max_distance = int(round(np.sqrt(np.sum(np.array(lattice_shape) ** 2))))

    if frac is None:
        pass
    else:
        idxs = np.random.choice(a=np.arange(len(coords)), size=int(frac * len(coords)), replace=False)
        new_coords = np.empty(shape=(len(idxs), coords.shape[1]), dtype=coords.dtype)

        for i in range(len(idxs)):
            new_coords[i] = coords[idxs[i]]
        coords = new_coords

    correlation_function = np.zeros(max_distance + 1)
    correlation_function[0] = 1
    counts = np.zeros(max_distance + 1)

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            inner_coord = coords[i]
            outer_coord = coords[j]
            inner_site = labeled_lattice[inner_coord[0], inner_coord[1]]
            outer_site = labeled_lattice[outer_coord[0], outer_coord[1]]
            if inner_site == 0 and outer_site == 0:
                pass
            elif (inner_site == 0 and outer_site != 0) or (inner_site != 0 and outer_site == 0):
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 1
            elif (inner_site != 0 and outer_site != 0) and inner_site != outer_site:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 2
            elif (inner_site != 0 and outer_site != 0) and inner_site == outer_site:
                dx = inner_coord[0] - outer_coord[0]
                dy = inner_coord[1] - outer_coord[1]
                r_squared = dx * dx + dy * dy
                r = round(np.sqrt(r_squared))
                counts[r] += 2
                correlation_function[r] += 2

    for r in range(max_distance + 1):
        if counts[r] > 0:
            correlation_function[r] /= counts[r]

    possible_distances = np.arange(0, max_distance + 1)

    return possible_distances, correlation_function


if __name__ == '__main__':
    prob = 0.405
    lx = ly = 7

    my_arr = np.random.choice([0, 1], size=(ly, lx), p=[1 - prob, prob])
    my_max_distance = int(round(np.sqrt(np.sum(np.array(my_arr.shape) ** 2))))

    my_arr = binary_fill_holes(my_arr).astype(int)
    my_arr, num_features = label(my_arr)

    slices = find_objects(my_arr)

    count_same = np.zeros(my_max_distance + 1)
    count_total = np.zeros(my_max_distance + 1)
    for k in range(num_features):
        cloud_slice = my_arr[slices[k]]
        cloud_slice = np.where(cloud_slice == k + 1, k + 1, 0)
        temp_same_count, temp_total_count = new_corr_func(cloud=cloud_slice, cloud_shape=cloud_slice.shape, max_distance=my_max_distance)
        count_same += temp_same_count
        count_total += temp_total_count

    new_dists = np.arange(0, my_max_distance + 1)
    new_cf = np.divide(count_same, count_total, out=np.zeros_like(count_same, dtype=float), where=count_total != 0)
    new_cf[0] = 1

    old_dists, old_cf = old_corr_func(labeled_lattice=my_arr)

    # # Plot old CF
    # plt.plot(old_dists, old_cf, label='old')
    # # Plot new CF
    # plt.plot(new_dists, new_cf, label='new')
    #
    # plt.legend()
    # plt.show()

    # def run_old_jit():
    #     old_corr_func_jit(my_arr)
    #
    # def run_old_jit_nopython():
    #     old_corr_func_jit_nopython(my_arr)
    #
    # # 2) time each one with timeit.repeat
    # reps = 5
    # nloops = 3   # number of repetitions within each timing
    #
    # t_old_jit_nopython = min(timeit.repeat("run_old_jit_nopython()",
    #                setup="from __main__ import run_old_jit_nopython",
    #                repeat=reps, number=nloops)) / nloops
    #
    # t_old_jit = min(timeit.repeat("run_old_jit()",
    #                setup="from __main__ import run_old_jit",
    #                repeat=reps, number=nloops)) / nloops
    #
    #
    # # print(f"Old:     {t_old:.4f} s per call")
    # print(f"Old (jit):    {t_old_jit:.4f} s per call")
    # # print(f"New:  {t_new:.4f} s per call")
    # print(f"Old (jit, nopython): {t_old_jit_nopython:.4f} s per call")

