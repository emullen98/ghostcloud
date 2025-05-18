"""
Created Oct 01 2024
Updated Oct 01 2024

(IN CLUSTER)
Load in a directory's collection of .txt files with exponents in the first line listed as:
kappa_area, kappa_perim, d_f
"""
import numpy as np
import os
import sys

load_dir = sys.argv[1]
save_dir = sys.argv[2]
out_name = sys.argv[3]


# Updated Oct 01 2024
def main(load_loc=None, save_loc=None, output_name=None):
    """

    :param load_loc:
    :param save_loc:
    :param output_name:
    :return:
    """
    if load_loc is None or save_loc is None or output_name is None:
        print('Please enter a directory and an output file name.')

    file_names = [file for file in os.listdir(load_loc) if file.endswith('.txt')]
    rows = len(file_names)
    columns = 3
    exp_arr_master = np.empty(shape=(rows, columns))
    for i in range(rows):
        exp_arr = np.loadtxt(f'{load_loc}/{file_names[i]}')
        exp_arr_master[i] = exp_arr

    np.savetxt(f'{save_loc}/{output_name}.txt', exp_arr_master)

    return 'temp'


if __name__ == '__main__':
    main(load_dir, save_dir, out_name)
