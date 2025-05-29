"""
Created May 17 2025
Updated May 22 2025

(IN CLUSTER)
Computes important quantities for a single PNG cloud image and stores all of this in a .csv file formatted as shown below

Thresh | Occ. frac. | # of clusters | g(0) | g(1) | ...
1      | ...        | ...           | ...  | ...  | ...
...    | ...        | ...           | ...  | ...  | ...
255    | ...        | ...           | ...  | ...  | ...
"""
import sys
import os
import re
import numpy as np
from _nlc_image_utils import get_corr_func, label_image, fill_and_label_image, set_thread_count

id_num = int(sys.argv[1])
fill_holes = sys.argv[2].lower() == "true"
corr_func_frac = float(sys.argv[3])
thresh_min = int(sys.argv[4])
thresh_max = int(sys.argv[5])
thread_count = int(sys.argv[6])

# Tells Numba how many threads to use
set_thread_count(thread_count)

rem_border_clouds = True

png_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images/useful'
png_file_names = [file for file in os.listdir(png_directory) if file.endswith('.png')]  # These are the PNG file names WITHOUT parent directory path
png_file_name = next((name for name in png_file_names if re.search(f'id={id_num}.png$', name)), None)

if fill_holes:
    check_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/fill'
else:
    check_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/no_fill'
completed_file_names = [file.split('_')[0] for file in os.listdir(check_directory) if file.endswith('.csv')]  # These are the names of the files up through the date and random number

if not png_file_name.split('_')[0] in completed_file_names:
    save_arr = []
    if fill_holes:
        save_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/fill'
        save_file_name = png_file_name.split('.')[0] + f'_corrfunc_fill_threshrange={thresh_min}-{thresh_max}.csv'
        for thresh in np.linspace(start=thresh_min, stop=thresh_max, num=(thresh_max - thresh_min) + 1).astype(int):
            processed_arr, num_features = fill_and_label_image(path=f'{png_directory}/{png_file_name}', thresh=thresh, rem_border_clouds=rem_border_clouds)
            occup_prob = np.sum(processed_arr > 0) / processed_arr.size
            w, h = processed_arr.shape
            max_distance = int(np.hypot(w, h))
            if occup_prob == 1.0 or occup_prob == 0.0:  # If the lattice is completely full or empty, enter -1 for everything and avoid having to compute correlation function
                save_arr.append([-1.0 for i in range(max_distance + 4)])
            else:
                corr_func = get_corr_func(processed_lattice=processed_arr, num_features=num_features, max_dist=max_distance, frac=corr_func_frac)
                temp_arr = [float(thresh), float(occup_prob), float(num_features), *list(corr_func)]
                save_arr.append(temp_arr)

    else:
        save_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/no_fill'
        save_file_name = png_file_name.split('.')[0] + f'_corrfunc_nofill_threshrange={thresh_min}-{thresh_max}.csv'
        for thresh in np.linspace(start=thresh_min, stop=thresh_max, num=(thresh_max - thresh_min) + 1).astype(int):
            processed_arr, num_features = label_image(path=f'{png_directory}/{png_file_name}', thresh=thresh, rem_border_clouds=rem_border_clouds)
            occup_prob = np.sum(processed_arr > 0) / processed_arr.size
            w, h = processed_arr.shape
            max_distance = int(np.hypot(w, h))
            if occup_prob == 1.0 or occup_prob == 0.0:
                save_arr.append([-1.0 for i in range(max_distance + 4)])
            else:
                corr_func = get_corr_func(processed_lattice=processed_arr, num_features=num_features, max_dist=max_distance, frac=corr_func_frac)
                temp_arr = [float(thresh), float(occup_prob), float(num_features), *list(corr_func)]
                save_arr.append(temp_arr)

    header = "thresh,occup_prob,num_clouds," + ",".join(f"g({i})" for i in range(max_distance + 1))
    np.savetxt(f'{save_directory}/{save_file_name}', np.array(save_arr), delimiter=',', header=header, comments='')
