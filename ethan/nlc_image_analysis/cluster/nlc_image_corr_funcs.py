"""
Created May 17 2025
Updated May 31 2025

(IN CLUSTER)
Computes important quantities for a single PNG cloud image and stores all of this in a .csv file formatted as shown below

Thresh | Occ. frac. (w/ border clouds) | # of clusters (w/ border clouds) | Occ. frac. (w/o border clouds)| # of clusters (w/o border clouds) | g(0) | g(1) | ...
1      | ...                           | ...                              | ...                           | ...                               | ...  | ...  | ...
...    | ...                           | ...                              | ...                           | ...                               | ...  | ...  | ...
255    | ...                           | ...                              | ...                           | ...                               | ...  | ...  | ... 
"""
import sys
import os
import re
import numpy as np
from clouds_helpers import get_corr_func, label_image, fill_and_label_image, set_thread_count

# ============================
# Section: Import command-line arguments
# ============================

id_num = int(sys.argv[1])
fill_holes = sys.argv[2].lower() == "true"
corr_func_frac = float(sys.argv[3])
thread_count = int(sys.argv[4])
# Tells Numba how many threads to use
set_thread_count(thread_count) 

# ============================
# Section: Set parameters & file paths
# ============================

# Goes from 10 to 250 in increments of 10
thresholds = range(10, 260, 10) 

png_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images/xsc1/good'
# These are the PNG file names WITHOUT parent directory path
png_file_names = [file for file in os.listdir(png_directory) if file.endswith('.png')]  
png_file_name = next((name for name in png_file_names if re.search(f'id={id_num}.png$', name)), None)

if fill_holes:
    check_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/fill'
else:
    check_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/no_fill'
# These are the names of the files up through the date and random number
completed_file_names = [file.split('_')[0] for file in os.listdir(check_directory) if file.endswith('.csv')]  

# ============================
# Section: Main code
# ============================

if not png_file_name.split('_')[0] in completed_file_names:
    save_arr = []
    if fill_holes:
        save_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/fill'
        save_file_name = png_file_name.split('.')[0] + f'_corrfunc_fill.csv'
        for thresh in thresholds:
            processed_arr_with_border_clouds, num_features_with_border_clouds = fill_and_label_image(path=f'{png_directory}/{png_file_name}', thresh=thresh, rem_border_clouds=False)
            occup_prob_with_border_clouds = np.sum(processed_arr_with_border_clouds > 0) / processed_arr_with_border_clouds.size
            
            processed_arr, num_features = fill_and_label_image(path=f'{png_directory}/{png_file_name}', thresh=thresh, rem_border_clouds=True)
            occup_prob = np.sum(processed_arr > 0) / processed_arr.size
            
            w, h = processed_arr.shape
            max_distance = int(np.hypot(w, h))
            # The +6 accounts for the five image parameters in temp_arr and the correlation function values (which go from 0 to max_distance, inclusive)
            num_columns = max_distance + 6  
            # If the lattice (without border-touching clouds) is completely full or empty, enter -1 for everything and avoid having to compute correlation function
            if occup_prob == 1.0 or occup_prob == 0.0:  
                save_arr.append([-1.0 for i in range(num_columns)])
            else:
                corr_func = get_corr_func(processed_lattice=processed_arr, num_features=num_features, max_dist=max_distance, frac=corr_func_frac)
                temp_arr = [float(thresh), float(occup_prob_with_border_clouds), float(num_features_with_border_clouds), float(occup_prob), float(num_features), *list(corr_func)]
                save_arr.append(temp_arr)

    else:
        save_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/no_fill'
        save_file_name = png_file_name.split('.')[0] + f'_corrfunc_nofill.csv'
        for thresh in thresholds:
            processed_arr_with_border_clouds, num_features_with_border_clouds = label_image(path=f'{png_directory}/{png_file_name}', thresh=thresh, rem_border_clouds=False)
            occup_prob_with_border_clouds = np.sum(processed_arr_with_border_clouds > 0) / processed_arr_with_border_clouds.size
            
            processed_arr, num_features = label_image(path=f'{png_directory}/{png_file_name}', thresh=thresh, rem_border_clouds=True)
            occup_prob = np.sum(processed_arr > 0) / processed_arr.size

            w, h = processed_arr.shape
            max_distance = int(np.hypot(w, h))
            num_columns = max_distance + 6
            if occup_prob == 1.0 or occup_prob == 0.0:
                save_arr.append([-1.0 for i in range(num_columns)])
            else:
                corr_func = get_corr_func(processed_lattice=processed_arr, num_features=num_features, max_dist=max_distance, frac=corr_func_frac)
                temp_arr = [float(thresh), float(occup_prob_with_border_clouds), float(num_features_with_border_clouds), float(occup_prob), float(num_features), *list(corr_func)]
                save_arr.append(temp_arr)

    header = "thresh,occup_prob_with_bcs,num_clouds_with_bcs,occup_prob_no_bcs,num_clouds_no_bcs," + ",".join(f"g({i})" for i in range(max_distance + 1))
    np.savetxt(f'{save_directory}/{save_file_name}', np.array(save_arr), delimiter=',', header=header, comments='')
