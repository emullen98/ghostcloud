"""
Created May 17 2025
Updated May 17 2025

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
from _nlc_image_utils import corr_func, label_image, fill_and_label_image

id_num = int(sys.argv[1])
fill_holes = sys.argv[2].lower() == "true"
corr_func_frac = float(sys.argv[3])

if fill_holes:
    save_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/fill'
else:
    save_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_corr_funcs/no_fill'

png_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images/useful'
file_names = [file for file in os.listdir(png_directory) if file.endswith('.png')]  # These are file names WITHOUT parent directory path
file_name = next((name for name in file_names if re.search(f'id={id_num}.png$', name)), None)

save_arr = []
if fill_holes:
    save_file_name = file_name.split('_')[0] + '_corrfunc_fill.csv'
    for thresh in np.linspace(start=1, stop=255, num=255).astype(int):
        processed_arr, num_features = fill_and_label_image(path=f'{png_directory}/{file_name}', thresh=thresh)
        occup_prob = np.sum(processed_arr > 0) / processed_arr.size
        w, h = processed_arr.shape
        max_distance = int(np.hypot(w, h))

        # If the lattice is completely full or empty, enter -1 for everything and avoid having to compute correlation function
        if occup_prob == 1.0 or occup_prob == 0.0:
            save_arr.append([-1.0 for i in range(max_distance + 4)])
        else:
            distances, cf = corr_func(labeled_lattice=processed_arr, frac=corr_func_frac)
            temp_arr = [float(thresh), float(occup_prob), float(num_features)]
            temp_arr.extend(cf)
            save_arr.append(temp_arr)
else:
    save_file_name = file_name.split('_')[0] + '_corrfunc_nofill.csv'
    for thresh in np.linspace(start=1, stop=255, num=255).astype(int):
        processed_arr, num_features = label_image(path=f'{png_directory}/{file_name}', thresh=thresh)
        occup_prob = np.sum(processed_arr > 0) / processed_arr.size
        w, h = processed_arr.shape
        max_distance = int(np.hypot(w, h))

        # If the lattice is completely full or empty, enter -1 for everything and avoid having to compute correlation function
        if occup_prob == 1.0 or occup_prob == 0.0:
            save_arr.append([-1.0 for i in range(max_distance + 4)])
        else:
            distances, cf = corr_func(labeled_lattice=processed_arr, frac=corr_func_frac)
            temp_arr = [float(thresh), float(occup_prob), float(num_features)]
            temp_arr.extend(cf)
            save_arr.append(temp_arr)

header = "thresh,occup_prob,num_clouds," + ",".join(f"g({i})" for i in range(max_distance + 1))
np.savetxt(f'{save_directory}/{save_file_name}', np.array(save_arr), delimiter=',', header=header, comments='')
