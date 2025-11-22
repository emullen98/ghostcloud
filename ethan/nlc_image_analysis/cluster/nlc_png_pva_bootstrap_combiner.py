"""
Created Nov 22 2025
Updated Nov 22 2025

(IN CLUSTER)
-- Combines .npy files output from NLC image PvA bootstrap jobs into single .npy files.
"""
import os
import sys
import numpy as np  

thresh_type = str(sys.argv[1])
hours = str(sys.argv[2])
amin = int(sys.argv[3])
logbin_count = int(sys.argv[4])

# Ensure that we combine only the .npy files that correspond to the arguments provided
files_to_combine = [file_name for file_name in os.listdir('/projects/illinois/eng/physics/dahmen/mullen/Clouds/clouds_paper_data/png_pva_fits') if file_name.endswith('.npy') and f'{thresh_type}_{hours}_amin={amin}_logbins={logbin_count}' in file_name]

df_hull, df_hull_err, df_acce, df_acce_err = [], [], [], []
for file in files_to_combine:
    temp_data = np.load(f'/projects/illinois/eng/physics/dahmen/mullen/Clouds/clouds_paper_data/png_pva_fits/{file}')
    df_hull.extend(list(temp_data[0]))
    df_hull_err.extend(list(temp_data[1]))
    df_acce.extend(list(temp_data[2]))
    df_acce_err.extend(list(temp_data[3]))

df_hull = np.array(df_hull)
df_hull_err = np.array(df_hull_err)
df_acce = np.array(df_acce)
df_acce_err = np.array(df_acce_err)

np.save(f'/projects/illinois/eng/physics/dahmen/mullen/Clouds/clouds_paper_data/png_pva_fits/png_bootstrap_df_{thresh_type}_{hours}_amin={amin}_logbins={logbin_count}.npy', np.array([df_hull, df_hull_err, df_acce, df_acce_err]))
