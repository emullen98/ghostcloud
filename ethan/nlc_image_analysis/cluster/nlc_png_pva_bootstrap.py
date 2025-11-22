"""
Created Nov 22 2025
Updated Nov 22 2025

(IN CLUSTER)
-- Conducts bootstrap fits to logbinned perimeter vs. area data from NLC PNG images
-- Ethan guestimates that a job array of 400 jobs, each doing 25 bootstrap runs, will be good enough
-- Must specify min_area for fitting and logbin_count for logbinning
-- Current procedure:
    1. Extract only clouds whose area > min_area
    2. Do bootstrap resampling with replacement of these clouds
    3. Logbin the resampled data into logbin_count bins
    4. Fit the log10(perimeter) vs. log10(area) data to extract fractal dimension
    5. Repeat steps 2-4 for bootstrap_runs times
-- In future, may want to first bootstrap resample, then logbin to whole dataset, then fit the logbinned data using some min area in the fit function.
    -- I'm not sure if there's any benefit to this, but there is certainly a difference between this and the current method.
    -- If anything, it slows the whole process down by bootstrapping over a larger dataset first.
-- File information:
    -- Input: CSV files from Manas (one file per threshold type and hours for a connectivity pair choice)
    -- Output: Numpy files containing arrays of fractal dimensions and their errors from each bootstrap run
        -- Will have to combine all 400 files once they're done
"""
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/projects/illinois/eng/physics/dahmen/mullen')
import helper_scripts as hs

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Load & set parameters (see docstring for details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

jobid = int(sys.argv[1])
taskid = int(sys.argv[2])
min_area = int(sys.argv[3])
logbin_count = int(sys.argv[4])
bootstrap_runs = int(sys.argv[5])
thresh_type = str(sys.argv[6]) 
hours = str(sys.argv[7])
load_path = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/clouds_paper_data/png_csvs'
save_path = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/clouds_paper_data/png_pva_fits'

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

for file_name in os.listdir(load_path):
    if file_name.endswith('.csv') and f'{thresh_type}_{hours}' in file_name:
        cur_file_name = file_name.split('.csv')[0]  # Use this when saving later
        data_file = os.path.join(load_path, file_name)
        break

df = pd.read_csv(data_file)

areas = df['area_px'].to_numpy()
perims_hull = df['perim_hull_edge'].to_numpy()
perims_acce = df['perim_accessible_edge'].to_numpy()

good_idxs = np.where(areas > min_area)[0]
areas = areas[good_idxs]
perims_hull = perims_hull[good_idxs]
perims_acce = perims_acce[good_idxs]

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Do bootstrapping to extract fractal dimensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

frac_dims_hull = np.zeros(bootstrap_runs)
frac_dims_hull_err = np.zeros(bootstrap_runs)
frac_dims_acce = np.zeros(bootstrap_runs)
frac_dims_acce_err = np.zeros(bootstrap_runs)

for run in range(bootstrap_runs):
    # Resample with replacement
    indices = np.random.randint(0, len(areas), len(areas))
    resampled_areas = areas[indices]
    resampled_perims_hull = perims_hull[indices]
    resampled_perims_acce = perims_acce[indices]

    # Logbin data
    logbin_areas_hull, logbin_perims_hull, _ = hs.logbinnning(resampled_areas, 
                                                              resampled_perims_hull, 
                                                              logbin_count)
    logbin_areas_acce, logbin_perims_acce, _ = hs.logbinnning(resampled_areas, 
                                                              resampled_perims_acce, 
                                                              logbin_count)

    # Fit data and extract fractal dimensions
    params_hull, errs_hull, _ = hs.fit(xdata=np.log10(logbin_areas_hull), 
                                       ydata=np.log10(logbin_perims_hull))
    frac_dims_hull[run] = params_hull[0]
    frac_dims_hull_err[run] = errs_hull[0]
    params_acce, errs_acce, _ = hs.fit(xdata=np.log10(logbin_areas_acce), 
                                       ydata=np.log10(logbin_perims_acce))
    frac_dims_acce[run] = params_acce[0]
    frac_dims_acce_err[run] = errs_acce[0]

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Save data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

np.save(file=f'{save_path}/png_bootstrap_df_{thresh_type}_{hours}_amin={min_area}_logbins={logbin_count}_taskid={taskid}.npy', arr=np.array([frac_dims_hull, frac_dims_hull_err, frac_dims_acce, frac_dims_acce_err]))
