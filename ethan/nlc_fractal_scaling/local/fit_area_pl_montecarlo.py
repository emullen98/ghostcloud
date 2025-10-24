"""
Created Oct 18 2025
Updated Oct 19 2025

(LOCAL)
-- Uses the original CSV data containing NLC areas & perimeters.
-- Performs discrete Monte Carlo power-law fitting procedure on area data for a desired albedo threshold.
"""
import sys
sys.path.append('/Users/emullen98/Desktop')
sys.path.append('/Users/emullen98/Desktop/ghostcloud/ethan/nlc_fractal_scaling/local')
from _get_data import get_data
import helper_scripts as hs

thresh = '40G'  # Desired albedo threshold
min_area = 100  # Minimum area for fitting; can be an int or None for default behavior (min_area=15) in get_data()
csv_loc = f'/Users/emullen98/Downloads/og_clouds/{thresh}_v4.csv'

areas, perims = get_data(csvloc=csv_loc, min_area=min_area)
areas_xmin, areas_xmax, areas_exp = hs.find_pl_montecarlo(data=areas, stepsize=1)

print('Area Power-Law Fit Results:')
print(f'  xmin: {areas_xmin}')  
print(f'  xmax: {areas_xmax}')  
print(f'  exponent: {areas_exp}')  