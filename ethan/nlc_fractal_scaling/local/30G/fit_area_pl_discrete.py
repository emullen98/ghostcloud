"""
Created Oct 18 2025
Updated Oct 18 2025

(LOCAL)
-- Uses the original CSV data containing NLC areas & perimeters.
-- Performs discrete power-law fitting on area data for a desired albedo threshold.
-- Unlike the Monte Carlo method, here we provide explicit boundaries for the scaling regime
"""
import sys
sys.path.append('/Users/emullen98/Desktop')
sys.path.append('/Users/emullen98/Desktop/ghostcloud/ethan/nlc_fractal_scaling/local/')
from _get_data import get_data
import helper_scripts as hs

thresh = '30G'  # Desired albedo threshold
csv_loc = f'/Users/emullen98/Downloads/og_clouds/{thresh}_v4.csv'

areas, perims = get_data(csvloc=csv_loc)
areas_exp, log_likelihood = hs.find_pl_discrete(x=areas, xmin=100, xmax=max(areas), stepsize=1)
print(f'Exponent when xmax = max(data): {areas_exp}') 
areas_exp, log_likelihood = hs.find_pl_discrete(x=areas, xmin=100, xmax=1000, stepsize=1)
print(f'Exponent when xmax = 1000: {areas_exp}')  