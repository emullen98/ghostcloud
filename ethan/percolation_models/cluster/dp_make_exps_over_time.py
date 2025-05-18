"""
Created Jun 09 2024
Updated Sep 30 2024

(IN CLUSTER)
Saves the perimeters & areas of clusters in a single lattice for each of some list of timesteps.
"""
import sys
from scipy.ndimage import label, binary_fill_holes
from helper_scripts.timestep import timestep
from helper_scripts.get_pa import get_pa
from helper_scripts.find_powerlaw import *
from helper_scripts.lsq_fit import *
from helper_scripts.logbinning import *

job_id = sys.argv[1]
task_id = sys.argv[2]
size = int(sys.argv[3])
prob = float(sys.argv[4])

lx = ly = size

start = 3
end = 14

# The initialized lattice is timestep 0
lattice = np.ones((ly, lx), dtype='int8')
for end_time in range(1, end):
    lattice = timestep(lattice, prob, lx, ly)
    if end_time in list(range(start, end)):
        filledLattice = binary_fill_holes(lattice).astype('int8')
        labeledArray, numFeatures = label(filledLattice, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

        perims, areas = get_pa(labeledArray)
        perims = perims[~np.isnan(perims)]
        areas = areas[~np.isnan(areas)]

        kappa_perim = find_pl(perims, xmin=2.8*10**2, xmax=2.8*10**4)
        kappa_area = find_pl(areas, xmin=2.1*10**2, xmax=2*10**5)

        area_bin, perim_bin = logbinning(areas, perims, numBins=50)[:2]
        d_f = fit(np.log10(area_bin), np.log10(perim_bin), xmin=np.log10(210))[0][0]
        d_f *= 2

        np.savetxt(f'./results/dp/s={size}/end_time={end_time}/dp_exps_s={size}_p={prob}_end={end_time}_job={job_id}_{task_id}.txt', np.array([kappa_area, kappa_perim, d_f]))
