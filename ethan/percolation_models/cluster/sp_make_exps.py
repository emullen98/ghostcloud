"""
Created Jul 03 2024
Updated Oct 01 2024

(IN CLUSTER)
Create site percolation lattice at p_c (for hole filling) and save its three exponents.
"""
import sys
from scipy.ndimage import binary_fill_holes, label
from helper_scripts.get_pa import get_pa
from helper_scripts.find_powerlaw import *
from helper_scripts.logbinning import *
from helper_scripts.lsq_fit import *

jobid = sys.argv[1]
taskid = sys.argv[2]
size = lx = ly = int(sys.argv[3])
prob = float(sys.argv[4])

save_loc = f'/home/emullen2/scratch/DirectedPercolation/results/sp/s={size}'

lattice = np.random.choice([0, 1], size=(ly, lx), p=[1 - prob, prob])

filledLattice = binary_fill_holes(lattice).astype('int8')
m = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
labeledArray, numFeatures = label(filledLattice, structure=m)

perims, areas = get_pa(labeledArray)
perims = perims[~np.isnan(perims)]
areas = areas[~np.isnan(areas)]

kappa_perim = find_pl(perims, xmin=2.8*10**2, xmax=2.8*10**4)
kappa_area = find_pl(areas, xmin=2.1*10**2, xmax=2*10**5)

area_bin, perim_bin = logbinning(areas, perims, numBins=50)[:2]
d_f = fit(np.log10(area_bin), np.log10(perim_bin), xmin=np.log10(210))[0][0]
d_f *= 2

np.savetxt(f'{save_loc}/sp_exps_s={size}_p={prob}_job={jobid}_{taskid}.txt',
           np.array([kappa_area, kappa_perim, d_f]))
