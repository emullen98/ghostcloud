"""
Created May 16 2024
Updated Sep 30 2024

(IN CLUSTER)
Saves the all three exponents for a lattice at t_c = 7.
"""
import sys
from helper_scripts.make_lattice import make_lattice
from helper_scripts.get_pa import get_pa
from helper_scripts.find_powerlaw import *
from helper_scripts.lsq_fit import *
from helper_scripts.logbinning import *

job_id = sys.argv[1]
task_id = sys.argv[2]
size = int(sys.argv[3])
prob = 0.381
end_time = 7

save_loc = f'/home/emullen2/scratch/DirectedPercolation/results/dp/s={size}'

lattice, _, _ = make_lattice(size=size, p=prob, endTime=end_time, fillHoles=True, includeDiags=False)

perims, areas = get_pa(lattice)
perims = perims[~np.isnan(perims)]
areas = areas[~np.isnan(areas)]

kappa_perim = find_pl(perims, xmin=2.8*10**2, xmax=2.8*10**4)
kappa_area = find_pl(areas, xmin=2.1*10**2, xmax=2*10**5)

area_bin, perim_bin = logbinning(areas, perims, numBins=50)[:2]
d_f = fit(np.log10(area_bin), np.log10(perim_bin), xmin=np.log10(210))[0][0]
d_f *= 2

np.savetxt(f'{save_loc}/dp_exps_s={size}_p={prob}_end={end_time}_job={job_id}_{task_id}.txt',
           np.array([kappa_area, kappa_perim, d_f]))
