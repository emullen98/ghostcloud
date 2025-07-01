import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from cloud_utils import *

# Parameters for correlated percolation
width = 256
height = 256
gamma_exp = 0.2
p_val = 0.5927
seed = 42

# Generate the correlated percolation lattice
lattice = generate_correlated_percolation_lattice(
    width=width,
    height=height,
    gamma_exp=gamma_exp,
    p_val=p_val,
    seed=seed
)

# Show the lattice as an image
plt.figure(figsize=(6, 6))
plt.imshow(lattice, cmap='Greys_r', origin='lower')
plt.title(f"Correlated Percolation Lattice\nγ={gamma_exp}, p={p_val}")
plt.axis('off')
plt.show()