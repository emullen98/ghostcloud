import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn
from scipy.special import gamma, kv
from scipy.ndimage import binary_fill_holes


def compute_spectral_density(q, gamma_exp):
    beta = (gamma_exp - 2) / 2
    q = np.where(q == 0, 1e-10, q)  # Avoid division by zero at q = 0

    prefactor = (2 * np.pi) / gamma(beta + 1)
    S_q = prefactor * (q / 2)**beta * kv(beta, q)
    S_q = np.nan_to_num(np.real(S_q), nan=0.0, posinf=0.0, neginf=0.0)
    return S_q


def generate_2d_correlated_field(L, gamma_exp, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Frequency space coordinates
    # -1 in reshape() tells numpy to automatically calculate the size of that dimension based on the total number of elements and the other specified dimensions
    # I.e., .reshape(-1, 1) means "Make this a 2D array with as many rows as needed and 1 column"
    kx = np.fft.fftfreq(L).reshape(-1, 1)
    ky = np.fft.fftfreq(L).reshape(1, -1)
    q2 = kx**2 + ky**2
    q = np.sqrt(q2)
    q[0, 0] = 1e-10  # avoid division by zero

    # Spectral density S(q)
    S_q = compute_spectral_density(q, gamma_exp)

    # Generate uncorrelated Gaussian noise
    noise = np.random.normal(0, 1, (L, L)) + 1j * np.random.normal(0, 1, (L, L))

    # Apply filter in Fourier domain
    hq = fftn(noise) * np.sqrt(S_q)
    field = np.real(ifftn(hq))

    # Normalize to [0, 1]
    field -= np.min(field)
    field /= np.max(field)
    return field


def generate_percolation_map(field, p):
    # Threshold the field to simulate site occupation
    return field < p


L = 1024              
gamma_val = 0.2     
p_val = 0.5927    
seed = 42            

field = generate_2d_correlated_field(L, gamma_val, seed)
perc_map = generate_percolation_map(field, p_val)
print(np.sum(perc_map) / (L * L))  # Print the fraction of occupied sites
perc_map_filled = binary_fill_holes(perc_map)  # Fill holes in the percolation map

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(perc_map, cmap='Greys_r', origin='lower')
ax[0].set_title(f"2D Correlated Percolation (γ={gamma_val}, p={p_val})")
ax[0].axis('off')

ax[1].imshow(perc_map_filled, cmap='Greys_r', origin='lower')
ax[1].set_title(f"2D Correlated Percolation (filled) (γ={gamma_val}, p={p_val})")
ax[1].axis('off')

plt.show()