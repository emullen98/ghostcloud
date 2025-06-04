"""
Created May 27 2025
Updated Jun 04 2025

Utility functions for working with percolation models in the cluster
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from scipy.special import gamma, kv
from scipy.fftpack import fftn, ifftn


# ============================
# These functions should NOT be called directly
# ============================


def _compute_2d_spectral_density(q: np.ndarray, gamma_exp: float) -> np.ndarray:    
    """
    Helper function to compute the 2D spectral density S(q) for a given q and gamma exponent

    See Makse et al. (1996)

    Parameters
    ----------
    q : np.ndarray
        Fourier space coordinates (magnitude)
    gamma_exp : float

    Returns
    -------
    np.ndarray
        Spectral density S(q) computed for the given q and gamma exponent
    """
    beta = (gamma_exp - 2) / 2
    q = np.where(q == 0, 1e-10, q)
    prefactor = (2 * np.pi) / gamma(beta + 1)
    S_q = prefactor * (q / 2) ** beta * kv(beta, q)

    return np.nan_to_num(np.real(S_q), nan=0.0, posinf=0.0, neginf=0.0)


# ============================
# These functions SHOULD be called directly
# ============================


def fill_and_label_lattice(arr: np.ndarray, rem_border_clusters: bool) -> tuple[np.ndarray, int]:
    """
    Fills and labels an input lattice

    Parameters
    ----------
    arr : ndarray
        Input lattice
    rem_border_clusters : bool
        Whether to remove clusters that touch the boundary of the lattice

    Returns
    -------
    labeled_filled_lattice : ndarray of ints
        Labeled & filled version of input lattice
    num_features : int
        Number of clusters after filling & labeling the lattice
    """
    my_arr = binary_fill_holes(arr)

    if rem_border_clusters:
        labeled_filled_lattice, _ = label(my_arr)  # Number of features needs to be calculated AFTER removing boundary-touching clusters, if applicable

        t_row = labeled_filled_lattice[0, :]
        b_row = labeled_filled_lattice[-1, :]
        l_col = labeled_filled_lattice[:, 0]
        r_col = labeled_filled_lattice[:, -1]

        unique_border_labels = np.unique(np.concatenate((t_row, b_row, l_col, r_col)))
        unique_border_labels = unique_border_labels[unique_border_labels > 0]

        mask = np.isin(labeled_filled_lattice, unique_border_labels)
        labeled_filled_lattice[mask] = 0

        labeled_filled_lattice = (labeled_filled_lattice > 0).astype(int)
        labeled_filled_lattice, num_features = label(labeled_filled_lattice)
    else:
        labeled_filled_lattice, num_features = label(my_arr)

    return labeled_filled_lattice, num_features


def label_lattice(arr: np.ndarray, rem_border_clusters: bool) -> tuple[np.ndarray, int]:
    """
    Labels (but does not fill) an input lattice

    Parameters
    ----------
    arr : ndarray
        Input lattice
    rem_border_clouds : bool
        Whether to remove clusters that touch the boundary of the lattice

    Returns
    -------
    labeled_binary_lattice : np.ndarray
        Labeled version of lattice
    num_features : int
        Number of clusters after labeling the lattice
    """
    if rem_border_clusters:
        labeled_lattice, _ = label(arr)

        t_row = labeled_lattice[0, :]
        b_row = labeled_lattice[-1, :]
        l_col = labeled_lattice[:, 0]
        r_col = labeled_lattice[:, -1]

        unique_border_labels = np.unique(np.concatenate((t_row, b_row, l_col, r_col)))
        unique_border_labels = unique_border_labels[unique_border_labels > 0]

        mask = np.isin(labeled_lattice, unique_border_labels)
        labeled_lattice[mask] = 0

        labeled_lattice = (labeled_lattice > 0).astype(int)
        labeled_lattice, num_features = label(labeled_lattice)
    else:
        labeled_lattice, num_features = label(arr)

    return labeled_lattice, num_features


def timestep_dp(arr: np.ndarray, prob: float, lx: int, ly: int) -> np.ndarray:
    """
    Evolves an input lattice by one timestep according to directed percolation (DP)

    Logic made efficient by Sid Mansingh on Nov 30 2023

    Parameters
    ----------
    arr : np.ndarray
    prob : float
        Bond probability
    lx : int
    ly : int

    Returns
    -------
    slice_new : np.ndarray
    """
    prob1 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob2 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    prob3 = np.random.choice(2, (ly, lx), p=[1 - prob, prob]).astype('int8')
    slice2 = np.roll(arr, shift=(0, -1), axis=(0, 1)).astype('int8')
    slice3 = np.roll(arr, shift=(-1, 0), axis=(0, 1)).astype('int8')
    slice_new = prob1 * arr + prob2 * slice2 + prob3 * slice3
    slice_new = (slice_new > 0).astype('int8')

    return slice_new


def make_lattice_dp(size: int = 100, p: float = 0.381, end_time: int = 7, fill_holes: bool = True, include_diags: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a 2D lattice evolved for some amount of time according to directed percolation

    Parameters
    ----------
    size : int, default=100
        Linear system size
    p : float, default=0.381
        Bond probability
    end_time : int, default=7
        Number of times to evolve the lattice
    fill_holes : bool, default=True
    include_diags : bool, default=True
        Gives the labeling procedure (i.e., the convolution operation) an 8-connected structure if set to True

    Returns
    -------
    labeled_filled_lattice : np.ndarray
    filled_lattice : np.ndarray
    lattice : np.ndarray
    """
    lx = ly = size

    if include_diags:
        m = np.ones((3, 3))
    else:
        m = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    lattice = np.ones((ly, lx), dtype='int8')
    for i in range(end_time):
        lattice = timestep_dp(lattice, p, lx, ly)  # Timestep returns the lattice as an array of 8-bit (1-byte) integers

    # Not explicity assigning a type to 'labeledArray'.
    # The largest label will dictate the type of the array, so for very large systems this will likely be an array of 64-bit integers.
    if fill_holes:
        filled_lattice = binary_fill_holes(lattice).astype('int8')
        labeled_filled_lattice, _ = label(filled_lattice, structure=m)
    else:
        filled_lattice = lattice
        labeled_filled_lattice, _ = label(filled_lattice, structure=m)

    return labeled_filled_lattice, filled_lattice, lattice


def generate_2d_correlated_field(L, gamma_exp, unit_normalize, seed=None):
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
    S_q = _compute_2d_spectral_density(q, gamma_exp)

    # Generate uncorrelated Gaussian noise
    noise = np.random.normal(0, 1, (L, L)) + 1j * np.random.normal(0, 1, (L, L))

    # Apply filter in Fourier domain
    hq = fftn(noise) * np.sqrt(S_q)
    field = np.real(ifftn(hq))

    # Normalize to [0, 1]
    if unit_normalize:
        field -= np.min(field)
        field /= np.max(field)
    
    return field
