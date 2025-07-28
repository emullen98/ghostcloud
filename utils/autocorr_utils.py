from . import config
from typing import Tuple, List, Literal
import matplotlib.pyplot as plt

# --- Backend Selection ---
xp_backend: Literal["numpy", "cupy"]

if config.USE_GPU:
    try:
        import cupy as cp
        xp = cp
        xp_backend = "cupy"
        print("Using CuPy for GPU acceleration.")
    except ImportError as e:
        raise ImportError("CuPy is required for GPU support but is not installed.") from e
else:
    import numpy as np
    xp = np
    xp_backend = "numpy"
    print("Using NumPy for CPU operations.")

def pad_image(image: xp.ndarray, pad: int) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Pad a binary cropped image with zeros and generate a cloud mask.

    Parameters:
        image (xp.ndarray): 2D binary array representing the cropped cloud.
        pad (int): Number of pixels to pad on each side of the image.

    Returns:
        Tuple[xp.ndarray, xp.ndarray]:
            - Padded image of shape (H + 2*pad, W + 2*pad)
            - Boolean cloud mask indicating where the cloud exists in the padded image
    """
    padded = xp.pad(image, pad_width=pad, mode='constant', constant_values=0)
    cloud_mask = padded.astype(bool)
    return padded, cloud_mask

def generate_annulus_stack(
    shape: Tuple[int, int], 
    radii: List[int]
) -> xp.ndarray:
    """
    Generate a stack of centered circular **annular** boolean masks.

    Each annulus includes points between radius r and r-1.

    Parameters:
        shape (Tuple[int, int]): Shape of the padded image (height, width).
        radii (List[int]): List of integer radii for which to generate annuli.

    Returns:
        xp.ndarray: 3D boolean array of shape (len(radii), H, W), each slice is an annular ring mask.
    """
    H, W = shape
    cy, cx = H // 2, W // 2

    Y, X = xp.meshgrid(xp.arange(H), xp.arange(W), indexing='ij')
    dist_sq = (Y - cy)**2 + (X - cx)**2

    masks = xp.zeros((len(radii), H, W), dtype=bool)
    prev_r_sq = 0

    for i, r in enumerate(radii):
        r_sq = r**2
        masks[i] = (dist_sq <= r_sq) & (dist_sq > prev_r_sq)
        prev_r_sq = r_sq

    return masks


def compute_radial_autocorr(
    image: xp.ndarray, 
    mask_stack: xp.ndarray, 
    cloud_mask: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Compute FFT-based radial autocorrelation over all cloud pixels,
    using circular masks of increasing radius.

    This version returns **unnormalized sums** designed to match the
    weighting strategy used in explicit pair-counting methods.

    Each radius r's correlation is computed by:
      - Centering the mask at each pixel in the cloud
      - Measuring the sum of values under the mask via FFT convolution
      - Accumulating the total sum across all cloud pixels
      - Tracking how many center–neighbor pairs contributed

    Use `C_r = sum_overlap_per_radius / total_pairs_per_radius` to compute the normalized correlation.

    Parameters:
        image (xp.ndarray): 2D binary array (padded) representing a single cloud.
        mask_stack (xp.ndarray): 3D array of circular masks, shape (N_radii, H, W).
        cloud_mask (xp.ndarray): Boolean mask of shape (H, W) indicating valid cloud centers.

    Returns:
        Tuple[xp.ndarray, xp.ndarray]:
            - sum_overlap_per_radius (xp.ndarray): Total summed correlation value at each radius (numerator), shape (N_radii,)
            - total_pairs_per_radius (xp.ndarray): Total number of valid center–neighbor pairs at each radius (denominator), shape (N_radii,)
    """ 
    # Step 1: FFT of the image (H, W)
    F_image = xp.fft.fft2(image)

    # Step 2: FFT of all masks (N_radii, H, W)
    F_masks = xp.fft.fft2(mask_stack, axes=(-2, -1))

    # Step 3: Convolve via FFT and bring back to real space
    conv_result = xp.fft.ifft2(F_image[None, :, :] * F_masks, axes=(-2, -1)).real  # (N_radii, H, W)

    # Step 4: Extract values only at positions where cloud_mask is True
    masked_result = conv_result[:, cloud_mask]  # (N_radii, N_valid_centers)

    # Step 5: Sum total overlap (numerator) for each radius
    sum_overlap_per_radius = masked_result.sum(axis=1)  # (N_radii,)

    # Step 6: Compute total number of contributing center–neighbor pairs
    N_valid_centers = xp.sum(cloud_mask)
    mask_area_per_radius = mask_stack.sum(axis=(1, 2))  # (N_radii,)
    total_pairs_per_radius = mask_area_per_radius * N_valid_centers  # (N_radii,)

    return sum_overlap_per_radius, total_pairs_per_radius

def to_numpy(arr):
    if xp_backend == "cupy":
        return xp.asnumpy(arr)
    return arr

def print_lattice(arr: xp.ndarray, title: str = "Lattice", cmap: str = "viridis") -> None:
    """
    Visualize a 2D lattice (CuPy or NumPy array) using matplotlib.

    Parameters:
        arr (xp.ndarray): 2D array to visualize. Can be either NumPy or CuPy.
        title (str): Title of the plot (default: "Lattice").
        cmap (str): Matplotlib colormap to use (default: "viridis").

    Returns:
        None. Displays a matplotlib figure.
    """
    arr = to_numpy(arr)  # Convert to NumPy if using CuPy

    plt.figure(figsize=(6, 5))
    plt.imshow(arr, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def extend_and_add(arr_a, arr_b) -> xp.ndarray:
    """Efficiently zero-extend and add two 1D xp arrays, returning the sum."""
    arr_a_len = arr_a.shape[0]
    arr_b_len = arr_b.shape[0]
    if arr_a_len == 0:
        return arr_b.copy()
    if arr_b_len == 0:
        return arr_a.copy()
    # If both are the same length, just return their sum
    if arr_b_len == arr_a_len:
        return arr_a + arr_b
    # Always extend the shorter one
    if arr_a_len < arr_b_len:
        total_ext = xp.zeros(arr_b_len, dtype=arr_a.dtype)
        total_ext[:arr_a_len] = arr_a
        total_ext += arr_b
        return total_ext
    else:
        arr_b_ext = xp.zeros(arr_a_len, dtype=arr_b.dtype)
        arr_b_ext[:arr_b_len] = arr_b
        return arr_a + arr_b_ext