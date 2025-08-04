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


def pad_image(image: xp.ndarray, pad: int) -> xp.ndarray:
    """
    Pad a binary cropped image with zeros.

    Parameters:
        image (xp.ndarray): 2D uint8 array (0 or 1).
        pad (int): Number of pixels to pad on each side of the image.

    Returns:
        xp.ndarray: Padded uint8 array.
    """
    assert image.dtype == xp.uint8  # Save memory, easy casting to float32 later
    padded = xp.pad(image, pad_width=pad, mode='constant', constant_values=0)
    return padded.astype(xp.uint8)


def generate_annulus_stack(
    shape: Tuple[int, int],
    radii: List[int]
) -> xp.ndarray:
    """
    Generate a stack of centered circular annular masks.

    Each annulus includes points between radius r and r-1.

    Parameters:
        shape (Tuple[int, int]): Shape of the padded image (H, W).
        radii (List[int]): Radii for which to generate annuli.

    Returns:
        xp.ndarray: 3D uint8 array of shape (len(radii), H, W), each slice is an annular ring mask.
    """
    H, W = shape
    cy, cx = H // 2, W // 2

    Y, X = xp.meshgrid(xp.arange(H), xp.arange(W), indexing='ij')
    dist_sq = (Y - cy) ** 2 + (X - cx) ** 2

    masks = xp.zeros((len(radii), H, W), dtype=xp.uint8)
    prev_r_sq = 0

    for i, r in enumerate(radii):
        r_sq = r ** 2
        # Save as uint8 mask for memory efficiency
        masks[i] = ((dist_sq <= r_sq) & (dist_sq > prev_r_sq)).astype(xp.uint8)
        prev_r_sq = r_sq

    return masks


def compute_radial_autocorr(
    image: xp.ndarray,
    mask_stack: xp.ndarray
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Estimate the radial autocorrelation function C(r) using FFT-based convolution.

    For each radius r, this computes:
        C(r) = number of cloud pixels found at distance r from cloud centers
               divided by
               total number of pixels tested at that distance

    Parameters:
        image: 2D uint8 array representing the padded cloud (nonzero = cloud pixels).
        mask_stack: 3D uint8 array of annular masks (N_radii, H, W).

    Returns:
        Tuple (numerator, denominator) arrays for each radius r.
    """
    # Convert to float32 for FFT (ensures complex64 transforms, memory efficient)
    image_f = image.astype(xp.float32)
    mask_stack_f = mask_stack.astype(xp.float32)

    # Compute valid cloud centers as nonzero pixels
    cloud_mask = (image > 0)

    # FFT of image and annuli stack
    F_image = xp.fft.fft2(image_f)
    
    # Shift annulus stack to center before FFT
    shifted_masks = xp.fft.ifftshift(mask_stack_f, axes=(-2, -1))
    F_masks = xp.fft.fft2(shifted_masks, axes=(-2, -1))

    # Convolve and take real part
    conv_result = xp.real(xp.fft.ifft2(F_image[None, :, :] * F_masks, axes=(-2, -1)))

    # Extract result at valid cloud pixels
    masked_result = conv_result[:, cloud_mask]  # shape: (N_radii, N_valid_centers)
    sum_overlap_per_radius = masked_result.sum(axis=1)  # numerator

    # Denominator = total center–neighbor pairs
    N_valid_centers = xp.sum(cloud_mask)
    mask_area_per_radius = mask_stack.sum(axis=(1, 2))  # still uint8 → auto-promoted
    total_pairs_per_radius = mask_area_per_radius * N_valid_centers

    return sum_overlap_per_radius, total_pairs_per_radius



def to_numpy(arr: xp.ndarray) -> np.ndarray:
    """
    Convert CuPy array to NumPy if needed. No-op for NumPy arrays.
    """
    if xp_backend == "cupy":
        return xp.asnumpy(arr)
    return arr


def print_lattice(arr: xp.ndarray, title: str = "Lattice", cmap: str = "viridis") -> None:
    """
    Visualize a 2D lattice using matplotlib.

    Parameters:
        arr: 2D NumPy or CuPy array.
        title: Plot title.
        cmap: Color map to use.
    """
    arr = to_numpy(arr)
    plt.figure(figsize=(6, 5))
    plt.imshow(arr, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()


def extend_and_add(arr_a: xp.ndarray, arr_b: xp.ndarray) -> xp.ndarray:
    """
    Efficiently zero-extend and add two 1D xp arrays.
    Used to aggregate autocorr numerators/denominators across variable-sized clouds.
    """
    arr_a_len = arr_a.shape[0]
    arr_b_len = arr_b.shape[0]

    if arr_a_len == 0:
        return arr_b.copy()
    if arr_b_len == 0:
        return arr_a.copy()

    if arr_a_len == arr_b_len:
        return arr_a + arr_b

    if arr_a_len < arr_b_len:
        total_ext = xp.zeros(arr_b_len, dtype=arr_a.dtype)
        total_ext[:arr_a_len] = arr_a
        total_ext += arr_b
        return total_ext
    else:
        arr_b_ext = xp.zeros(arr_a_len, dtype=arr_b.dtype)
        arr_b_ext[:arr_b_len] = arr_b
        return arr_a + arr_b_ext