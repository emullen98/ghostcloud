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
    image = xp.asarray(image)
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


import numpy as np
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


# ================================
#     Radial Binning Helpers
# ================================

def rmax_diagonal_batch(h_list, w_list):
    """
    Vectorized r_max for many clouds.

    Parameters
    ----------
    h_list, w_list : array-like of ints
        Heights and widths of each tightly-cropped cloud (not the full sky).

    Returns
    -------
    r_arr : array of int64, shape (N,)
        Per-cloud r_max.
    R_global : int
        Global maximum r across clouds (0 if no clouds).
    """
    import numpy as _np
    H = _np.asarray(h_list, dtype=_np.int64)
    W = _np.asarray(w_list, dtype=_np.int64)
    r_arr = _np.ceil(_np.hypot(H - 1, W - 1)).astype(_np.int64)
    return r_arr, int(r_arr.max()) if r_arr.size else 0

# module-level state (uses your global `xp`)
_R   = 0           # largest radius built so far
_IDX = None        # (2R+1, 2R+1) int32 ring indices (centered)
_CNT = None        # (R+1,) int64 ring counts

def build_rings_to(R_target: int) -> None:
    """Build (or grow) the cached ring-index grid up to R_target."""
    assert R_target >= 0, "R_target must be non-negative"
    global _R, _IDX, _CNT

    if _IDX is not None and R_target <= _R:
        return  # already big enough

    R = int(R_target)
    coords = xp.arange(-R, R + 1, dtype=xp.int32)
    y2 = coords[:, None].astype(xp.int64) ** 2
    x2 = coords[None, :].astype(xp.int64) ** 2
    s = x2 + y2
    r_idx = xp.ceil(xp.sqrt(s.astype(xp.float64))).astype(xp.int32)
    xp.clip(r_idx, 0, R, out=r_idx)

    cnt = xp.bincount(r_idx.ravel(), minlength=R + 1).astype(xp.int64)

    _R, _IDX, _CNT = R, r_idx, cnt

def ring_index(r: int):
    """Centered (2r+1, 2r+1) view of the cached ring indices."""
    if _IDX is None or r > _R:
        raise RuntimeError("Cache too small. Call build_rings_to(R_target) first.")
    if r == _R:
        return _IDX
    pad = _R - r
    return _IDX[pad:-pad, pad:-pad]

def ring_counts(r: int):
    """Counts per ring 0..r (length r+1)."""
    if _CNT is None or r > _R:
        raise RuntimeError("Cache too small. Call build_rings_to(R_target) first.")
    return _CNT[: r + 1]

def center_crop_to_radius(img, r: int):
    """Center-crop `img` to (2r+1, 2r+1). Expect zero-lag at center (fftshift first)."""
    H, W = img.shape
    cy, cx = H // 2, W // 2
    return img[cy - r: cy + r + 1, cx - r: cx + r + 1]

def radial_sum(img_cropped, r: int):
    """Sum values per integer radius 0..r using cached ring indices."""
    idx = ring_index(r)
    return xp.bincount(idx.ravel(), weights=img_cropped.ravel(), minlength=r + 1)

def crop_and_bin(img, r: int):
    """Convenience: crop to r and return (sums, counts) in one call."""
    c = center_crop_to_radius(img, r)
    sums = radial_sum(c, r)
    return sums, ring_counts(r)



# ================================
#           WK AUTOCORR
# ================================

def wk_radial_autocorr(single_cloud_binary,
                       return_numpy: bool = False,
                       dtype: str = "float64"):
    """
    Wiener–Khinchin radial autocorrelation for a SINGLE-CLOUD binary lattice.
    Backend-agnostic: works with either NumPy or CuPy via the global `xp`.

    Parameters
    ----------
    single_cloud_binary : (H, W) array-like of bool or {0,1} on host/device matching `xp`
        Binary image with exactly one connected cloud (foreground=1).
    return_numpy : bool, default False
        If True and xp is CuPy, convert outputs to NumPy on return.
    dtype : {"float64","float32"}, default "float64"
        Working dtype for FFT/computation. Use "float32" for GPU speed if acceptable.

    Returns
    -------
    num_r : (R,) array
        Radial numerator: sum of linear auto-correlation f ⋆ f over offsets in each ring.
    den_r : (R,) array
        Radial denominator: (# cloud pixels) × (number of offsets in annulus at radius r).
        This assumes a full ring is available around every cloud pixel. Ensure padding or
        truncate radii accordingly in your caller if needed.

    Notes
    -----
    * Uses linear (non-circular) correlation via FFT size (2H-1, 2W-1).
    * Zero-lag lives at (H-1, W-1) before shifting; we fftshift to center for radial binning.
    * Integer annuli via floor(sqrt(dx^2 + dy^2)).
    * Denominator is 'full-ring' normalization: area(cloud) * ring_size[r].
      For strict correctness, either pad the cloud with >= r_max margin on all sides
      and only trust r <= r_max, or truncate by a bbox-based safe radius.
    """

    # ---- helpers ----
    to_array = lambda a: xp.asarray(a, dtype=xp.float32 if dtype == "float32" else xp.float64)
    asnumpy = getattr(xp, "asnumpy", None)

    f = to_array(single_cloud_binary)
    if f.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    # Coerce to {0,1}
    f = (f > 0).astype(f.dtype, copy=False)

    if not xp.any(f):
        # empty cloud → trivial outputs (keep them on host if requested)
        r_vals = xp.asarray([0.0], dtype=f.dtype)
        zeros = xp.asarray([0.0], dtype=f.dtype)
        if return_numpy and asnumpy is not None:
            return asnumpy(zeros), asnumpy(zeros)
        return zeros, zeros

    H, W = f.shape
    out_shape = (int(2*H - 1), int(2*W - 1))  # Python ints are fine for both backends

    # --- Numerator map: linear autocorrelation via WK ---
    F = xp.fft.fftn(f, s=out_shape)
    num_map = xp.fft.ifftn(F * xp.conj(F)).real  # (2H-1, 2W-1)

    # Optional: harden against tiny negatives from roundoff (uncomment if you like)
    num_map = xp.rint(num_map)
    num_map = xp.maximum(num_map, 0)

    # Center zero-lag for radial binning
    num_map = xp.fft.fftshift(num_map)

    # --- Integer-radius annuli (on the displacement grid) ---
    cy, cx = (out_shape[0] // 2, out_shape[1] // 2)
    yy, xx = xp.ogrid[:out_shape[0], :out_shape[1]]
    dy = yy - cy
    dx = xx - cx
    # Use float64 for radius to avoid overflow/precision issues, then cast
    r_int = xp.floor(
        xp.sqrt(dy.astype("float64") * dy.astype("float64") +
                dx.astype("float64") * dx.astype("float64"))
    ).astype(xp.int32)

    r_max = int(r_int.max().item() if hasattr(r_int, "item") else r_int.max())

    # --- Radial numerator: sum num_map within each radius bin ---
    num_r = xp.bincount(r_int.ravel(), weights=num_map.ravel(), minlength=r_max + 1)

    # --- Radial denominator: area(cloud) * ring_size[r] ---
    cloud_area = f.sum()  # number of cloud pixels
    ring_size = xp.bincount(r_int.ravel(), minlength=r_max + 1).astype(f.dtype)
    den_r = cloud_area * ring_size

    if return_numpy and asnumpy is not None:
        return asnumpy(num_r), asnumpy(den_r)
    return num_r, den_r

# xp can be numpy or cupy
def wk_radial_autocorr_matching(
    f_uint8,                  # 2D uint8 {0,1} cloud (already padded)
    r_max,                    # max radius to consider (integer)
    dtype_fft=xp.float64,     # use float64 to validate; you can switch to float32 later
):
    """
    Cloud-centered WK autocorr with analytic ring binning r = ceil(sqrt(dx^2+dy^2)).
    Denominator matches your annulus version: centers inside cloud, neighbors anywhere in support.
    Assumes padding >= r_max on all sides of f_uint8 inside the support rectangle.
    """

    # --- 1) Build support mask m: 1's over the entire padded rectangle
    H, W = f_uint8.shape
    m_uint8 = xp.ones((H, W), dtype=xp.uint8)

    # --- 2) Cast once for FFTs
    f = f_uint8.astype(dtype_fft)
    m = m_uint8.astype(dtype_fft)

    # --- 3) FFTs
    F = xp.fft.fft2(f)
    M = xp.fft.fft2(m)

    # --- 4) Displacement-wise maps (WK)
    # Numerator: cloud–cloud pairs at each shift Δ
    N_img = xp.fft.ifft2(F * xp.conj(F)).real
    # Denominator: cloud-centered (centers in f, neighbor in support m)
    D_img = xp.fft.ifft2(F * xp.conj(M)).real

    # Numerical cleanup
    eps = 1e-12 if dtype_fft == xp.float64 else 1e-6
    N_img = xp.where(N_img > eps, N_img, 0.0)
    D_img = xp.where(D_img > eps, D_img, 0.0)

    # --- 5) Put zero-lag at center
    N_img = xp.fft.fftshift(N_img)
    D_img = xp.fft.fftshift(D_img)

    # --- 6) Analytic ring indices: r = ceil(sqrt(dx^2+dy^2))
    cy, cx = H // 2, W // 2
    y = xp.arange(H) - cy
    x = xp.arange(W) - cx
    Y, X = xp.meshgrid(y, x, indexing='ij')
    R_idx = xp.ceil(xp.sqrt(Y*Y + X*X)).astype(xp.int32)

    # Limit to desired radii and where denominator is valid
    valid = (R_idx <= r_max) & (D_img > 0)

    # --- 7) One-pass radialization via bincount
    Rv = R_idx[valid].ravel()
    Nv = N_img[valid].ravel()
    Dv = D_img[valid].ravel()

    minlength = int(r_max) + 1   # include r = 0
    N_r = xp.bincount(Rv, weights=Nv, minlength=minlength)
    D_r = xp.bincount(Rv, weights=Dv, minlength=minlength)

    return N_r, D_r

def wk_radial_autocorr_matching_optimized(
    f_uint8,                  # 2D uint8 {0,1} cloud (already padded by >= r_max)
    r_max: int,               # max radius to consider
    dtype_fft=xp.float32,     # float64 for validation; float32 for speed later
):
    """
    Cloud-centered WK autocorr with analytic ring binning r = ceil(sqrt(dx^2+dy^2)).
    Optimized:
      - Real FFTs for numerator
      - No denominator FFT: D_r = |f| * ring_counts(r)
      - Cached ring indices/counts; zero per-call meshgrid

    Assumes padding >= r_max on all sides of f_uint8 inside the support rectangle.
    Returns:
      N_r, D_r  (both length r_max+1)
    """

    # --- 0) Make sure our ring cache can serve up to r_max
    build_rings_to(int(r_max))

    # --- 1) Cast once for FFT
    # Keep the original uint8 for counting ones exactly and cheaply.
    H, W = f_uint8.shape
    f = f_uint8.astype(dtype_fft, copy=False)

    # --- 2) Numerator image via WK using real FFTs
    # N_img = ifft2(F * conj(F)).real, but use rfft2/irfft2 for memory savings
    F = xp.fft.rfft2(f)                                       # shape (H, W//2+1)
    N_img = xp.fft.irfft2(F * xp.conj(F), s=(H, W))           # real autocorr image

    # --- 3) Numerical cleanup (clip tiny negatives to 0 to avoid bincount drift)
    eps = 1e-12 if dtype_fft == xp.float64 else 1e-6
    N_img = xp.where(N_img > eps, N_img, 0.0)

    # --- 4) Put zero-lag at center and crop to (2r+1, 2r+1)
    N_img = xp.fft.fftshift(N_img)
    N_c = center_crop_to_radius(N_img, int(r_max))

    # --- 5) Radialize numerator (fast bincount using cached indices)
    N_r = radial_sum(N_c, int(r_max))                         # length r_max+1

    # --- 6) Denominator shortcut: constant plane inside the crop
    # |f| = number of cloud centers
    sum_f = f_uint8.sum(dtype=dtype_fft)
    cnt = ring_counts(int(r_max))                             # int64 counts per ring
    D_r = sum_f * cnt.astype(dtype_fft, copy=False)

    return N_r, D_r