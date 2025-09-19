# autocorr_utils.py
"""
Autocorrelation utilities (WK and annulus) with NumPy/CuPy backends.

Philosophy:
- Compute C(r) *definitionally* for integer shells k = ceil(sqrt(dx^2+dy^2)).
- r_max ("r_bin") is a physical cutoff where C(r) has effectively decayed to ~0 for the cropped cloud.
- Padding radius ("r_pad") is solely to guarantee linear convolution for FFTs; it may be >= r_bin.
- Denominators are purely geometric: D[k] = |f| * # { lattice points with ring index == k }.
- Ring counts are computed *locally* inside the (2r+1)^2 crop and exclude indices > r.
"""

from scipy.optimize import curve_fit
from dataclasses import dataclass
from . import config
from typing import Tuple, List, Literal, Optional
import math
import json

# ---------------------------
# Backend selection (NumPy/CuPy)
# ---------------------------
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

# ---------------------------
# Small helpers
# ---------------------------

def to_numpy(arr):
    """Convert CuPy array to NumPy if needed. No-op for NumPy arrays."""
    if xp_backend == "cupy":
        return xp.asnumpy(arr)
    return arr

def extend_and_add(a, b):
    """
    Efficiently zero-extend and add two 1D arrays (xp-compatible).
    Used to aggregate numerators/denominators across clouds whose r_max differ.
    """
    la, lb = a.shape[0], b.shape[0]
    if la == 0: return b.copy()
    if lb == 0: return a.copy()
    if la == lb: return a + b
    if la < lb:
        out = xp.zeros(lb, dtype=a.dtype)
        out[:la] = a
        out += b
        return out
    else:
        out = xp.zeros(la, dtype=b.dtype)
        out[:lb] = b
        return a + out

# -----------------------------------
# r_max (binning) for cropped clouds
# -----------------------------------

def rmax_diagonal_batch(h_list, w_list):
    """
    Tight per-cloud *binning* radius based on the cropped bounding-box diagonal.

    Returns
    -------
    r_arr : int64[N]
        ceil(hypot(H-1, W-1)) per cloud (no doubling).
    R_global : int
        max(r_arr) or 0 if no clouds.
    """
    import numpy as _np
    H = _np.asarray(h_list, dtype=_np.int64)
    W = _np.asarray(w_list, dtype=_np.int64)
    r_arr = _np.ceil(_np.hypot(H - 1, W - 1)).astype(_np.int64)
    return r_arr, int(r_arr.max()) if r_arr.size else 0

# -----------------------------------
# Ring-index cache (ceil Euclidean)
# -----------------------------------
# Stores ring indices for a centered (2R+1)×(2R+1) grid.
# r = ceil(sqrt(x^2 + y^2)) with exact square detection; r ∈ [0, R].
_R   = 0             # largest cached radius
_IDX = None          # (2R+1, 2R+1) int32 of ring indices
_CNT = None          # (R+1,) int64 global histogram (not used for binning)

def clear_ring_cache():
    """Free the global ring cache (useful for memory experiments)."""
    global _R, _IDX, _CNT
    _R, _IDX, _CNT = 0, None, None

def build_rings_to(R_target: int) -> None:
    """
    Build/extend the global ring-index cache to at least R_target.
    The cache is centered and uses integer arithmetic for exactness.
    """
    assert R_target >= 0
    global _R, _IDX, _CNT
    if _IDX is not None and R_target <= _R:
        return

    R = int(R_target)
    coords = xp.arange(-R, R + 1, dtype=xp.int64)
    y = coords[:, None]
    x = coords[None, :]
    s = x * x + y * y                      # exact in int64

    # floor(sqrt) in float, correct at perfect squares to emulate ceil
    r_floor = xp.floor(xp.sqrt(s.astype(xp.float64))).astype(xp.int64)
    is_square = (r_floor * r_floor == s)   # exact
    r_idx = r_floor + (~is_square)         # ceil: add 1 if not a square
    r_idx = r_idx.astype(xp.int32)
    xp.clip(r_idx, 0, R, out=r_idx)

    _R   = R
    _IDX = r_idx
    _CNT = xp.bincount(r_idx.ravel(), minlength=R + 1).astype(xp.int64)  # unused but handy

def ring_index(r: int):
    """
    Return a *view* of the centered (2r+1)×(2r+1) ring-index matrix
    from the cached (2R+1)×(2R+1). Expect zero-lag at the center.
    """
    if _IDX is None or r > _R:
        raise RuntimeError("Cache too small. Call build_rings_to(R_target) first.")
    if r == _R:
        return _IDX
    pad = _R - r
    if r == 0:
        return _IDX[pad:pad+1, pad:pad+1]
    return _IDX[pad:-pad, pad:-pad]

def ring_counts(r: int):
    """
    Counts for shells 0..r inside the (2r+1)^2 crop (EXCLUDES indices > r).
    Returns int64[r+1].
    """
    idx = ring_index(r)
    flat = idx.ravel()
    m = flat <= r
    return xp.bincount(flat[m], minlength=r+1).astype(xp.int64)

def center_crop_to_radius(img, r: int):
    """Crop `img` to (2r+1, 2r+1) around the center pixel."""
    H, W = img.shape
    cy, cx = H // 2, W // 2
    return img[cy - r: cy + r + 1, cx - r: cx + r + 1]

def radial_sum(img_cropped, r: int):
    """
    Weighted sums per integer shell 0..r using the cached ring indices.
    EXCLUDES indices > r to preserve definitional shells.
    """
    idx = ring_index(r)
    flat_i = idx.ravel()
    flat_w = img_cropped.ravel()
    m = flat_i <= r
    return xp.bincount(flat_i[m], weights=flat_w[m], minlength=r+1)

def crop_and_bin(img, r: int):
    """Convenience: crop to r and return (sums, counts) in one call."""
    c = center_crop_to_radius(img, r)
    return radial_sum(c, r), ring_counts(r)

# -----------------------------------
# Padding helpers (WK linearity)
# -----------------------------------

def _calc_padding_per_side(H: int, W: int, R: int, guard: int = 0):
    """
    Minimal per-side halo for lags |dx|,|dy| <= R to be linear (no wrap).
    Also enforces odd×odd padded size so the center pixel is well-defined.
    Returns (Py, Px) to add to *each* side.
    """
    Py = R + guard
    Px = R + guard
    if (H + 2*Py) % 2 == 0: Py += 1
    if (W + 2*Px) % 2 == 0: Px += 1
    return Py, Px

def pad_for_wk(image, R: int, guard: int = 0):
    """
    Pad with the minimal linear-safe halo for a given *padding radius* R.
    NOTE: R here is your *r_pad* (NOT r_bin). You still pass r_bin to binning.
    Returns (padded_uint8, (Py, Px)).
    """
    H, W = image.shape
    Py, Px = _calc_padding_per_side(H, W, R, guard=guard)
    padded = xp.pad(image, ((Py, Py), (Px, Px)), mode="constant")
    return padded.astype(xp.uint8), (Py, Px)

# -----------------------------------
# WK autocorrelation (fast path)
# -----------------------------------

def wk_radial_autocorr(
    f_uint8,                  # 2D uint8 {0,1} cloud (already padded by >= r_bin *in practice r_pad* )
    r_max: int,               # r_bin: max radius to *bin* (physical support)
    dtype_fft=None,           # None -> float32 by default; use float64 for validation
):
    """
    Cloud-centered WK autocorr with analytic ring binning r = ceil(sqrt(dx^2+dy^2)).

    - Numerator N_img = ifft2(F * conj(F)).real (via rfft2/irfft2 for memory).
    - Denominator D_r = |f| * ring_counts(r) (no FFT).
    - Shells are definitional: counts and sums *exclude* indices > r in the crop.

    Assumptions:
    - f_uint8 has been padded by a halo sufficient for linearity (your r_pad >= r_max).
    - Zero-lag (autocorr peak) ends up at the center after fftshift.
    """
    # 0) Types & tolerances
    if dtype_fft is None:
        dtype_fft = xp.float64 
    eps = 1e-12 if dtype_fft == xp.float64 else 1e-6

    # 1) Ensure ring cache can serve up to r_max
    r_max = int(r_max)
    build_rings_to(r_max)

    # 2) Cast to FFT dtype (keep uint8 around for exact |f|)
    H, W = f_uint8.shape
    f = f_uint8.astype(dtype_fft, copy=False)

    # 3) WK numerator via real FFTs
    #    Explicit s=(H,W) to avoid surprises after padding.
    F = xp.fft.rfft2(f, s=(H, W))
    N_img = xp.fft.irfft2(F * xp.conj(F), s=(H, W))

    # 4) Numerical cleanup: zero out tiny negatives (FFT noise)
    #    Using a threshold avoids small negative values polluting bincount shells.
    if eps > 0:
        mask = N_img <= eps
        if mask.any():
            N_img[mask] = 0.0

    # 5) Center zero-lag and crop to (2r+1)^2
    N_img = xp.fft.fftshift(N_img)
    N_c = center_crop_to_radius(N_img, r_max)

    # 6) Radialize numerator & denominator
    N_r = radial_sum(N_c, r_max).astype(dtype_fft, copy=False)  # length r_max+1
    sum_f = f_uint8.sum(dtype=xp.int64)                         # # of ones (exact, int64)
    cnt = ring_counts(r_max)                                    # int64[r+1]
    D_r = (sum_f * cnt).astype(dtype_fft, copy=False)           # cast once at the edge

    return N_r, D_r

# -----------------------------------
# Annulus method (validation/legacy)
# -----------------------------------

def generate_annulus_stack(shape: Tuple[int, int], radii: List[int]):
    """
    [Validation/legacy] Build a dense stack of annuli masks (O(len(r)*H*W) memory).
    Prefer the WK path for speed/memory; keep this for cross-checks.
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    Y, X = xp.meshgrid(xp.arange(H), xp.arange(W), indexing='ij')
    dist_sq = (Y - cy) ** 2 + (X - cx) ** 2

    masks = xp.zeros((len(radii), H, W), dtype=xp.uint8)
    prev_r_sq = 0
    for i, r in enumerate(radii):
        r_sq = int(r) ** 2
        masks[i] = ((dist_sq <= r_sq) & (dist_sq > prev_r_sq)).astype(xp.uint8)
        prev_r_sq = r_sq
    return masks

def compute_radial_autocorr(image, mask_stack):
    """
    [Validation/legacy] Annulus convolution path for C(r). Heavy on memory; slower.
    """
    image_f = image.astype(xp.float32, copy=False)
    mask_stack_f = mask_stack.astype(xp.float32, copy=False)

    cloud_mask = (image > 0)
    F_image = xp.fft.fft2(image_f)
    shifted_masks = xp.fft.ifftshift(mask_stack_f, axes=(-2, -1))
    F_masks = xp.fft.fft2(shifted_masks, axes=(-2, -1))

    conv_result = xp.real(xp.fft.ifft2(F_image[None, :, :] * F_masks, axes=(-2, -1)))
    masked = conv_result[:, cloud_mask]             # (N_radii, N_centers)
    numerator = masked.sum(axis=1)                  # N[r]
    n_centers = cloud_mask.sum(dtype=xp.int64)
    area = mask_stack.sum(axis=(1, 2), dtype=xp.int64)
    denominator = area * n_centers                  # D[r]
    return numerator, denominator


# ...existing code...

@dataclass
class ExponentialFit:
    """Results from fitting C(r) to exponential decay."""
    correlation_length: float
    success: bool
    r_eff: float  # effective range where C(r) drops below threshold
    r_half: float  # r where C(r) = 0.5
    error: Optional[str] = None

def find_characteristic_lengths(r: np.ndarray, cr: np.ndarray, 
                             threshold: float = 0.01) -> Tuple[float, float]:
    """Find characteristic lengths from correlation data."""
    # Find r_eff where C(r) first drops below threshold
    below_thresh = np.where(cr < threshold)[0]
    r_eff = r[below_thresh[0]] if len(below_thresh) > 0 else r[-1]
    
    # Find r_half where C(r) crosses 0.5
    half_crosses = np.where(np.diff(np.signbit(cr - 0.5)))[0]
    r_half = r[half_crosses[0]] if len(half_crosses) > 0 else r_eff/2
    
    return r_eff, r_half

def fit_exponential_decay(r: np.ndarray, cr: np.ndarray, 
                         threshold: float = 0.01) -> ExponentialFit:
    """
    Fit correlation function to exponential decay after normalizing by r_eff.
    
    Args:
        r: Array of r values
        cr: Array of C(r) values
        threshold: Value defining effective zero
    """
    try:
        # Find characteristic lengths
        r_eff, r_half = find_characteristic_lengths(r, cr, threshold)
        
        # Normalize r by r_eff
        r_norm = r / r_eff
        
        # Fit only up to where C(r) > threshold
        mask = cr >= threshold
        r_fit = r_norm[mask]
        cr_fit = cr[mask]
        
        def exp_model(r, xi):
            return np.exp(-r/xi)
        
        # Fit normalized exponential
        popt, _ = curve_fit(exp_model, r_fit, cr_fit, p0=[0.5])
        
        return ExponentialFit(
            correlation_length=float(popt[0]),
            success=True,
            r_eff=float(r_eff),
            r_half=float(r_half)
        )
        
    except Exception as e:
        return ExponentialFit(
            correlation_length=float('nan'),
            success=False,
            r_eff=float('nan'),
            r_half=float('nan'),
            error=str(e)
        )

def save_fit_results(fits: List[ExponentialFit], output_path: str) -> None:
    """Save fitting results to JSON file."""
    results = {
        "fits": [
            {
                "correlation_length": fit.correlation_length,
                "r_eff": fit.r_eff,
                "r_half": fit.r_half,
                "success": fit.success,
                "error": fit.error
            }
            for fit in fits
        ],
        "summary": {
            "n_success": sum(1 for f in fits if f.success),
            "mean_correlation_length": float(np.mean([f.correlation_length 
                                                    for f in fits if f.success])),
            "std_correlation_length": float(np.std([f.correlation_length 
                                                  for f in fits if f.success]))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)