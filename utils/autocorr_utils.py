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
from __future__ import annotations
from dataclasses import dataclass
from . import config
from typing import Tuple, List, Literal, Optional, Dict
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.ndimage as ndimage
from clouds.utils.cloud_utils import compute_perimeter as compute_perimeter4c
import numpy as np
import math

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

# ---- Boundary mask via 4-connected neighbor counts (perimeter-style) ----
# Uses ndimage.convolve with the 4-neighbor cross kernel.
# GPU-friendly: will use cupyx.scipy.ndimage if xp is CuPy.
def make_boundary_mask4(mask_uint8):
    """
    4-connected boundary of a binary mask.
    Works for NumPy (CPU) and CuPy (GPU). Returns xp.uint8 on the active backend.
    """
    if xp_backend == "cupy":
        import cupy as cp
        from cupyx.scipy import ndimage as cnd

        # Ensure data and kernel are BOTH on GPU
        m = cp.asarray(mask_uint8, dtype=cp.uint8)
        m = (m != 0).astype(cp.uint8, copy=False)

        k4 = cp.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=cp.uint8)

        # Count 4-neighbors; constant padding = outside
        neigh = cnd.convolve(m, k4, mode="constant", cval=0)

        # Boundary: inside pixels that have <4 inside neighbors
        b = (m & (neigh < 4)).astype(cp.uint8, copy=False)
        return b

    else:
        import numpy as np
        from scipy import ndimage as snd

        m = (np.asarray(mask_uint8) != 0).astype(np.uint8, copy=False)

        k4 = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)

        neigh = snd.convolve(m, k4, mode="constant", cval=0)
        b = (m & (neigh < 4)).astype(np.uint8, copy=False)
        return b


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
    import numpy as np
    H = np.asarray(h_list, dtype=np.int64)
    W = np.asarray(w_list, dtype=np.int64)
    r_arr = np.ceil(np.hypot(H - 1, W - 1)).astype(np.int64)
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
# Efficient ring count builder
# -----------------------------------

def ring_counts_quadrant(R: int, dtype=np.int64) -> np.ndarray:
    """
    Exact integer lattice ring counts for the full 2D grid using a quadrant shortcut,
    with NO caching. Intended to be called once per aggregator run.

    Shell index:
        k = ceil(sqrt(x^2 + y^2)) for integer (x, y).

    Method:
      1) Count shells on the first quadrant (x >= 0, y >= 0):
           For k >= 1, the points with ceil(sqrt(.)) == k are those with
             (k-1)^2 < x^2 + y^2 <= k^2.
         We compute, for each x, the y-interval size between these two circles
         using integer square roots (isqrt) to avoid floating point issues.
      2) Mirror to the full plane with axis correction:
           cnt[0] = 1
           cnt[k] = 4 * Q[k] - 4     (k >= 1),
         since naive mirroring would overcount the 4 axis points for each k.

    Parameters
    ----------
    R : int
        Maximum radius (inclusive) of shells to count.
    dtype : np.dtype, optional
        Integer dtype of the returned counts.

    Returns
    -------
    cnt : np.ndarray, shape (R+1,), dtype=dtype
        Exact counts of lattice sites with shell index k = 0..R.
    """
    if R < 0:
        raise ValueError("R must be >= 0")

    Q = np.zeros(R + 1, dtype=dtype)

    # k = 0 shell: only the origin (0,0) in the quadrant counting
    Q[0] = 1

    # For k >= 1, count points with (k-1)^2 < x^2 + y^2 <= k^2 in the first quadrant
    for k in range(1, R + 1):
        kk = k * k
        km = (k - 1) * (k - 1)

        qk = 0
        # Only x in [0, k] can contribute since kk - x^2 must be >= 0
        for x in range(0, k + 1):
            x2 = x * x

            a = kk - x2       # upper circle (<= k^2)
            if a < 0:
                break
            y_max = math.isqrt(a)

            b = km - x2       # lower circle (<= (k-1)^2)
            y_prev = math.isqrt(b) if b >= 0 else -1

            # Count y such that y_prev < y <= y_max  ⇒  (k-1)^2 < x^2 + y^2 <= k^2
            qk += (y_max - y_prev)

        Q[k] = qk

    # Mirror to full plane with axis correction
    cnt = (4 * Q).astype(dtype, copy=False)
    if R >= 1:
        cnt[1:] -= 4
    cnt[0] = 1

    return cnt

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

def wk_radial_autocorr_dual(
    cloud_uint8,               # 2D uint8 array {0,1}, already padded by >= r_max
    r_max: int,                # maximum radial bin (shells up to this radius)
    dtype_fft=None,            # None -> float64 (for accuracy); use float32 for speed
):
    """
    Wiener–Khinchin radial autocorrelation with dual center sets:
      (A) centers = all cloud pixels
      (B) centers = boundary cloud pixels (4-connected)

    The numerator is obtained from the FFT-based autocorrelation:
        N_img = ifft2( F(f) * conj(F(mask)) )
    where f is the full cloud, and mask is the set of valid centers.

    The denominator is purely analytic:
        D_r = (# centers) * ring_counts(r)

    Assumptions
    -----------
    - cloud_uint8 is binary {0,1} and padded with >= r_max halo so that
      every cloud pixel (or boundary pixel) can act as a valid center.
    - The ring cache has been prebuilt via build_rings_to(r_max).

    Parameters
    ----------
    cloud_uint8 : ndarray (H, W), dtype=uint8
        Binary padded cloud image.
    r_max : int
        Maximum radius to bin into shells.
    dtype_fft : dtype or None
        Floating-point precision for FFTs. None -> float64 (safer).
        Use float32 for large production runs if memory bound.

    Returns
    -------
    results : dict
        {
          "all":      (N_r_all, D_r_all),       # centers = all cloud pixels
          "boundary": (N_r_bnd, D_r_bnd)        # centers = 4-connected boundary
        }
        Each N_r, D_r is a 1D array of length (r_max+1), dtype=dtype_fft.
    """
    # ------------------------------
    # 0) Types & tolerances
    # ------------------------------
    if dtype_fft is None:
        dtype_fft = xp.float64
    eps = 1e-12 if dtype_fft == xp.float64 else 1e-6

    r_max = int(r_max)
    build_rings_to(r_max)  # ensure ring cache covers up to r_max

    H, W = cloud_uint8.shape
    f = cloud_uint8.astype(dtype_fft, copy=False)

    # ------------------------------
    # 1) FFT of full cloud (expensive, do once)
    # ------------------------------
    F_f = xp.fft.rfft2(f, s=(H, W))

    # ------------------------------
    # 2) Define center masks
    # ------------------------------
    mask_all = cloud_uint8                        # all cloud pixels as centers
    mask_bnd = make_boundary_mask4(cloud_uint8)   # boundary-only centers (4-conn)

    # ------------------------------
    # 3) FFTs of masks (cheap compared to #1)
    # ------------------------------
    F_mask_all = xp.fft.rfft2(mask_all.astype(dtype_fft, copy=False), s=(H, W))
    F_mask_bnd = xp.fft.rfft2(mask_bnd.astype(dtype_fft, copy=False), s=(H, W))

    # ------------------------------
    # 4) Helper: cross-correlation with given mask
    # ------------------------------
    def _corr_with_mask(F_mask, mask_uint8):
        # WK numerator: cross-correlation f ⊗ mask
        N_img = xp.fft.irfft2(F_f * xp.conj(F_mask), s=(H, W))

        # Numerical cleanup: eliminate tiny negatives from roundoff
        if eps > 0:
            nz = N_img <= eps
            if nz.any():
                N_img[nz] = 0.0

        # Center zero-lag, crop to (2r+1)^2, then radialize
        N_img = xp.fft.fftshift(N_img)
        N_c = center_crop_to_radius(N_img, r_max)
        N_r = radial_sum(N_c, r_max).astype(dtype_fft, copy=False)

        # Denominator: (# of centers) * ring_counts
        num_centers = mask_uint8.sum(dtype=xp.int64)
        cnt = ring_counts(r_max)
        D_r = (num_centers * cnt).astype(dtype_fft, copy=False)

        return N_r, D_r

    # ------------------------------
    # 5) Compute both outputs
    # ------------------------------
    N_all, D_all = _corr_with_mask(F_mask_all, mask_all)
    N_bnd, D_bnd = _corr_with_mask(F_mask_bnd, mask_bnd)

    return {
        "all":      (N_all, D_all),
        "boundary": (N_bnd, D_bnd),
    }

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

# -----------------------------------
# Per-cloud C(r) Parquet writer
# -----------------------------------

@dataclass
class CloudRow:
    # Geometry
    cloud_idx: int
    perim: int
    area: int

    # Tags
    threshold: Optional[float] = None
    p_val: Optional[float] = None

    # Existing WK / accumulators (keep if used elsewhere)
    cr: Optional[np.ndarray] = None
    cr_bnd: Optional[np.ndarray] = None
    num_all: Optional[np.ndarray] = None
    den_all: Optional[np.ndarray] = None
    num_bnd: Optional[np.ndarray] = None
    den_bnd: Optional[np.ndarray] = None

    # Minimal new fields
    com: Optional[np.ndarray] = None          # [cy, cx]
    rg_area: Optional[float] = None           # scalar
    rg_bnd: Optional[float] = None            # scalar (nullable)
    bd_r: Optional[np.ndarray] = None         # r-bin centers
    bd_counts: Optional[np.ndarray] = None    # raw counts

    # Minimal disambiguation tags
    bd_bin_width: Optional[float] = None      # Δr used for bd histogram
    center_method: Optional[str] = None       # e.g., "com"
    boundary_connectivity: Optional[str] = None  # e.g., "4c"
    bd_n: Optional[int] = None                # boundary pixel count (Nb)


class ParquetWriter:
    """
    Append-style writer for per-cloud outputs, now including optional radial profile columns.

    Columns:
      - cloud_idx (int64)
      - perim     (int64)
      - area      (int64)
      - threshold (float64, nullable)
      - p_val     (float64, nullable)

      - cr        (List<float64>, nullable)
      - cr_bnd    (List<float64>, nullable)
      - num_all   (List<float64>, nullable)
      - den_all   (List<float64>, nullable)
      - num_bnd   (List<float64>, nullable)
      - den_bnd   (List<float64>, nullable)

      - rp_r      (List<float64>, nullable)      # NEW
      - rp_counts (List<float64>, nullable)      # NEW
      - rp_pdf    (List<float64>, nullable)      # NEW
      - rp_f_ring (List<float64>, nullable)      # NEW

    Writes to sharded part files for safe incremental appends.
    """
    def __init__(self, outdir: str, basename: str,
                 rows_per_flush: int | None = None,
                 max_bytes_per_flush: int = 128 * 1024 * 1024):
        os.makedirs(outdir, exist_ok=True)
        self.base = os.path.join(outdir, basename)
        self.rows_per_flush = rows_per_flush
        self.max_bytes_per_flush = max_bytes_per_flush
        self._rows: List[CloudRow] = []
        self._approx_bytes = 0
        self._part_idx = 0

    def add(self, row: CloudRow):
        # dtype normalization (arrays -> float64)
        def _as_f64(x):
            return None if x is None else np.asarray(x, dtype=np.float64)

        row.cr       = _as_f64(row.cr)
        row.cr_bnd   = _as_f64(row.cr_bnd)
        row.num_all  = _as_f64(row.num_all)
        row.den_all  = _as_f64(row.den_all)
        row.num_bnd  = _as_f64(row.num_bnd)
        row.den_bnd  = _as_f64(row.den_bnd)

        row.com       = _as_f64(row.com)
        row.bd_r      = _as_f64(row.bd_r)
        row.bd_counts = _as_f64(row.bd_counts)

        # queue row
        self._rows.append(row)

        # conservative size estimate for flush policy
        size = 32
        for arr in (row.cr, row.cr_bnd, row.num_all, row.den_all,
                    row.num_bnd, row.den_bnd, row.com, row.bd_r, row.bd_counts):
            if arr is not None:
                size += arr.nbytes + 16
        if row.rg_area is not None: size += 16
        if row.rg_bnd  is not None: size += 16
        self._approx_bytes += size

        # flush when hitting row or byte thresholds
        need_row_flush = (self.rows_per_flush is not None) and (len(self._rows) >= self.rows_per_flush)
        need_byte_flush = (self._approx_bytes >= self.max_bytes_per_flush)
        if need_row_flush or need_byte_flush:
            self.flush()

    def flush(self):
        if not self._rows:
            return
        part_path = f"{self.base}.part{self._part_idx:05d}.parquet"
        table = self._rows_to_table(self._rows)
        pq.write_table(table, part_path, compression="zstd", use_dictionary=False)
        self._rows.clear()
        self._approx_bytes = 0
        self._part_idx += 1

    def close(self):
        self.flush()

    @staticmethod
    def _rows_to_table(rows: List['CloudRow']) -> pa.Table:
        # scalars
        cloud_idx = pa.array([r.cloud_idx for r in rows], type=pa.int64())
        perim     = pa.array([r.perim     for r in rows], type=pa.int64())
        area      = pa.array([r.area      for r in rows], type=pa.int64())
        threshold = pa.array([None if r.threshold is None else float(r.threshold) for r in rows], type=pa.float64())
        p_val     = pa.array([None if r.p_val    is None else float(r.p_val)     for r in rows], type=pa.float64())

        list_f64 = pa.list_(pa.float64())

        # helper: optional list-typed arrays
        def _opt_list(getter):
            out = []
            for r in rows:
                arr = getter(r)
                out.append(None if arr is None else np.asarray(arr, dtype=np.float64).tolist())
            return out

        cr        = pa.array(_opt_list(lambda r: r.cr),        type=list_f64)
        cr_bnd    = pa.array(_opt_list(lambda r: r.cr_bnd),    type=list_f64)
        num_all   = pa.array(_opt_list(lambda r: r.num_all),   type=list_f64)
        den_all   = pa.array(_opt_list(lambda r: r.den_all),   type=list_f64)
        num_bnd   = pa.array(_opt_list(lambda r: r.num_bnd),   type=list_f64)
        den_bnd   = pa.array(_opt_list(lambda r: r.den_bnd),   type=list_f64)

        com       = pa.array(_opt_list(lambda r: r.com),       type=list_f64)
        bd_r      = pa.array(_opt_list(lambda r: r.bd_r),      type=list_f64)
        bd_counts = pa.array(_opt_list(lambda r: r.bd_counts), type=list_f64)

        rg_area   = pa.array([None if r.rg_area is None else float(r.rg_area) for r in rows], type=pa.float64())
        rg_bnd    = pa.array([None if r.rg_bnd  is None else float(r.rg_bnd)  for r in rows], type=pa.float64())

        bd_bin_width = pa.array([None if r.bd_bin_width is None else float(r.bd_bin_width) for r in rows], type=pa.float64())
        center_method = pa.array([None if r.center_method is None else str(r.center_method) for r in rows], type=pa.string())
        boundary_connectivity = pa.array([None if r.boundary_connectivity is None else str(r.boundary_connectivity) for r in rows], type=pa.string())
        bd_n = pa.array([None if r.bd_n is None else int(r.bd_n) for r in rows], type=pa.int64())

        return pa.table({
            "cloud_idx": cloud_idx,
            "perim":     perim,
            "area":      area,
            "threshold": threshold,
            "p_val":     p_val,

            "cr":        cr,
            "cr_bnd":    cr_bnd,
            "num_all":   num_all,
            "den_all":   den_all,
            "num_bnd":   num_bnd,
            "den_bnd":   den_bnd,

            "com":       com,
            "rg_area":   rg_area,
            "rg_bnd":    rg_bnd,
            "bd_r":      bd_r,
            "bd_counts": bd_counts,

            "bd_bin_width": bd_bin_width,
            "center_method": center_method,
            "boundary_connectivity": boundary_connectivity,
            "bd_n": bd_n,
        })


# =========================
# Boundary radial profiles
# =========================

def compute_com(mask_uint8) -> Tuple[float, float]:
    """
    Center of mass for a tight, flood-filled cloud (row/col floats).
    Returns: (cy, cx)
    """
    m = to_numpy((mask_uint8 != 0).astype(np.uint8, copy=False))
    if not m.any():
        raise ValueError("compute_com: empty mask.")
    ys, xs = np.nonzero(m)
    return float(ys.mean()), float(xs.mean())


def boundary_distances_min(
    mask_uint8,
    center: Optional[Tuple[float, float]] = None,
    center_method: Literal["com"] = "com",
) -> np.ndarray:
    """
    Distances from center to 4-connected boundary pixels (host float64).
    Returns: (Nb,)
    """
    # boundary mask on xp (gpu/cpu)
    if xp_backend == "cupy":
        import cupy as cp
        mask_uint8 = cp.asarray(mask_uint8, dtype=cp.uint8)
    else:
        mask_uint8 = np.asarray(mask_uint8, dtype=np.uint8)

    bmask = make_boundary_mask4(mask_uint8)
    ys_xp, xs_xp = xp.nonzero(bmask)
    if ys_xp.size == 0:
        return np.zeros(0, dtype=np.float64)

    # center (COM for minimal path)
    if center is None:
        cy, cx = compute_com(mask_uint8)
    else:
        cy, cx = float(center[0]), float(center[1])

    # move coords to host
    ys = to_numpy(ys_xp).astype(np.float64, copy=False)
    xs = to_numpy(xs_xp).astype(np.float64, copy=False)

    # Euclidean distances to center
    return np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)


def boundary_histogram_min(
    r: np.ndarray,
    bin_width: float = 1.0,
    include_zero_bin: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal histogram for boundary distances.
    - zero-origin alignment (shared across clouds)
    - returns centers and raw counts (no PDF, no ring-correction)
    """
    r = np.asarray(r, dtype=np.float64)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rmax = float(np.ceil(r.max()))
    start = 0.0 if include_zero_bin else float(np.floor(r.min() / bin_width) * bin_width)

    edges = np.arange(start, rmax + bin_width + 1e-12, bin_width, dtype=np.float64)
    h, _ = np.histogram(r, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h.astype(np.float64, copy=False)


def radius_of_gyration(
    mask_uint8,
    center: Optional[Tuple[float, float]] = None,
    pixels: Literal["area", "boundary"] = "area",
) -> float:
    """
    Rg about center (defaults to COM). 'pixels' chooses contributors.
    - 'area'     : all mass pixels
    - 'boundary' : only boundary pixels
    """
    # center
    if center is None:
        cy, cx = compute_com(mask_uint8)
    else:
        cy, cx = float(center[0]), float(center[1])

    # contributors
    if pixels == "area":
        m = to_numpy((mask_uint8 != 0).astype(np.uint8, copy=False))
        ys, xs = np.nonzero(m)
    elif pixels == "boundary":
        bmask = make_boundary_mask4(mask_uint8.astype(xp.uint8, copy=False))
        ys_xp, xs_xp = xp.nonzero(bmask)
        ys = to_numpy(ys_xp)
        xs = to_numpy(xs_xp)
    else:
        raise ValueError("radius_of_gyration: pixels must be 'area' or 'boundary'.")

    if ys.size == 0:
        return 0.0

    dy = ys.astype(np.float64, copy=False) - cy
    dx = xs.astype(np.float64, copy=False) - cx
    return float(np.sqrt(np.mean(dy * dy + dx * dx)))