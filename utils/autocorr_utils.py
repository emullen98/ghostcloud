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

from dataclasses import dataclass
from . import config
from typing import Tuple, List, Literal, Optional, Dict
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.ndimage as ndimage
from clouds.utils.cloud_utils import compute_perimeter as compute_perimeter4c


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
def make_boundary_mask4(cloud_uint8):
    """
    4-connected boundary mask: select cloud pixels that have <4 filled
    N/S/E/W neighbors (zeros outside count as background).

    Parameters
    ----------
    cloud_uint8 : (H,W) uint8 in {0,1}

    Returns
    -------
    boundary_uint8 : (H,W) uint8 in {0,1}
    """
    m = (cloud_uint8 != 0).astype(xp.uint8, copy=False)

    # Try ndimage first (fast & concise on both CPU/GPU)
    ndimage = None
    try:
        if xp.__name__ == "cupy":
            import cupyx.scipy.ndimage as ndimage  # GPU path
        else:
            import scipy.ndimage as ndimage        # CPU path
    except Exception:
        ndimage = None

    if ndimage is not None:
        k4 = xp.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=xp.uint8)
        neigh = ndimage.convolve(m, k4, mode="constant", cval=0)
        boundary = (m & (neigh < 4)).astype(xp.uint8, copy=False)
        return boundary

    # Fallback (no ndimage): zero-padded shifts
    def _shift_zeros(a, dy, dx):
        H, W = a.shape
        out = xp.zeros_like(a)
        y0s = max(0, -dy); y1s = min(H, H - dy)
        x0s = max(0, -dx); x1s = min(W, W - dx)
        y0d = max(0,  dy); y1d = min(H, H + dy)
        x0d = max(0,  dx); x1d = min(W, W + dx)
        if y0s < y1s and x0s < x1s:
            out[y0d:y1d, x0d:x1d] = a[y0s:y1s, x0s:x1s]
        return out

    up    = _shift_zeros(m, -1,  0)
    down  = _shift_zeros(m,  1,  0)
    left  = _shift_zeros(m,  0, -1)
    right = _shift_zeros(m,  0,  1)
    neigh = (up + down + left + right)
    boundary = (m & (neigh < 4)).astype(xp.uint8, copy=False)
    return boundary

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
    """
    Per-cloud row with C(r), optional boundary C(r), and optional boundary radial profile.

    Geometry
    --------
    - cloud_idx : int
    - perim     : int (4-connected, pixel-edge units)
    - area      : int

    Optional Correlations
    ---------------------
    - cr        : np.ndarray | None        # C(r) for centers = all cloud pixels
    - cr_bnd    : np.ndarray | None        # C(r) for centers = boundary pixels

    Optional Numerators/Denominators (for downstream aggregation)
    -------------------------------------------------------------
    - num_all   : np.ndarray | None
    - den_all   : np.ndarray | None
    - num_bnd   : np.ndarray | None
    - den_bnd   : np.ndarray | None

    Optional Experiment Tags
    ------------------------
    - threshold : float | None            # used by image-thresholding pipeline (nullable)
    - p_val     : float | None            # used by site-percolation pipeline (nullable)

    Optional Boundary Radial Profile (this request)
    -----------------------------------------------
    - rp_r       : np.ndarray | None   # r-bin centers
    - rp_counts  : np.ndarray | None   # raw counts per r-bin
    - rp_pdf     : np.ndarray | None   # counts normalized as a density over r
    - rp_f_ring  : np.ndarray | None   # optional ring-corrected series (counts / (2π r Δr))
    """
    cloud_idx: int
    perim: int
    area: int

    # Optional C(r) arrays
    cr: Optional[np.ndarray] = None
    cr_bnd: Optional[np.ndarray] = None

    # Optional raw accumulators
    num_all: Optional[np.ndarray] = None
    den_all: Optional[np.ndarray] = None
    num_bnd: Optional[np.ndarray] = None
    den_bnd: Optional[np.ndarray] = None

    # Optional experiment tags
    threshold: Optional[float] = None
    p_val: Optional[float] = None

    # Optional boundary radial profile
    rp_r: Optional[np.ndarray] = None
    rp_counts: Optional[np.ndarray] = None
    rp_pdf: Optional[np.ndarray] = None
    rp_f_ring: Optional[np.ndarray] = None


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
        # normalize dtype for all present arrays
        def _as_f64(x):
            return None if x is None else np.asarray(x, dtype=np.float64)

        row.cr       = _as_f64(row.cr)
        row.cr_bnd   = _as_f64(row.cr_bnd)
        row.num_all  = _as_f64(row.num_all)
        row.den_all  = _as_f64(row.den_all)
        row.num_bnd  = _as_f64(row.num_bnd)
        row.den_bnd  = _as_f64(row.den_bnd)

        # NEW: radial profile arrays
        row.rp_r       = _as_f64(row.rp_r)
        row.rp_counts  = _as_f64(row.rp_counts)
        row.rp_pdf     = _as_f64(row.rp_pdf)
        row.rp_f_ring  = _as_f64(row.rp_f_ring)

        # queue row
        self._rows.append(row)

        # update size estimate conservatively
        size = 32  # base overhead
        for arr in (row.cr, row.cr_bnd,
                    row.num_all, row.den_all, row.num_bnd, row.den_bnd,
                    row.rp_r, row.rp_counts, row.rp_pdf, row.rp_f_ring):
            if arr is not None:
                size += arr.nbytes + 16
        self._approx_bytes += size

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
        thr_py    = [None if (r.threshold is None) else float(r.threshold) for r in rows]
        pval_py   = [None if (r.p_val    is None) else float(r.p_val)     for r in rows]
        threshold = pa.array(thr_py, type=pa.float64())
        p_val     = pa.array(pval_py, type=pa.float64())

        # helper to coerce optional 1D numpy arrays -> python lists (or None)
        def _opt_list(getter):
            out = []
            for r in rows:
                arr = getter(r)
                out.append(None if arr is None else arr.tolist())
            return out

        list_f64 = pa.list_(pa.float64())

        cr        = pa.array(_opt_list(lambda r: r.cr),        type=list_f64)
        cr_bnd    = pa.array(_opt_list(lambda r: r.cr_bnd),    type=list_f64)
        num_all   = pa.array(_opt_list(lambda r: r.num_all),   type=list_f64)
        den_all   = pa.array(_opt_list(lambda r: r.den_all),   type=list_f64)
        num_bnd   = pa.array(_opt_list(lambda r: r.num_bnd),   type=list_f64)
        den_bnd   = pa.array(_opt_list(lambda r: r.den_bnd),   type=list_f64)

        # NEW: radial profile columns
        rp_r       = pa.array(_opt_list(lambda r: r.rp_r),       type=list_f64)
        rp_counts  = pa.array(_opt_list(lambda r: r.rp_counts),  type=list_f64)
        rp_pdf     = pa.array(_opt_list(lambda r: r.rp_pdf),     type=list_f64)
        rp_f_ring  = pa.array(_opt_list(lambda r: r.rp_f_ring),  type=list_f64)

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

            "rp_r":       rp_r,
            "rp_counts":  rp_counts,
            "rp_pdf":     rp_pdf,
            "rp_f_ring":  rp_f_ring,
        })


# =========================
# Boundary radial profiles
# =========================

def compute_center_for_cloud(
    cloud_uint8,
    method: Literal["com","max_inscribed"] = "com"
) -> Tuple[float, float]:
    """
    Center for a flood-filled, tightly-cropped cloud.
    Returns (cy, cx) as Python floats (row, col).

    method:
      - "com"           : center of mass
      - "max_inscribed" : argmax of Euclidean distance transform inside cloud
    """
    # ensure NumPy view for center math / EDT
    m_np = to_numpy((cloud_uint8 != 0).astype(np.uint8, copy=False))
    if not m_np.any():
        raise ValueError("compute_center_for_cloud: empty mask.")

    if method == "com":
        ys, xs = np.nonzero(m_np)
        return float(ys.mean()), float(xs.mean())

    if method == "max_inscribed":
        dt = ndimage.distance_transform_edt(m_np)
        cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        return float(cy), float(cx)

    raise ValueError(f"Unknown center method: {method}")


def boundary_distances(
    cloud_uint8,
    center: Optional[Tuple[float,float]] = None,
    center_method: Literal["com","max_inscribed"] = "com",
    return_coords: bool = False
) -> Dict[str, np.ndarray]:
    """
    Distances from chosen center to *boundary pixels* of a tight, flood-filled cloud.

    Returns (NumPy on host):
      - 'dist'     : (N,) float64 distances
      - 'center'   : (2,) float64 (cy, cx)
      - 'area'     : int area (|S_i|)
      - 'perim_4c' : int 4-connected perimeter (pixel-edge units) via cloud utils
      - 'coords'   : optional (N,2) int [y,x] boundary pixel coords
    """
    # boundary via your existing helper (xp-aware)
    bmask = make_boundary_mask4(cloud_uint8.astype(xp.uint8, copy=False))

    ys_xp, xs_xp = xp.nonzero(bmask)
    if ys_xp.size == 0:
        raise ValueError("boundary_distances: no boundary pixels found.")

    # center
    if center is None:
        cy, cx = compute_center_for_cloud(cloud_uint8, method=center_method)
    else:
        cy, cx = float(center[0]), float(center[1])

    # move coords to host for distance math
    ys = to_numpy(ys_xp).astype(np.float64, copy=False)
    xs = to_numpy(xs_xp).astype(np.float64, copy=False)
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)

    # area & perimeter via cloud utils (on NumPy bool array)
    cloud_np_bool = to_numpy(cloud_uint8).astype(bool, copy=False)
    area = int(np.count_nonzero(cloud_np_bool))
    perim_4c = int(compute_perimeter4c(cloud_np_bool))

    out = {
        "dist": dist,
        "center": np.array([cy, cx], dtype=np.float64),
        "area": area,
        "perim_4c": perim_4c,
    }
    if return_coords:
        out["coords"] = np.stack([ys.astype(np.int64, copy=False),
                                  xs.astype(np.int64, copy=False)], axis=1)
    return out


def radial_histogram_boundary(
    r: np.ndarray,
    bin_width: float = 1.0,
    r_max: Optional[float] = None,
    include_zero_bin: bool = False,
    make_pdf: bool = True,
    ring_corrected: bool = False,
    subpixel_splat: bool = True
) -> Dict[str, np.ndarray]:
    """
    Build radial histogram(s) from boundary distances (NumPy in/out).
      - 'counts' (raw) is the primary curve to inspect first.
      - 'pdf' integrates to 1 over r (if make_pdf=True).
      - Optional 'f_ring' = counts / (2π r Δr).
      - 'subpixel_splat' linearly splits samples between neighboring bins.
    """
    if r.size == 0:
        raise ValueError("radial_histogram_boundary: empty distance array.")
    r = np.asarray(r, dtype=np.float64)

    rmax = float(np.ceil(r.max())) if r_max is None else float(r_max)
    start = 0.0 if include_zero_bin else float(np.floor(r.min()/bin_width)*bin_width)

    edges = np.arange(start, rmax + bin_width + 1e-12, bin_width, dtype=np.float64)
    centers = 0.5*(edges[:-1] + edges[1:])
    delta = (edges[1:] - edges[:-1])

    K = centers.size
    counts = np.zeros(K, dtype=np.float64)

    if not subpixel_splat:
        h, _ = np.histogram(r, bins=edges)
        counts = h.astype(np.float64, copy=False)
    else:
        idxf = (r - start) / bin_width
        i0 = np.floor(idxf).astype(np.int64)
        frac = idxf - i0
        m0 = (i0 >= 0) & (i0 < K)
        if m0.any():
            counts += np.bincount(i0[m0], weights=(1.0 - frac[m0]), minlength=K)
        i1 = i0 + 1
        m1 = (i1 >= 0) & (i1 < K)
        if m1.any():
            counts += np.bincount(i1[m1], weights=frac[m1], minlength=K)

    result = {"bin_edges": edges, "r_centers": centers, "counts": counts}

    if make_pdf:
        total = counts.sum()
        result["pdf"] = counts / (total * delta) if total > 0 else np.zeros_like(counts)

    if ring_corrected:
        ring_len = 2.0*np.pi*centers*delta
        safe = ring_len > 0
        f_ring = np.zeros_like(counts)
        f_ring[safe] = counts[safe] / ring_len[safe]
        result["f_ring"] = f_ring
        result["ring_len"] = ring_len

    return result


def radius_scale(
    r_centers: np.ndarray,
    mask_area: int,
    method: Literal["Req","percentile"] = "Req",
    p_low: float = 5.0,
    p_high: float = 95.0
) -> np.ndarray:
    """
    r -> s scaling for within-cloud overlays (not used until you compare clouds).
      - "Req"        : s = r / R_eq, with R_eq = sqrt(area/pi)
      - "percentile" : s = (r - r_pLow)/(r_pHigh - r_pLow)
    """
    r_centers = np.asarray(r_centers, dtype=np.float64)
    if method == "Req":
        R_eq = np.sqrt(float(mask_area) / np.pi)
        if not np.isfinite(R_eq) or R_eq <= 0:
            raise ValueError("radius_scale: non-positive R_eq.")
        return r_centers / R_eq
    elif method == "percentile":
        rlo, rhi = np.percentile(r_centers, [p_low, p_high])
        denom = max(rhi - rlo, 1e-9)
        return (r_centers - rlo) / denom
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def boundary_radial_profile(
    cloud_uint8,
    bin_width: float = 1.0,
    center_method: Literal["com","max_inscribed"] = "com",
    ring_corrected: bool = False,
    subpixel_splat: bool = True
) -> Dict[str, np.ndarray]:
    """
    One-call profile for a single cloud.
    Returns:
      'r_centers','counts','pdf' (and optionally 'f_ring','ring_len') +
      'area','perim_4c','center'
    """
    d = boundary_distances(
        cloud_uint8,
        center=None,
        center_method=center_method,
        return_coords=False
    )
    h = radial_histogram_boundary(
        d["dist"],
        bin_width=bin_width,
        make_pdf=True,
        ring_corrected=ring_corrected,
        subpixel_splat=subpixel_splat
    )
    return {**h, "area": d["area"], "perim_4c": d["perim_4c"], "center": d["center"]}
