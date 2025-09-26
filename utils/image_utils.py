import numpy as np
from typing import Optional, Tuple, Dict, Union
import cv2
from pathlib import Path

def _rescale_percentiles(
    img: np.ndarray, 
    percentiles: Optional[Tuple[float, float]] = (1.0, 99.0)
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Rescale image using percentile-based contrast stretching.
    
    Args:
        img: Input image array
        percentiles: (low, high) percentiles for contrast stretching.
                    If None, skip rescaling but return range.
    
    Returns:
        Tuple of (rescaled_image, (min_val, max_val))
    """
    img_float = img.astype(np.float64)
    
    if percentiles is None:
        lo, hi = np.nanmin(img_float), np.nanmax(img_float)
        return img_float, (float(lo), float(hi))
    
    lo, hi = np.nanpercentile(img_float, percentiles)
    
    if lo >= hi:  # Flat or degenerate image
        return img_float, (float(lo), float(hi))
        
    # Rescale to [0,1]
    rescaled = (img_float - lo) / (hi - lo)
    np.clip(rescaled, 0, 1, out=rescaled)
    
    return rescaled, (float(lo), float(hi))

def binarize_by_fill_fraction(
    img: np.ndarray,
    p_target: float = 0.407,
    rescale_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    nan_as_background: bool = True
) -> Dict[str, Union[np.ndarray, float, str, Tuple[float, float]]]:
    """
    Binarize image by targeting a specific fill fraction using quantile thresholding.
    
    Args:
        img: Input grayscale image
        p_target: Target fill fraction (0 < p_target < 1)
        rescale_percentiles: Percentiles for contrast stretching
        nan_as_background: If True, treat NaN values as background
    
    Returns:
        Dictionary containing:
            - mask: Binary mask
            - threshold: Computed threshold value
            - fill_fraction: Actual fill fraction achieved
            - method: Always "quantile"
            - rescaled_range: (min, max) of rescaled image
            
    Raises:
        ValueError: If p_target invalid or image is empty
    """
    if not 0 < p_target < 1:
        raise ValueError("p_target must be between 0 and 1")
    if np.all(~np.isfinite(img)):
        raise ValueError("Input image is empty (all NaN)")
        
    rescaled, (lo, hi) = _rescale_percentiles(img, rescale_percentiles)
    
    # Compute threshold as inverse quantile
    threshold = float(np.nanquantile(rescaled, 1.0 - p_target))
    
    # Create mask
    mask = rescaled >= threshold
    if nan_as_background:
        mask = np.where(np.isfinite(rescaled), mask, False)
        
    return {
        "mask": mask,
        "threshold": threshold,
        "fill_fraction": float(np.nanmean(mask)),
        "method": "quantile",
        "rescaled_range": (lo, hi)
    }

def threshold_yen(
    img: np.ndarray,
    rescale_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    nan_as_background: bool = True
) -> Dict[str, Union[np.ndarray, float, str, Tuple[float, float]]]:
    """
    Apply Yen's thresholding method to image.
    
    Args:
        img: Input grayscale image
        rescale_percentiles: Percentiles for contrast stretching
        nan_as_background: If True, treat NaN values as background
    
    Returns:
        Dictionary containing:
            - mask: Binary mask
            - threshold: Yen threshold value
            - fill_fraction: Resulting fill fraction
            - method: Always "yen"
            - rescaled_range: (min, max) of rescaled image
            
    Raises:
        ImportError: If scikit-image is not available
    """
    try:
        from skimage.filters import threshold_yen
    except ImportError:
        raise ImportError(
            "scikit-image is required for threshold_yen_only; "
            "install via 'pip install scikit-image'"
        )
    
    rescaled, (lo, hi) = _rescale_percentiles(img, rescale_percentiles)
    
    # Handle flat image case
    if lo == hi:
        threshold = hi
    else:
        threshold = float(threshold_yen(rescaled))
    
    mask = rescaled >= threshold
    if nan_as_background:
        mask = np.where(np.isfinite(rescaled), mask, False)
        
    return {
        "mask": mask,
        "threshold": threshold,
        "fill_fraction": float(np.nanmean(mask)),
        "method": "yen",
        "rescaled_range": (lo, hi)
    }

def binarize_cloud_method2(
    img: np.ndarray,
    p_target: float = 0.407,
    tol: float = 0.05,
    rescale_percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    nan_as_background: bool = True
) -> Dict[str, Union[np.ndarray, float, str, Tuple[float, float]]]:
    """
    Binarize cloud image using Yen's method with percolation-aware fallback.
    
    This method attempts to use Yen's thresholding first, but falls back to
    quantile-based thresholding if the resulting fill fraction deviates too much
    from the target. The target fill fraction can be chosen based on percolation
    theory:
    - For 4-connectivity: p_c ≈ 0.593
    - For 8-connectivity: p_c ≈ 0.407
    
    Choose p_target near these values depending on your connectivity model.
    
    Args:
        img: Input grayscale image
        p_target: Target fill fraction (0 < p_target < 1)
        tol: Tolerance for fill fraction deviation (0 <= tol < 1)
        rescale_percentiles: Percentiles for contrast stretching
        nan_as_background: If True, treat NaN values as background
    
    Returns:
        Dictionary containing:
            - mask: Binary mask
            - threshold: Computed threshold
            - fill_fraction: Actual fill fraction
            - method: One of ["yen", "quantile_fallback", "quantile_no_skimage"]
            - rescaled_range: (min, max) of rescaled image
            
    Raises:
        ValueError: If p_target or tol are invalid
    """
    if not 0 < p_target < 1:
        raise ValueError("p_target must be between 0 and 1")
    if not 0 <= tol < 1:
        raise ValueError("tol must be between 0 and 1")
        
    try:
        # Try Yen first
        result = threshold_yen(
            img, 
            rescale_percentiles=rescale_percentiles,
            nan_as_background=nan_as_background
        )
        
        # Check if fill fraction is acceptable
        if abs(result["fill_fraction"] - p_target) <= tol:
            return result
            
        # Fall back to quantile with modified method name
        result = binarize_by_fill_fraction(
            img,
            p_target=p_target,
            rescale_percentiles=rescale_percentiles,
            nan_as_background=nan_as_background
        )
        result["method"] = "quantile_fallback"
        return result
        
    except ImportError:
        # Fallback if scikit-image not available
        result = binarize_by_fill_fraction(
            img,
            p_target=p_target,
            rescale_percentiles=rescale_percentiles,
            nan_as_background=nan_as_background
        )
        result["method"] = "quantile_no_skimage"
        return result
    



def edge_binarize(image: np.ndarray, method="sobel", thresh=20) -> np.ndarray:
    """
    Simple edge/contrast-based binarizer.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (uint8 or float).
    method : str, optional
        Edge detection method. Options: "sobel", "laplacian", "canny".
    thresh : float, optional
        Threshold on edge magnitude (for sobel/laplacian). 
        For canny, this is used as the lower threshold.

    Returns
    -------
    mask : np.ndarray
        Binary mask of detected edges (uint8, {0,255}).
    """

    # Ensure grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)

    if method == "sobel":
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mask = (mag > thresh).astype(np.uint8) * 255

    elif method == "laplacian":
        lap = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
        mag = np.abs(lap)
        mask = (mag > thresh).astype(np.uint8) * 255

    elif method == "canny":
        # Here `thresh` is the lower threshold, set upper = 3×lower
        mask = cv2.Canny(image.astype(np.uint8),
                         threshold1=thresh,
                         threshold2=3*thresh)

    else:
        raise ValueError(f"Unknown method: {method}")

    return mask


def save_lattice_png(lattice: np.ndarray, path: str) -> None:
    """
    Save a binary lattice (0/1 array) as a PNG image.

    Parameters
    ----------
    lattice : np.ndarray
        2D array with values 0/1 (binary lattice).
    path : str
        Output file path (should end with .png).

    Notes
    -----
    - Lattice is converted to uint8 with values {0, 255}.
    - Saved as grayscale PNG (CV_8U).
    - PNG is inherently lossless, no extra compression options needed.
    """
    # Ensure array is binary {0,1}
    arr = (lattice > 0).astype(np.uint8) * 255

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as grayscale PNG
    success = cv2.imwrite(str(out_path), arr)
    if not success:
        raise IOError(f"Failed to save lattice to {out_path}")

def load_gray_float(path: Path) -> np.ndarray:
    """
    Load image from path as grayscale float32 (no normalization).
    If already grayscale, preserves it. Otherwise converts from BGR.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32, copy=False)

def normalize_to_unit(img: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0,1] range. If flat (all values equal), returns zeros.
    """
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img, dtype=np.float32)

def threshold_binary(img: np.ndarray, thr: float) -> np.ndarray:
    """
    Apply threshold to normalized grayscale image.
    
    Parameters
    ----------
    img : np.ndarray
        Grayscale float32 image, assumed in [0,1].
    thr : float
        Threshold in [0,1]. Pixels > thr become 1, else 0.
    
    Returns
    -------
    np.ndarray of dtype uint8 with values {0,1}
    """
    if not (0.0 <= thr <= 1.0):
        raise ValueError(f"Threshold {thr} not in [0,1]")
    return (img > thr)
