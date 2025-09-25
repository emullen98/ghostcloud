#!/usr/bin/env python3
# plot_edge_binarize.py
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


# Adjust this import to match your project structure
# e.g., from clouds.utils.image_utils import edge_binarize
from utils.image_utils import edge_binarize


def _load_raw(path: str) -> np.ndarray:
    """
    Load a grayscale image from common formats or a numpy array from .npy/.npz.
    Returns float32 grayscale array (no normalization applied).
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in [".npy"]:
        arr = np.load(str(p))
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
        return arr.astype(np.float32)

    if suffix in [".npz"]:
        data = np.load(str(p))
        # try common keys or take first array
        for key in ("image", "img", "arr", "data"):
            if key in data:
                arr = data[key]
                break
        else:
            # take the first item
            key = list(data.keys())[0]
            arr = data[key]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
        return arr.astype(np.float32)

    # Fallback: read via OpenCV
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read file: {path}")

    # Convert to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float32 for downstream ops
    return img.astype(np.float32)


def _edge_magnitude(image_f32: np.ndarray, method: str) -> np.ndarray | None:
    """
    Compute edge/contrast magnitude for visualization (Sobel/Laplacian only).
    Returns None for methods that don't have a simple magnitude (e.g., Canny).
    """
    if method == "sobel":
        gx = cv2.Sobel(image_f32, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image_f32, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)
    elif method == "laplacian":
        lap = cv2.Laplacian(image_f32, cv2.CV_32F, ksize=3)
        return np.abs(lap)
    else:
        return None  # e.g., canny


def _auto_threshold_from_percentile(mag: np.ndarray, pct: float) -> float:
    flat = mag.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("inf")
    return float(np.percentile(flat, pct))


def visualize_edge_binarize(
    image_path: str,
    method: str = "sobel",
    thresh: float = 20.0,
    auto_pct: float | None = None,
    cmap_mag: str = "magma",
    save_path: str | None = None,
    show: bool = True,
    norm_clip_pct: float = 99.5,
):
    img = _load_raw(image_path)
    img_u8_for_canny = np.clip(img, 0, 255).astype(np.uint8)

    # For Sobel/Laplacian, support auto percentile by computing magnitude here.
    mag = _edge_magnitude(img, method)
    eff_thresh = thresh

    if auto_pct is not None and mag is not None:
        eff_thresh = _auto_threshold_from_percentile(mag, auto_pct)

    # Call your binarizer with the effective threshold
    # Note: for "canny", your edge_binarize likely interprets `thresh` as the lower threshold.
    if method == "canny":
        mask = edge_binarize(img_u8_for_canny, method=method, thresh=eff_thresh)
    else:
        mask = edge_binarize(img, method=method, thresh=eff_thresh)

    # Build the figure
    if mag is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        ax0, ax1 = axes

        # Original (auto-scale for viz)
        ax0.imshow(img, cmap="gray")
        ax0.set_title("Original")
        ax0.axis("off")

        ax1.imshow(mask, cmap="gray")
        if auto_pct is not None:
            ax1.set_title(f"Mask ({method}, auto p{auto_pct:.1f} → {eff_thresh:.2f})")
        else:
            ax1.set_title(f"Mask ({method}, thresh={eff_thresh:.2f})")
        ax1.axis("off")

    else:
        # Normalize magnitude just for display
        denom = np.percentile(mag, norm_clip_pct) + 1e-8
        mag_viz = np.clip(mag / denom, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
        ax0, ax1, ax2 = axes

        ax0.imshow(img, cmap="gray")
        ax0.set_title("Original")
        ax0.axis("off")

        im1 = ax1.imshow(mag_viz, cmap=cmap_mag)
        ax1.set_title(f"{method.capitalize()} magnitude")
        ax1.axis("off")
        cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("norm. mag", rotation=270, labelpad=12)

        ax2.imshow(mask, cmap="gray")
        if auto_pct is not None:
            ax2.set_title(f"Mask ({method}, auto p{auto_pct:.1f} → {eff_thresh:.2f})")
        else:
            ax2.set_title(f"Mask ({method}, thresh={eff_thresh:.2f})")
        ax2.axis("off")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="Quick visualizer for edge/contrast binarizer.")
    p.add_argument("--image", "-i", required=True, help="Path to input raw file (.png/.tif/.jpg/.npy/.npz)")
    p.add_argument("--method", "-m", default="sobel", choices=["sobel", "laplacian", "canny"])
    p.add_argument("--thresh", "-t", type=float, default=20.0, help="Threshold (lower for canny).")
    p.add_argument("--auto-pct", type=float, default=None,
                   help="If set (e.g., 90), use that percentile of edge magnitude for Sobel/Laplacian.")
    p.add_argument("--save", type=str, default=None, help="Optional path to save the figure.")
    p.add_argument("--no-show", action="store_true", help="Do not display the figure; useful for batch runs.")
    p.add_argument("--cmap-mag", type=str, default="magma", help="Matplotlib colormap for magnitude.")
    p.add_argument("--norm-clip-pct", type=float, default=99.5,
                   help="Clip percentile for magnitude visualization scaling.")
    return p.parse_args()


def main():
    args = _parse_args()
    visualize_edge_binarize(
        image_path=args.image,
        method=args.method,
        thresh=args.thresh,
        auto_pct=args.auto_pct,
        cmap_mag=args.cmap_mag,
        save_path=args.save,
        show=not args.no_show,
        norm_clip_pct=args.norm_clip_pct,
    )


if __name__ == "__main__":
    main()
