#!/usr/bin/env python3
"""
Apply Yen threshold to all images in a directory and save binary PNGs.
Output names: <orig_stem>_yen_t{THRESH}.png  (e.g., foo_yen_t0p773.png)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from clouds.utils.image_utils import threshold_yen  # returns dict with 'mask' and 'threshold'


def load_gray(path: Path) -> np.ndarray:
    """Load image as grayscale float32."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32, copy=False)


def save_png_bool_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean mask as {0,255} uint8 PNG."""
    mask_u8 = (mask.astype(np.uint8) * 255)  # True->255, False->0
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask_u8)
    if not ok:
        raise IOError(f"Failed to write {path}")


def main():
    p = argparse.ArgumentParser(description="Batch Yen binarizer (serial).")
    p.add_argument("--indir", required=True, help="Input directory of images")
    p.add_argument("--outdir", required=True, help="Output directory for PNG masks")
    p.add_argument("--exts", default=".png,.jpg,.jpeg,.tif,.tiff,.bmp",
                   help="Comma-separated extensions to include")
    args = p.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.exts.split(",")}

    files = [p for p in sorted(indir.iterdir())
             if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"[WARN] No images found in {indir} with {sorted(exts)}")
        return

    for src in files:
        img = load_gray(src)
        result = threshold_yen(img)              # dict with 'mask' (bool) and 'threshold' (float on rescaled domain)
        t = float(result["threshold"])
        mask = result["mask"]

        # Filename threshold string with 3 decimals, dot => 'p'  (e.g., 0.773 -> '0p773')
        t_str = f"{t:.3f}".replace(".", "p")
        dst = outdir / f"{src.stem}_yen_t{t_str}.png"

        save_png_bool_mask(mask, dst)
        print(f"[OK] {src.name} -> {dst.name} (t={t:.6f})")


if __name__ == "__main__":
    main()
