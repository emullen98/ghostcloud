#!/usr/bin/env python3
"""
Apply Yen threshold to all images in a directory and save binary PNGs.
Outputs are named with the original stem + "_yen_t{THRESH}" suffix.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from clouds.utils.image_utils import threshold_yen

def load_gray(path: Path) -> np.ndarray:
    """Load image as grayscale float32."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32, copy=False)

def save_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask)
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
    exts = [e if e.startswith(".") else "."+e for e in args.exts.split(",")]

    files = [p for p in indir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"[WARN] No images found in {indir} with {exts}")
        return

    for src in files:
        img = load_gray(src)
        t = float(yen_threshold(img))
        mask = (img > t).astype(np.uint8) * 255
        t_str = f"{t:.4f}".replace(".", "p")
        dst = outdir / f"{src.stem}_yen_t{t_str}.png"
        save_png(mask, dst)
        print(f"[OK] {src.name} -> {dst.name}")

if __name__ == "__main__":
    main()
