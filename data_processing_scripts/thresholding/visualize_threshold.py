#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.image_utils import binarize_by_fill_fraction, threshold_yen, binarize_cloud_method2, save_lattice_png

def visualize_thresholding(image_path: str, p_target: float = 0.407, tol: float = 0.05) -> None:
    """
    Visualize cloud image thresholding with all methods.
    Shows original, Yen's method, and quantile-based thresholding.
    
    Args:
        image_path: Path to input image file
        p_target: Target fill fraction for thresholding
        tol: Tolerance for Yen method acceptance
    """
    # Load image
    try:
        img = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
        # Invert image so clouds are white (high values)
        #img = 255 - img
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        sys.exit(1)

    # Apply all thresholding methods
    try:
        yen_result = threshold_yen(img)
    except ImportError:
        print("Warning: scikit-image not available, skipping Yen's method")
        yen_result = None

    quantile_result = binarize_by_fill_fraction(img, p_target=p_target)
    method2_result = binarize_cloud_method2(img, p_target=p_target, tol=tol)

    # Print detailed results
    print("\nThresholding Results:")
    print("-" * 70)
    print("Yen's Method:")
    if yen_result:
        print(f"  Threshold value:        {yen_result['threshold']:.4f}")
        print(f"  Fill fraction:          {yen_result['fill_fraction']:.4f}")
    else:
        print("  Not available (scikit-image required)")
    
    print("\nQuantile Method:")
    print(f"  Threshold value:        {quantile_result['threshold']:.4f}")
    print(f"  Target fill fraction:   {p_target:.4f}")
    print(f"  Achieved fill fraction: {quantile_result['fill_fraction']:.4f}")
    
    print("\nMethod 2 (Combined):")
    print(f"  Selected method:        {method2_result['method']}")
    print(f"  Threshold value:        {method2_result['threshold']:.4f}")
    print(f"  Achieved fill fraction: {method2_result['fill_fraction']:.4f}")
    print(f"  Image rescaled range:   [{method2_result['rescaled_range'][0]:.4f}, "
          f"{method2_result['rescaled_range'][1]:.4f}]")
    print("-" * 70)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    im1 = axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image\n(White = Cloud)')
    plt.colorbar(im1, ax=axes[0])
    axes[0].axis('off')
    
    # Yen's method
    if yen_result:
        im2 = axes[1].imshow(yen_result['mask'], cmap='binary')
        title = f"Yen's Method\nFill: {yen_result['fill_fraction']:.3f}"
    else:
        im2 = axes[1].imshow(np.zeros_like(img), cmap='binary')
        title = "Yen's Method\n(Not Available)"
    axes[1].set_title(title)
    plt.colorbar(im2, ax=axes[1])
    axes[1].axis('off')
    
    # Quantile method
    im3 = axes[2].imshow(quantile_result['mask'], cmap='binary')
    axes[2].set_title(f'Quantile Method\nFill: {quantile_result["fill_fraction"]:.3f}')
    plt.colorbar(im3, ax=axes[2])
    axes[2].axis('off')
    
    # Method 2 (Combined)
    im4 = axes[3].imshow(method2_result['mask'], cmap='binary')
    axes[3].set_title(f'Method 2 ({method2_result["method"]})\n'
                      f'Fill: {method2_result["fill_fraction"]:.3f}')
    plt.colorbar(im4, ax=axes[3])
    axes[3].axis('off')
    
    plt.suptitle(f'Cloud Image Thresholding Comparison\nInput: {Path(image_path).name}')
    plt.tight_layout()
    plt.show()

    # Save the binary masks as PNG files
    output_dir = Path(image_path).parent / "thresholded_outputs"
    output_dir.mkdir(exist_ok=True)
    if yen_result:
        save_lattice_png(yen_result['mask'], output_dir / "yen_threshold_flipped.png")
    save_lattice_png(quantile_result['mask'], output_dir / "quantile_threshold_flipped.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize cloud image thresholding')
    parser.add_argument('--image-path', type=str, help='Path to input image')
    parser.add_argument('--p-target', type=float, default=0.407,
                       help='Target fill fraction (default: 0.407)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                       help='Tolerance for Yen method acceptance (default: 0.05)')
    
    args = parser.parse_args()
    
    visualize_thresholding(args.image_path, args.p_target, args.tolerance)

if __name__ == "__main__":
    main()