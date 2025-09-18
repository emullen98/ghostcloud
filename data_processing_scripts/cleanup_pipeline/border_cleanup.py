#!/usr/bin/env python3

import numpy as np
from PIL import Image
import argparse
import sys

def find_content_bounds(img_array):
    """Find the bounds of the non-white content in the image."""
    # Convert to boolean where True is non-white pixels
    mask = ~(img_array == 255).all(axis=2)
    
    # Find the first and last non-white rows/columns
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin, rmax, cmin, cmax

def clean_image(input_path, output_path):
    """Remove white strips from image edges and save cleaned version."""
    try:
        # Open and convert image to numpy array
        img = Image.open(input_path)
        img_array = np.array(img)
        
        # Get original dimensions
        orig_height, orig_width = img_array.shape[:2]
        
        # Find bounds of content
        rmin, rmax, cmin, cmax = find_content_bounds(img_array)
        
        # Crop the image array
        cleaned_array = img_array[rmin:rmax+1, cmin:cmax+1]
        
        # Convert back to PIL Image and save
        cleaned_img = Image.fromarray(cleaned_array)
        cleaned_img.save(output_path)
        
        # Print statistics
        print(f"Rows removed from top: {rmin}")
        print(f"Rows removed from bottom: {orig_height - (rmax + 1)}")
        print(f"Columns removed from left: {cmin}")
        print(f"Columns removed from right: {orig_width - (cmax + 1)}")
        print(f"New image size: {cleaned_array.shape[1]}x{cleaned_array.shape[0]} pixels")
        
        return True
        
    except Exception as e:
        print(f"Error processing image: {str(e)}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Clean white strips from image edges')
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument('output_path', help='Path to save cleaned image')
    
    args = parser.parse_args()
    
    success = clean_image(args.input_path, args.output_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()