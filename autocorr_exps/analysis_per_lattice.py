#!/usr/bin/env python3

import argparse
import json
import numpy as np
from pathlib import Path
import glob
from typing import List, Dict, Tuple
from utils.autocorr_utils import fit_exponential_decay, save_fit_results, ExponentialFit

def load_correlation_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw correlation data from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return np.array(data["N_r"]), np.array(data["D_r"])

def process_correlation_file(filepath: Path) -> ExponentialFit:
    """Process a single correlation data file."""
    N_r, D_r = load_correlation_data(filepath)
    
    # Compute C(r)
    mask = D_r > 0
    C_r = np.zeros_like(N_r, dtype=float)
    C_r[mask] = N_r[mask] / D_r[mask]
    r_values = np.arange(len(N_r))
    
    # Fit exponential
    return fit_exponential_decay(r_values, C_r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with raw correlation data")
    parser.add_argument("--output-dir", required=True, help="Directory for fit results")
    parser.add_argument("--array-task", type=int, required=True, help="Current array task ID")
    parser.add_argument("--array-size", type=int, required=True, help="Total number of array tasks")
    parser.add_argument("--prefix", default="wk", help="File prefix")
    args = parser.parse_args()
    
    # Get all correlation files
    files = sorted(glob.glob(f"{args.input_dir}/{args.prefix}_*_corr_raw.json"))
    
    # Split files among array tasks
    task_files = np.array_split(files, args.array_size)[args.array_task]
    
    # Process files assigned to this task
    fits = []
    for filepath in task_files:
        fit = process_correlation_file(Path(filepath))
        fits.append(fit)
    
    # Save results
    output_path = Path(args.output_dir) / f"fits_task_{args.array_task}.json"
    save_fit_results(fits, str(output_path))
    print(f"[OK] Processed {len(fits)} files, wrote results to {output_path}")

if __name__ == "__main__":
    main()