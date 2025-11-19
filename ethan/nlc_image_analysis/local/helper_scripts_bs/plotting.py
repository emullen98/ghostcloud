"""
Created Nov 18 2024
Updated Nov 18 2024
"""
import matplotlib.colors as mcolors
import numpy as np


def truncate_colormap(colormap, minval=0.2, maxval=0.8, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'trunc({colormap.save_dir},{minval:.2f},{maxval:.2f})', colormap(np.linspace(minval, maxval, n)))
    return new_cmap
