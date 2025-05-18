# -*- coding: utf-8 -*-
"""
Created on Oct 02 2023
Updated on Feb 14 2024
@author: sm38
"""

import glob
import contextlib
from PIL import Image
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
import glob
from matplotlib import rc, rcParams

# Give the locations where the individual frame images are located as well as the destination for the gif file
# The asterisk is a placeholder such that every glob.glob will grab every file with name frame_[something].png
fp_in = f"/Users/emullen98/Desktop/DirectedPercolation/plots/Movie/frame*.png"
fp_out = f"/Users/emullen98/Desktop/DirectedPercolation/plots/Movie/movie4.gif"

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(fp_in), key=os.path.getmtime))

    # extract first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
