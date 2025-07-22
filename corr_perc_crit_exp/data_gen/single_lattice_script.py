import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../utils')))
from cloud_utils import *
import gzip
from pathlib import Path
import csv

OUT_DIR_PATH     = sys.argv[1]        
LATTICE_SIZE     = int(sys.argv[2])
FILL_THRESH      = float(sys.argv[3])
GAMMA            = float(sys.argv[4])
MIN_CLOUD_AREA   = int(sys.argv[5])
NUM_SLICES       = int(sys.argv[6])
MIN_SLICE_WIDTH  = int(sys.argv[7])

# Generate lattice and save
raw_lattice = generate_correlated_percolation_lattice_optimized(LATTICE_SIZE, LATTICE_SIZE, GAMMA, FILL_THRESH)
save_lattice_npy(raw_lattice, OUT_DIR_PATH, "raw_lattice")

flood_filled_lattice, _ = flood_fill_and_label_features(raw_lattice)
save_lattice_npy(flood_filled_lattice, OUT_DIR_PATH, "flood_filled_lattice")

# Extract clouds
cropped_clouds = extract_cropped_clouds_by_size(flood_filled_lattice, MIN_CLOUD_AREA)

cloud_data = []
for cloud in cropped_clouds:
    segment_list = slice_cloud_into_segments(cloud, NUM_SLICES, MIN_SLICE_WIDTH)
    if segment_list:
        cloud_data.append(compute_mirrored_slice_geometry(segment_list))

flattened_data = flatten_cloud_metadata_for_csv(cloud_data)

filename = Path(OUT_DIR_PATH) / "slice_data.csv.gz"
with gzip.open(filename, "wt", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
    writer.writeheader()
    writer.writerows(flattened_data)
