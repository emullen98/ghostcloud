from cloud_utils import *
import gzip
from pathlib import Path
import csv
import sys

OUT_DIR_PATH     = (sys.argv[1])        # Convert to Path object
LATTICE_SIZE     = int(sys.argv[2])
FILL_PROB        = float(sys.argv[3])
MIN_CLOUD_AREA   = int(sys.argv[4])
NUM_SLICES       = int(sys.argv[5])
MIN_SLICE_WIDTH  = int(sys.argv[6])

# Generate single lattice, and save the array, filtered over the same criteria as downsteam
raw_lattice = generate_site_percolation_lattice(LATTICE_SIZE, LATTICE_SIZE, FILL_PROB)
filename = "raw_lattice"
save_lattice_npy(raw_lattice, OUT_DIR_PATH, filename)

flood_filled_lattice, _ = flood_fill_and_label_features(raw_lattice)
filename = "flood_filled_lattice"
save_lattice_npy(flood_filled_lattice, OUT_DIR_PATH, filename)

# Get list of single clouds, 
cropped_clouds = extract_cropped_clouds_by_size(flood_filled_lattice, MIN_CLOUD_AREA)

cloud_data = []
# segment them, 
for cloud in cropped_clouds:
    segment_list = slice_cloud_into_segments(cloud, NUM_SLICES, MIN_SLICE_WIDTH)
    # slice if we get a valid segmentation
    if segment_list:
        cloud_data.append(compute_mirrored_slice_geometry(segment_list))

flattened_data = flatten_cloud_metadata_for_csv(cloud_data)

# then save list of dicts for all clouds in lattice given they have valid slices to a single csv file. 
filename = OUT_DIR_PATH + "slice_data.csv.gz"
filename = Path(filename)

with gzip.open(filename, "wt", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
    writer.writeheader()
    writer.writerows(flattened_data)
