#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all PNG files in input directory
files=($(find "$INPUT_DIR" -type f -name "*.png"))
total_files=${#files[@]}
processed=0
failed=0

echo "Found $total_files PNG files to process"

# Process each file
for file in "${files[@]}"; do
    # Get the filename without path
    filename=$(basename "$file")
    output_path="$OUTPUT_DIR/$filename"
    
    echo "Processing: $filename"
    
    # Call the Python script
    if python3 data_processing_scripts/cleanup_pipeline/border_cleanup.py "$file" "$output_path"; then
        ((processed++))
    else
        ((failed++))
        echo "Failed to process: $filename"
    fi
done

echo "Processing complete!"
echo "Successfully processed: $processed files"
echo "Failed to process: $failed files"
echo "Total files: $total_files"