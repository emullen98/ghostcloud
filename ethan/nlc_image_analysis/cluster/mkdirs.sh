#!/bin/bash

# Get a list of all PNG files in the directory
files=(*.png)

# Number of subdirectories
num_subdirs=7

# Files per subdirectory
files_per_subdir=100

# Initialize the file counter
file_count=0

# Initialize the directory counter
dir_count=1

# Create directories and move files
for file in "${files[@]}"; do
    # Check if the subdirectory exists, if not, create it
    subdir="subdir_$dir_count"
    if [ ! -d "$subdir" ]; then
        mkdir "$subdir"
    fi

    # Move the file to the subdirectory
    mv "$file" "$subdir/"

    # Increment the file counter
    file_count=$((file_count + 1))

    # Check if the file counter has reached the limit
    if [ "$file_count" -eq "$files_per_subdir" ]; then
        # Reset the file counter
        file_count=0

        # Increment the directory counter
        dir_count=$((dir_count + 1))

        # Adjust the files per subdirectory for the last one
        if [ "$dir_count" -eq "$num_subdirs" ]; then
            files_per_subdir=22
        fi
    fi
done