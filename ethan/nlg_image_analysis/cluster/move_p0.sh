#!/bin/bash

# Create the p0 directory if it doesn't exist
mkdir -p p0

# Loop through all files in the current directory
for file in *p*; do
  # Check if the item is a file and not a directory
  if [[ -f $file ]]; then
    # Check if the file contains "p0"
    if [[ $file == *"p0"* ]]; then
      # Move the file to the p0 directory
      mv "$file" p0/
    fi
  fi
done
