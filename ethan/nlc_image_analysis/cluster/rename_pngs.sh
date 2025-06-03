#!/bin/bash

# This script renames every .png file in the current directory such that it ends with id=[integer].

png_directory="/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images/xsc1/good/"

counter=1  # Initialize a counter for each file's ID number

for file in "$png_directory"*.png; do
  base_name="${file%.png}"  # Extract the base name
  new_name="${png_directory}$(basename "$base_name")_id=${counter}.png"
  mv "$file" "$new_name"
  ((counter++))
done

echo "PNG files have been renamed successfully"
