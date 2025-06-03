#!/bin/bash
#SBATCH --job-name=data_gen_run
#SBATCH --output=/scratch/mbiju2/logs/data_gen_wrapper%j.out
#SBATCH --error=/scratch/mbiju2/logs/data_gen_wrapper%j.err
#SBATCH --partition=physics
#SBATCH -t 24:00:00
#SBATCH --ntasks=1

# Parameter Definitions
LATTICE_SIZE=20000
MIN_CLOUD_AREA=3000
NUM_SLICES=30
MIN_SLICE_WIDTH=3
FILL_PROBABILITIES=(0.4000 0.4025 0.4050 0.4075 0.4100)
NUM_LATTICES=500

# Create a unique output directory based on the current date and time
OUTPUT_DIR="/scratch/mbiju2/storm/multi_fill_prob_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

# Log file for parameters
PARAM_LOG="$OUTPUT_DIR/parameters_log.txt"
echo "Lattice Size: $LATTICE_SIZE" >> "$PARAM_LOG"
echo "Minimum Cloud Area: $MIN_CLOUD_AREA" >> "$PARAM_LOG"
echo "Number of Slices: $NUM_SLICES" >> "$PARAM_LOG"
echo "Minimum Slice Width: $MIN_SLICE_WIDTH" >> "$PARAM_LOG"
echo "Fill Probabilities: ${FILL_PROBABILITIES[*]}" >> "$PARAM_LOG"

# Loop Through Fill Probabilities
for FILL_PROB in "${FILL_PROBABILITIES[@]}"; do
    # Create a subdirectory for each fill probability
    FILL_DIR="$OUTPUT_DIR/fill_prob_${FILL_PROB//./_}"
    mkdir -p "$FILL_DIR"
    # Run the data_gen.py script 1000 times
    sbatch  --array=1-$NUM_LATTICES%100 \
            --cpus-per-task=1 \
            --time=4:00:00 \
            --job-name="fill_prob_data_gen_${FILL_PROB//./_}" \
            --mem=20G \
            data_gen.sbatch \
            $FILL_DIR \
            $LATTICE_SIZE \
            $FILL_PROB \
            $MIN_CLOUD_AREA \
            $NUM_SLICES \
            $MIN_SLICE_WIDTH 
             
done