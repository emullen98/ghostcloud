#!/bin/bash
#SBATCH --job-name=cp_data_gen_run
#SBATCH --output=/scratch/mbiju2/logs/cp_data_gen_wrapper_%j.out
#SBATCH --error=/scratch/mbiju2/logs/cp_data_gen_wrapper_%j.err
#SBATCH --partition=physics
#SBATCH -t 24:00:00
#SBATCH --ntasks=1

# Parameter definitions
LATTICE_SIZE=20000
MIN_CLOUD_AREA=3000
NUM_SLICES=30
MIN_SLICE_WIDTH=3
FILL_THRESHOLDS=(0.495)
GAMMAS=(0.205)
NUM_LATTICES=500

# Create a timestamp tag
DATE_TAG=$(date +'%Y%m%d_%H%M%S')

# Base output and log directories
OUTPUT_BASE="/scratch/mbiju2/storm/cp_crit_exp_${DATE_TAG}"
LOG_BASE="/scratch/mbiju2/logs/cp_crit_exp_${DATE_TAG}"

mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_BASE"

# Log file for parameter overview
PARAM_LOG="$OUTPUT_BASE/parameters_log.txt"
echo "Lattice Size: $LATTICE_SIZE" >> "$PARAM_LOG"
echo "Minimum Cloud Area: $MIN_CLOUD_AREA" >> "$PARAM_LOG"
echo "Number of Slices: $NUM_SLICES" >> "$PARAM_LOG"
echo "Minimum Slice Width: $MIN_SLICE_WIDTH" >> "$PARAM_LOG"
echo "Fill Thresholds: ${FILL_THRESHOLDS[*]}" >> "$PARAM_LOG"
echo "Gammas: ${GAMMAS[*]}" >> "$PARAM_LOG"

# Loop over GAMMA and Fill thresholds
for GAMMA in "${GAMMAS[@]}"; do
    GAMMA_TAG="g_${GAMMA/./p}"
    GAMMA_DIR="$OUTPUT_BASE/$GAMMA_TAG"
    LOG_GAMMA_DIR="$LOG_BASE/$GAMMA_TAG"
    mkdir -p "$GAMMA_DIR"
    mkdir -p "$LOG_GAMMA_DIR"

    for FILL_THRESH in "${FILL_THRESHOLDS[@]}"; do
        FILL_TAG="p_${FILL_THRESH/./p}"
        FILL_DIR="$GAMMA_DIR/$FILL_TAG"
        LOG_FILL_DIR="$LOG_GAMMA_DIR/$FILL_TAG"
        mkdir -p "$FILL_DIR"
        mkdir -p "$LOG_FILL_DIR"

        # Submit array job with mirrored log structure
        sbatch --array=1-$NUM_LATTICES%100 \
               --cpus-per-task=1 \
               --time=4:00:00 \
               --job-name="cp_${GAMMA_TAG}_${FILL_TAG}" \
               --mem=20G \
               --output="${LOG_FILL_DIR}/job_%A_task_%a.out" \
               --error="${LOG_FILL_DIR}/job_%A_task_%a.err" \
               single_lattice_gen.sbatch \
               "$FILL_DIR" \
               "$LATTICE_SIZE" \
               "$FILL_THRESH" \
               "$GAMMA" \
               "$MIN_CLOUD_AREA" \
               "$NUM_SLICES" \
               "$MIN_SLICE_WIDTH"
    done
done
