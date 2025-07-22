#!/bin/bash
#
#SBATCH --cpus-per-task=1
#SBATCH -t 4:00:00
#SBATCH -p secondary
#SBATCH --mail-user=mbiju2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = array job ID, %a = task ID

module load anaconda3/2024.10

FILL_DIR="$1"
LATTICE_SIZE="$2"
FILL_THRESH="$3"
GAMMA="$4"
MIN_CLOUD_AREA="$5"
NUM_SLICES="$6"
MIN_SLICE_WIDTH="$7"

LATTICE_DIR="$FILL_DIR/run_$SLURM_ARRAY_TASK_ID"
mkdir -p "$LATTICE_DIR"

python3 single_lattice_script.py "$LATTICE_DIR" "$LATTICE_SIZE" "$FILL_THRESH" "$GAMMA" "$MIN_CLOUD_AREA" "$NUM_SLICES" "$MIN_SLICE_WIDTH"
