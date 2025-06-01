#!/bin/bash
#
#SBATCH --cpus-per-task=1
#SBATCH -t 4:00:00
#SBATCH -J fill_prob_data_gen_%A
#SBATCH -p secondary
#SBATCH --mail-user=mbiju2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --error=/scratch/mbiju2/logs/err/job_%A/task_%a.err
#SBATCH --output=/scratch/mbiju2/logs/out/job_%A/task_%a.out
#
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10

FILL_DIR="$1"
LATTICE_SIZE="$2"
FILL_PROB="$3"
MIN_CLOUD_AREA="$4"
NUM_SLICES="$5"
MIN_SLICE_WIDTH="$6"

LATTICE_DIR="$FILL_DIR/lattice_run_$SLURM_ARRAY_TASK_ID"
mkdir -p "$LATTICE_DIR"
python3 data_gen.py "$LATTICE_DIR" "$LATTICE_SIZE" "$FILL_PROB" "$MIN_CLOUD_AREA" "$NUM_SLICES" "$MIN_SLICE_WIDTH"