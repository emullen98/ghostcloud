#!/bin/bash
#SBATCH --job-name=extract_cut_edge
#SBATCH --output=/scratch/mbiju2/logs/analysis/out/cut_edge/%j.out
#SBATCH --error=/scratch/mbiju2/logs/analysis/err/cut_edge/%j.err
#SBATCH --partition=secondary
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --array=0-149%100

# ==== CONFIG ====

FILL_PROBS=(0.4066 0.4069 0.4072 0.4074 0.4076)
NUM_SLICES=30

# ==== MAPPING LOGIC ====

FILL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SLICES))
SLICE_ID=$((SLURM_ARRAY_TASK_ID % NUM_SLICES))

FILL_PROB=${FILL_PROBS[$FILL_INDEX]}
FILL_PROB_DIR="fill_prob_${FILL_PROB/./_}"

# ==== PATHS ====

BASE_INPUT_DIR="/scratch/mbiju2/storm/multi_fill_prob_20250601_124857"
INPUT_DIR="$BASE_INPUT_DIR/$FILL_PROB_DIR"
OUT_DIR="$BASE_INPUT_DIR/analysis/$FILL_PROB_DIR"
mkdir -p "$OUT_DIR"
mkdir -p /scratch/mbiju2/logs

# ==== OUTPUT FILENAME ====

OUT_FILE="$OUT_DIR/cloud_slice_$(printf "%02d" $SLICE_ID)_cut_edge_data.csv"

# ==== RUN ====

source /u/mbiju2/clouds/venv/bin/activate
python extract_cut_edge_perims.py "$INPUT_DIR" "$SLICE_ID" "$OUT_FILE"
