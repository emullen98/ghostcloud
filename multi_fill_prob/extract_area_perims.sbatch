#!/bin/bash
#SBATCH --job-name=extract_slice
#SBATCH --output=/scratch/mbiju2/logs/analysis/out/%j.out
#SBATCH --error=/scratch/mbiju2/logs/analysis/err/%j.err
#SBATCH --partition=secondary
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --array=0-154%100

# ==== CONFIG ====

FILL_PROBS=(0.4000 0.4025 0.4050 0.4075 0.4100)
NUM_SLICES=30

# ==== MAPPING LOGIC ====

if [ "$SLURM_ARRAY_TASK_ID" -lt 150 ]; then
    FILL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SLICES))
    SLICE_ID=$((SLURM_ARRAY_TASK_ID % NUM_SLICES))
else
    FILL_INDEX=$((SLURM_ARRAY_TASK_ID - 150))
    SLICE_ID=-1
fi

FILL_PROB=${FILL_PROBS[$FILL_INDEX]}
FILL_PROB_DIR="fill_prob_0_${FILL_PROB/./}"

# ==== PATHS ====

BASE_INPUT_DIR="/scratch/mbiju2/storm/multi_fill_prob_20250530_151953"
INPUT_DIR="$BASE_INPUT_DIR/$FILL_PROB_DIR"
OUT_DIR="$BASE_INPUT_DIR/analysis/$FILL_PROB_DIR"
mkdir -p "$OUT_DIR"
mkdir -p /scratch/mbiju2/logs

# ==== OUTPUT FILENAME ====

if [ "$SLICE_ID" -eq -1 ]; then
    OUT_FILE="$OUT_DIR/full_cloud_log_data.csv"
else
    OUT_FILE="$OUT_DIR/cloud_slice_$(printf "%02d" $SLICE_ID)_log_data.csv"
fi

# ==== PYTHON SETUP ====

module load python
alias python=python3
source /u/mbiju2/clouds/clean_scripts/venv/bin/activate

# ==== RUN ====

python extract_slice_data.py "$INPUT_DIR" "$SLICE_ID" "$OUT_FILE"
