#!/bin/bash
#SBATCH --job-name=compute_d
#SBATCH --output=/scratch/mbiju2/logs/analysis/compute_d_%A_%a.out
#SBATCH --error=/scratch/mbiju2/logs/analysis/compute_d_%A_%a.err
#SBATCH --partition=physics
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --array=0-154%50  # 5 fill probs × 31 slices (0–30 where 30 = full cloud)

# ==== CONFIG ====
FILL_PROBS=(0.4066 0.4069 0.4072 0.4074 0.4076)
NUM_SLICES=30

if [ "$SLURM_ARRAY_TASK_ID" -lt 150 ]; then
    FILL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SLICES))
    SLICE_ID=$((SLURM_ARRAY_TASK_ID % NUM_SLICES))
else
    FILL_INDEX=$((SLURM_ARRAY_TASK_ID - 150))
    SLICE_ID=-1
fi

FILL_PROB=${FILL_PROBS[$FILL_INDEX]}
FILL_PROB_DIR="fill_prob_${FILL_PROB/./_}"
ANALYSIS_DIR="/scratch/mbiju2/storm/multi_fill_prob_20250601_124857/analysis/$FILL_PROB_DIR"

python estimate_d_vals.py "$ANALYSIS_DIR" "$SLICE_ID"
