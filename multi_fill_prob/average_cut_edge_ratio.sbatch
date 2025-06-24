#!/bin/bash
#SBATCH --job-name=avg_cut_edge
#SBATCH --output=/scratch/mbiju2/logs/analysis/avg_cut_edge_%A_%a.out
#SBATCH --error=/scratch/mbiju2/logs/analysis/avg_cut_edge_%A_%a.err
#SBATCH --partition=physics
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --array=0-149%50  # 5 fill probs × 30 slices (0–29)

# ==== CONFIG ====
FILL_PROBS=(0.4066 0.4069 0.4072 0.4074 0.4076)
NUM_SLICES=30

FILL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SLICES))
SLICE_ID=$((SLURM_ARRAY_TASK_ID % NUM_SLICES))

FILL_PROB=${FILL_PROBS[$FILL_INDEX]}
FILL_PROB_DIR="fill_prob_${FILL_PROB/./_}"
ANALYSIS_DIR="/scratch/mbiju2/storm/multi_fill_prob_20250601_124857/analysis/$FILL_PROB_DIR"

source /u/mbiju2/clouds/venv/bin/activate
python /u/mbiju2/clouds/multi_fill_prob/average_cut_edge_ratio.py "$ANALYSIS_DIR" "$SLICE_ID"