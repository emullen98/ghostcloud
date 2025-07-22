#!/bin/bash
#SBATCH --job-name=extract_cp_slices
#SBATCH --output=/scratch/mbiju2/logs/analysis/extract_%A_%a.out
#SBATCH --error=/scratch/mbiju2/logs/analysis/extract_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --array=0-991%100  # 4 gamma × 8 p × 31 slices = 992 jobs

# ==== CONFIG ====
GAMMAS=(g_0p1975 g_0p200 g_0p2025 g_0p205)
PS=(p_0p460 p_0p465 p_0p470 p_0p475 p_0p480 p_0p485 p_0p490 p_0p495)
NUM_SLICES=31  # Slices 0–29 and -1 (full cloud)

TOTAL_PS_SLICE=$(( ${#PS[@]} * NUM_SLICES ))

GAMMA_INDEX=$(( SLURM_ARRAY_TASK_ID / TOTAL_PS_SLICE ))
PS_SLICE_REMAINDER=$(( SLURM_ARRAY_TASK_ID % TOTAL_PS_SLICE ))

P_INDEX=$(( PS_SLICE_REMAINDER / NUM_SLICES ))
SLICE_ID_TMP=$(( PS_SLICE_REMAINDER % NUM_SLICES ))

if [ "$SLICE_ID_TMP" -eq 30 ]; then
    SLICE_ID=-1
else
    SLICE_ID=$SLICE_ID_TMP
fi

GAMMA_DIR=${GAMMAS[$GAMMA_INDEX]}
P_DIR=${PS[$P_INDEX]}

BASE_ANALYSIS_DIR="/scratch/mbiju2/storm/cp_crit_exp_aggregated_20250711/analysis/$GAMMA_DIR/$P_DIR"
mkdir -p "$BASE_ANALYSIS_DIR"

if [ "$SLICE_ID" -eq -1 ]; then
    OUT_FILE="$BASE_ANALYSIS_DIR/full_cloud_log_data.csv"
else
    OUT_FILE="$BASE_ANALYSIS_DIR/cloud_slice_$(printf "%02d" $SLICE_ID)_filtered_log_data.csv"
fi

echo "[INFO] Running gamma=$GAMMA_DIR, p=$P_DIR, slice_id=$SLICE_ID"
python extract_slice_data.py "$GAMMA_DIR" "$P_DIR" "$SLICE_ID" "$OUT_FILE"
