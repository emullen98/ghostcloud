#!/bin/bash
#SBATCH --job-name=extract_cp_slices
#SBATCH --output=/scratch/mbiju2/logs/analysis/extract_%A_%a.out
#SBATCH --error=/scratch/mbiju2/logs/analysis/extract_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=20G
# (num_slices + 1) * num_p * num_gamma = 21 * 7 * 7 = 1029
#SBATCH --array=0-441%100 

# Fail fast & show unset-var bugs
set -euo pipefail

# Always start in the directory where you ran `sbatch`
cd "$SLURM_SUBMIT_DIR"

# Point ROOT_DIR to your repo root relative to where you submit from.
# If you run `sbatch` inside the analysis/ folder:
ROOT_DIR="$(cd .. && pwd)"

# If you run `sbatch` from the repo root, instead use:
# ROOT_DIR="$(pwd)"

# Source shared params, build tags, compute slice count
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "../pipeline_params.sh"
build_tag_arrays   # sets GAMMA_TAGS[], P_TAGS[], NUM_SLICES_TOTAL

GAMMAS=("${GAMMA_TAGS[@]}")   
PS=("${P_TAGS[@]}")         
NUM_SLICES_TOTAL=${NUM_SLICES_TOTAL}  

TOTAL_PS_SLICE=$(( ${#PS[@]} * NUM_SLICES_TOTAL ))

GAMMA_INDEX=$(( SLURM_ARRAY_TASK_ID / TOTAL_PS_SLICE ))
PS_SLICE_REMAINDER=$(( SLURM_ARRAY_TASK_ID % TOTAL_PS_SLICE ))

P_INDEX=$(( PS_SLICE_REMAINDER / NUM_SLICES_TOTAL ))
SLICE_IDX=$(( PS_SLICE_REMAINDER % NUM_SLICES_TOTAL ))

# Map last index to -1 if INCLUDE_FULL_CLOUD=1
if [[ "${INCLUDE_FULL_CLOUD}" -eq 1 && "$SLICE_IDX" -eq $((NUM_SLICES_TOTAL-1)) ]]; then
  SLICE_ID=-1
else
  SLICE_ID=$SLICE_IDX
fi

GAMMA_DIR=${GAMMAS[$GAMMA_INDEX]}
P_DIR=${PS[$P_INDEX]}

BASE_ANALYSIS_DIR="${BASE_RUN_DIR}/analysis/${GAMMA_DIR}/${P_DIR}"  
mkdir -p "$BASE_ANALYSIS_DIR"

if [ "$SLICE_ID" -eq -1 ]; then
  OUT_FILE="$BASE_ANALYSIS_DIR/full_cloud_log_data.csv"
else
  OUT_FILE="$BASE_ANALYSIS_DIR/cloud_slice_$(printf "%02d" $SLICE_ID)_filtered_log_data.csv"
fi

echo "[INFO] Extract: base=${BASE_RUN_DIR} gamma=${GAMMA_DIR} p=${P_DIR} slice=${SLICE_ID}"
python extract_slice_data.py "$GAMMA_DIR" "$P_DIR" "$SLICE_ID" "$OUT_FILE" --base_dir "${BASE_RUN_DIR}"
