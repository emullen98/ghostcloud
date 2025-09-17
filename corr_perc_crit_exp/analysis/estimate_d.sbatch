#!/bin/bash
#SBATCH --job-name=estimate_cp_d
#SBATCH --output=/scratch/mbiju2/logs/analysis/estimate_d_%A_%a.out
#SBATCH --error=/scratch/mbiju2/logs/analysis/estimate_d_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=4G
# (num_slices + 1) * num_p * num_gamma = 21 * 7 * 7 = 1029
#SBATCH --array=0-588%100 

# Fail fast & show unset-var bugs
set -euo pipefail

# Always start in the directory where you ran `sbatch`
cd "$SLURM_SUBMIT_DIR"

# Point ROOT_DIR to your repo root relative to where you submit from.
# If you run `sbatch` inside the analysis/ folder:
ROOT_DIR="$(cd .. && pwd)"

# If you run `sbatch` from the repo root, instead use:
# ROOT_DIR="$(pwd)"

# Activate venv (unchanged)
source /u/mbiju2/clouds/venv/bin/activate

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

if [[ "${INCLUDE_FULL_CLOUD}" -eq 1 && "$SLICE_IDX" -eq $((NUM_SLICES_TOTAL-1)) ]]; then
  SLICE_ID=-1
else
  SLICE_ID=$SLICE_IDX
fi

GAMMA_DIR=${GAMMAS[$GAMMA_INDEX]}
P_DIR=${PS[$P_INDEX]}

echo "[INFO] Estimate D: base=${BASE_RUN_DIR} gamma=${GAMMA_DIR} p=${P_DIR} slice=${SLICE_ID}"
python estimate_d.py "$GAMMA_DIR" "$P_DIR" "$SLICE_ID" --base_dir "${BASE_RUN_DIR}"
