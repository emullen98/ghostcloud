#!/bin/bash
#SBATCH --job-name=thr_wk_image
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/logs/thr_wk_%A/thr_wk_%A_%a.out
#SBATCH --error=scratch/logs/thr_wk_%A/thr_wk_%A_%a.err

# ============================================================
# Example submit (done automatically by wrapper):
# sbatch --export=ALL,IMG=path/to/foo.png,OUTDIR=scratch/threshold_wk,THRESHOLDS=0.25,0.3,0.35,0.4,0.45,0.5,0.55,0,6,0.65,MIN_AREA=500,MAX_AREA=7500000,PREFIX=thr_wk clouds/exps/threshold_wk_per_image.sbatch
# ============================================================

set -euo pipefail

# --- Project root
PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- Config passed in from wrapper
: "${IMG:?Missing IMG}"
: "${OUTDIR:?Missing OUTDIR}"
: "${THRESHOLDS_FILE:?Missing THRESHOLDS_FILE}"
: "${MIN_AREA:=500}"
: "${MAX_AREA:=7500000}"
: "${PREFIX:=thr_wk}"
: "${ROWS_PER_FLUSH:=400}"
: "${MAX_BYTES_PER_FLUSH:=$((128*1024*1024))}"

# --- Environment
module load cuda/12.8
source clouds/venv/bin/activate

mkdir -p "${OUTDIR}" scratch/logs

STEM="$(basename "${IMG%.*}")"
echo "[INFO] Starting threshold WK job for ${STEM}"
echo "[INFO] IMG=${IMG}"
echo "[INFO] OUTDIR=${OUTDIR}"
echo "[INFO] THRESHOLDS_FILE=${THRESHOLDS_FILE}"
echo "[INFO] MIN_AREA=${MIN_AREA}, MAX_AREA=${MAX_AREA}"

# --- Run Python worker
ARGS=( --image "${IMG}"
       --thresholds-file "${THRESHOLDS_FILE}"
       --min-area "${MIN_AREA}"
       --max-area "${MAX_AREA}"
       --outdir "${OUTDIR}"
       --prefix "${PREFIX}"
       --rows-per-flush "${ROWS_PER_FLUSH}"
       --max-bytes-per-flush "${MAX_BYTES_PER_FLUSH}" )

srun -u python -u -m clouds.data_processing_scripts.image_proc.threshold_wk_from_image "${ARGS[@]}"

rc=$?
if [[ $rc -eq 0 ]]; then
  echo "[OK] Finished ${STEM}"
else
  echo "[ERR] Failed ${STEM} rc=${rc}"
  exit $rc
fi
