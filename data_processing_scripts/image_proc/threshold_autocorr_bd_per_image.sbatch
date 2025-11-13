#!/bin/bash
#SBATCH --job-name=thr_autocorr_bd
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/logs/thr_autocorr_bd_%A/thr_autocorr_bd_%A_%a.out
#SBATCH --error=scratch/logs/thr_autocorr_bd_%A/thr_autocorr_bd_%A_%a.err

# ============================================================
# Example submit:
# sbatch --export=ALL,IMG=path/to/foo.png,OUTDIR=scratch/threshold_autocorr_bd,THRESHOLDS_FILE=thresholds.txt,MIN_AREA=500,MAX_AREA=7500000,PREFIX=thr_autocorr_bd clouds/exps/threshold_autocorr_bd_per_image.sbatch
# ============================================================

set -euo pipefail

# --- Project root
PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- Config (passed in by wrapper or manually)
: "${IMG:?Missing IMG}"
: "${OUTDIR:?Missing OUTDIR}"
: "${THRESHOLDS_FILE:?Missing THRESHOLDS_FILE}"
: "${MIN_AREA:=500}"
: "${MAX_AREA:=7500000}"
: "${PREFIX:=thr_autocorr_bd}"
: "${ROWS_PER_FLUSH:=400}"
: "${MAX_BYTES_PER_FLUSH:=$((128*1024*1024))}"
: "${BD_BIN_WIDTH:=1.0}"

# NEW: preprocessing flags (default to Python script defaults)
: "${ORDER:=FL}"   # LF_bbox|FL
: "${CL:=4}"       # 4|8
: "${CF:=4}"       # 4|8
: "${BBOX_PAD:=1}" # integer

# Optional per-cloud save flags
: "${SAVE_CR:=0}"
: "${SAVE_CR_BND:=0}"
: "${SAVE_NUMDEN:=1}"

# --- Environment
module load cuda/12.8
source clouds/venv/bin/activate

mkdir -p "${OUTDIR}" scratch/logs/thr_autocorr_bd_${SLURM_JOB_ID:-NA}

STEM="$(basename "${IMG%.*}")"
echo "[INFO] Starting threshold+autocorr+BD job for ${STEM}"
echo "[INFO] IMG=${IMG}"
echo "[INFO] OUTDIR=${OUTDIR}"
echo "[INFO] THRESHOLDS_FILE=${THRESHOLDS_FILE}"
echo "[INFO] MIN_AREA=${MIN_AREA}, MAX_AREA=${MAX_AREA}"
echo "[INFO] ORDER=${ORDER} CL=${CL} CF=${CF} BBOX_PAD=${BBOX_PAD}"
echo "[INFO] BD_BIN_WIDTH=${BD_BIN_WIDTH}"
echo "[INFO] SAVE_CR=${SAVE_CR}, SAVE_CR_BND=${SAVE_CR_BND}, SAVE_NUMDEN=${SAVE_NUMDEN}"

# --- Build argument list
ARGS=(
  --image "${IMG}"
  --thresholds-file "${THRESHOLDS_FILE}"
  --min-area "${MIN_AREA}"
  --max-area "${MAX_AREA}"
  --outdir "${OUTDIR}"
  --prefix "${PREFIX}"
  --rows-per-flush "${ROWS_PER_FLUSH}"
  --max-bytes-per-flush "${MAX_BYTES_PER_FLUSH}"
  --bd-bin-width "${BD_BIN_WIDTH}"
  --order "${ORDER}"
  --cl "${CL}"
  --cf "${CF}"
  --bbox-pad "${BBOX_PAD}"
)

[[ "${SAVE_CR}" -eq 1 ]] && ARGS+=( --save-cr )
[[ "${SAVE_CR_BND}" -eq 1 ]] && ARGS+=( --save-cr-bnd )
[[ "${SAVE_NUMDEN}" -eq 1 ]] && ARGS+=( --save-numden )

# --- Run Python worker
srun -u python -u -m clouds.data_processing_scripts.image_proc.threshold_autocorr_bd_from_image "${ARGS[@]}"

rc=$?
if [[ $rc -eq 0 ]]; then
  echo "[OK] Finished ${STEM}"
else
  echo "[ERR] Failed ${STEM} rc=${rc}"
  exit $rc
fi
