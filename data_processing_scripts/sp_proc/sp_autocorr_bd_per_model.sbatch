#!/bin/bash
#SBATCH --job-name=sp_autocorr_bd
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/logs/sp_autocorr_bd_%A/sp_autocorr_bd_%A_%a.out
#SBATCH --error=scratch/logs/sp_autocorr_bd_%A/sp_autocorr_bd_%A_%a.err

# ============================================================
# Example:
# sbatch --export=ALL,WIDTH=10000,HEIGHT=10000,P=0.4074,SEED=123,OUTDIR=scratch/sp_autocorr_bd,PREFIX=sp_autocorr_bd clouds/data_processing_scripts/sp_proc/sp_autocorr_bd_per_model.sbatch
# Forwarded env (with defaults):
#   ORDER=FL, CL=4, CF=4, BBOX_PAD=1
# ============================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

: "${WIDTH:?Missing WIDTH}"
: "${HEIGHT:?Missing HEIGHT}"
: "${P:?Missing P}"
: "${SEED:?Missing SEED}"
: "${OUTDIR:?Missing OUTDIR}"
: "${MIN_AREA:=1000}"
: "${MAX_AREA:=7500000}"
: "${PREFIX:=sp_autocorr_bd}"
: "${ROWS_PER_FLUSH:=400}"
: "${MAX_BYTES_PER_FLUSH:=$((128*1024*1024))}"
: "${BD_BIN_WIDTH:=1.0}"

# NEW: preprocessing flags (default to Python script defaults)
: "${ORDER:=FL}"   # LF_bbox|FL
: "${CL:=4}"       # 4|8
: "${CF:=4}"       # 4|8
: "${BBOX_PAD:=1}" # integer

# Optional per-cloud save flags (WK)
: "${SAVE_CR:=1}"
: "${SAVE_CR_BND:=0}"
: "${SAVE_NUMDEN:=1}"

module load cuda/12.8
source clouds/venv/bin/activate

mkdir -p "${OUTDIR}" "scratch/logs/sp_autocorr_bd_${SLURM_JOB_ID:-NA}"

echo "[INFO] Start WIDTH=${WIDTH} HEIGHT=${HEIGHT} P=${P} SEED=${SEED}"
echo "[INFO] MIN_AREA=${MIN_AREA} MAX_AREA=${MAX_AREA}"
echo "[INFO] ORDER=${ORDER} CL=${CL} CF=${CF} BBOX_PAD=${BBOX_PAD}"
echo "[INFO] BD_BIN_WIDTH=${BD_BIN_WIDTH}"
echo "[INFO] SAVE_CR=${SAVE_CR} SAVE_CR_BND=${SAVE_CR_BND} SAVE_NUMDEN=${SAVE_NUMDEN}"

ARGS=(
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --p "${P}"
  --seed "${SEED}"
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

srun -u python -u -m clouds.data_processing_scripts.sp_proc.sp_autocorr_bd_from_model "${ARGS[@]}"

rc=$?
if [[ $rc -eq 0 ]]; then
  echo "[OK] Finished P=${P} SEED=${SEED}"
else
  echo "[ERR] Failed P=${P} SEED=${SEED} rc=${rc}"
  exit $rc
fi
