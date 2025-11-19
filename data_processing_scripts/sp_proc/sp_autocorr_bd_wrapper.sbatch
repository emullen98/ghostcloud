#!/bin/bash
#SBATCH --job-name=sp_autocorr_bd_wrap
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/logs/sp_autocorr_bd_wrap_%j.out
#SBATCH --error=scratch/logs/sp_autocorr_bd_wrap_%j.err

# ============================================================
# Example submit:
# sbatch --export=ALL,WIDTH=10000,HEIGHT=10000,P=0.4074,SEEDS_FILE=configs/seeds.txt,OUTDIR=scratch/sp_autocorr_bd,PREFIX=sp_autocorr_bd clouds/data_processing_scripts/sp_proc/sp_autocorr_bd_wrapper.sbatch
# sbatch --export=ALL,WIDTH=4000,HEIGHT=2666,P=0.407254,SEEDS_FILE=clouds/data_processing_scripts/sp_proc/seeds_42_285.txt,OUTDIR=scratch/new_data/sp_autocorr_bd/p0p407254_cl8_cf8_ordFL,PREFIX=sp_autocorr_bd,CL=8,CF=8,ORDER=FL,BBOX_PAD=1 clouds/data_processing_scripts/sp_proc/sp_autocorr_bd_wrapper.sbatch
# Notes:
#   - SEEDS_FILE: one integer seed per line (blanks and '#' allowed)
#   - P: single p value used for all submitted seeds
#   - ORDER/CL/CF/BBOX_PAD: forwarded to inner sbatch (defaults: FL,4,4,1)
# ============================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

: "${WIDTH:?Need WIDTH}"
: "${HEIGHT:?Need HEIGHT}"
: "${P:?Need P}"
: "${SEEDS_FILE:?Need SEEDS_FILE}"
: "${OUTDIR:=scratch/sp_autocorr_bd}"
: "${MIN_AREA:=100}"
: "${MAX_AREA:=7500000}"
: "${PREFIX:=sp_autocorr_bd}"
: "${ROWS_PER_FLUSH:=400}"
: "${MAX_BYTES_PER_FLUSH:=$((128*1024*1024))}"
: "${BD_BIN_WIDTH:=1.0}"
: "${INNER_SBATCH:=clouds/data_processing_scripts/sp_proc/sp_autocorr_bd_per_model.sbatch}"

# NEW: preprocessing flags (forwarded to inner job)
: "${ORDER:=FL}"   # LF_bbox|FL
: "${CL:=4}"       # 4|8 (label connectivity)
: "${CF:=4}"       # 4|8 (hole-fill connectivity)
: "${BBOX_PAD:=1}" # integer

# Optional WK persistence flags
: "${SAVE_CR:=1}"
: "${SAVE_CR_BND:=0}"
: "${SAVE_NUMDEN:=1}"

mkdir -p "$OUTDIR" scratch/logs

if [[ ! -f "$SEEDS_FILE" ]]; then
  echo "[ERR] SEEDS_FILE not found: $SEEDS_FILE" >&2
  exit 1
fi

# Read seeds (ignore blanks and comments)
mapfile -t SEEDS < <(grep -v '^\s*#' "$SEEDS_FILE" | awk 'NF' | tr -d '\r')
if (( ${#SEEDS[@]} == 0 )); then
  echo "[ERR] No seeds found in $SEEDS_FILE" >&2
  exit 1
fi

echo "[INFO] Submitting ${#SEEDS[@]} job(s) over seeds."
echo "[INFO] WIDTH=${WIDTH} HEIGHT=${HEIGHT} P=${P}"
echo "[INFO] OUTDIR=${OUTDIR}"
echo "[INFO] MIN_AREA=${MIN_AREA} MAX_AREA=${MAX_AREA}"
echo "[INFO] ORDER=${ORDER} CL=${CL} CF=${CF} BBOX_PAD=${BBOX_PAD}"
echo "[INFO] BD_BIN_WIDTH=${BD_BIN_WIDTH}"
echo "[INFO] SAVE_CR=${SAVE_CR} SAVE_CR_BND=${SAVE_CR_BND} SAVE_NUMDEN=${SAVE_NUMDEN}"

for SEED in "${SEEDS[@]}"; do
  sbatch --export=ALL,\
WIDTH="$WIDTH",\
HEIGHT="$HEIGHT",\
P="$P",\
SEED="$SEED",\
OUTDIR="$OUTDIR",\
MIN_AREA="$MIN_AREA",\
MAX_AREA="$MAX_AREA",\
PREFIX="$PREFIX",\
ROWS_PER_FLUSH="$ROWS_PER_FLUSH",\
MAX_BYTES_PER_FLUSH="$MAX_BYTES_PER_FLUSH",\
BD_BIN_WIDTH="$BD_BIN_WIDTH",\
ORDER="$ORDER",\
CL="$CL",\
CF="$CF",\
BBOX_PAD="$BBOX_PAD",\
SAVE_CR="$SAVE_CR",\
SAVE_CR_BND="$SAVE_CR_BND",\
SAVE_NUMDEN="$SAVE_NUMDEN" \
"$INNER_SBATCH"
done
