#!/bin/bash
#SBATCH --job-name=thr_wk_wrap
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/logs/thr_wk_wrap_%j.out
#SBATCH --error=scratch/logs/thr_wk_wrap_%j.err

# ============================================================
# Example submit command:
# sbatch --export=ALL,INDIR=clouds/cloud_data/cleaned,OUTDIR=scratch/threshold_wk,THRESHOLDS_FILE=clouds/data_processing_scripts/image_proc/thresholds.txt,PREFIX=thr_wk,SAVE_CR=1,SAVE_CR_BND=1,SAVE_NUMDEN=1 clouds/data_processing_scripts/image_proc/threshold_wk_wrapper.sbatch
# Notes:
#   - THRESHOLDS_FILE is REQUIRED (one threshold per line; '#' comments and blanks allowed)
#   - SAVE_CR / SAVE_CR_BND / SAVE_NUMDEN default to 0/0/1 if not provided
# ============================================================

set -euo pipefail

# --- Project root
PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- Config (env or defaults)
: "${INDIR:?Need INDIR (dir with PNG/TIF images)}"
: "${OUTDIR:=scratch/threshold_wk}"
: "${THRESHOLDS_FILE:?Need THRESHOLDS_FILE (path to text file with thresholds)}"
: "${MIN_AREA:=100}"
: "${MAX_AREA:=7500000}"
: "${PREFIX:=thr_wk}"
: "${ROWS_PER_FLUSH:=400}"
: "${MAX_BYTES_PER_FLUSH:=$((128*1024*1024))}"
: "${INNER_SBATCH:=clouds/data_processing_scripts/image_proc/threshold_wk_per_image.sbatch}"

# New optional persistence flags for the Python script
: "${SAVE_CR:=0}"
: "${SAVE_CR_BND:=0}"
: "${SAVE_NUMDEN:=1}"   # keep on by default since aggregates depend on it

mkdir -p "$OUTDIR" scratch/logs

# --- Validate THRESHOLDS_FILE early
if [[ ! -f "$THRESHOLDS_FILE" ]]; then
  echo "[ERR] THRESHOLDS_FILE not found: $THRESHOLDS_FILE" >&2
  exit 1
fi

# --- Collect images (PNG/TIF/TIFF, non-recursive)
shopt -s nullglob
mapfile -t IMAGES < <(find "$INDIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | sort)
if (( ${#IMAGES[@]} == 0 )); then
  echo "[ERR] No images found in $INDIR" >&2
  exit 1
fi

echo "[INFO] Found ${#IMAGES[@]} images in $INDIR"
echo "[INFO] OUTDIR=${OUTDIR}"
echo "[INFO] THRESHOLDS_FILE=${THRESHOLDS_FILE}"
echo "[INFO] MIN_AREA=${MIN_AREA}, MAX_AREA=${MAX_AREA}"
echo "[INFO] SAVE_CR=${SAVE_CR}, SAVE_CR_BND=${SAVE_CR_BND}, SAVE_NUMDEN=${SAVE_NUMDEN}"
echo "[INFO] Submitting one job per image..."

# --- Submit one job per image
for IMG in "${IMAGES[@]}"; do
  stem="$(basename "${IMG%.*}")"
  sbatch --export=ALL,\
IMG="$IMG",\
OUTDIR="$OUTDIR",\
THRESHOLDS_FILE="$THRESHOLDS_FILE",\
MIN_AREA="$MIN_AREA",\
MAX_AREA="$MAX_AREA",\
PREFIX="$PREFIX",\
ROWS_PER_FLUSH="$ROWS_PER_FLUSH",\
MAX_BYTES_PER_FLUSH="$MAX_BYTES_PER_FLUSH",\
SAVE_CR="$SAVE_CR",\
SAVE_CR_BND="$SAVE_CR_BND",\
SAVE_NUMDEN="$SAVE_NUMDEN" \
"$INNER_SBATCH"
done
