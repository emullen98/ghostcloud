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
# sbatch --export=ALL,INDIR=clouds/cloud_data/cleaned,OUTDIR=scratch/threshold_wk,THRESHOLDS_FILE=clouds/data_processing_scripts/image_proc/thresholds.txt,PREFIX=thr_wk clouds/data_processing_scripts/image_proc/threshold_wk_wrapper.sbatch
# ============================================================

set -euo pipefail

# --- Project root
PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# If a file is provided, build a CSV from it (one threshold per line)
if [[ -n "${THRESHOLDS_FILE:-}" ]]; then
  if [[ ! -f "$THRESHOLDS_FILE" ]]; then
    echo "[ERR] THRESHOLDS_FILE not found: $THRESHOLDS_FILE" >&2
    exit 1
  fi
  THRESHOLDS="$(tr '\n' ',' < "$THRESHOLDS_FILE" | sed 's/,$//')"
fi

if [[ -z "${THRESHOLDS:-}" ]]; then
  echo "[ERR] THRESHOLDS not set (use THRESHOLDS or THRESHOLDS_FILE)" >&2
  exit 1
fi


# --- Config (env or defaults)
: "${INDIR:?Need INDIR (dir with PNGs)}"
: "${OUTDIR:=scratch/threshold_wk}"
: "${THRESHOLDS_FILE:?Need THRESHOLDS_FILE (path to text file with thresholds)}"
: "${MIN_AREA:=100}"
: "${MAX_AREA:=7500000}"
: "${PREFIX:=thr_wk}"
: "${ROWS_PER_FLUSH:=400}"
: "${MAX_BYTES_PER_FLUSH:=$((128*1024*1024))}"
: "${INNER_SBATCH:=clouds/data_processing_scripts/image_proc/threshold_wk_per_image.sbatch}"

mkdir -p "$OUTDIR" scratch/logs

# --- Collect images
shopt -s nullglob
mapfile -t IMAGES < <(find "$INDIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | sort)

if (( ${#IMAGES[@]} == 0 )); then
  echo "[ERR] No images found in $INDIR" >&2
  exit 1
fi

echo "[INFO] Found ${#IMAGES[@]} images in $INDIR"
echo "[INFO] Submitting one job per image..."

for IMG in "${IMAGES[@]}"; do
  stem="$(basename "${IMG%.*}")"
  sbatch --export=ALL,IMG="$IMG",OUTDIR="$OUTDIR",THRESHOLDS_FILE="$THRESHOLDS_FILE",MIN_AREA="$MIN_AREA",MAX_AREA="$MAX_AREA",PREFIX="$PREFIX",ROWS_PER_FLUSH="$ROWS_PER_FLUSH",MAX_BYTES_PER_FLUSH="$MAX_BYTES_PER_FLUSH" "$INNER_SBATCH"
done
