#!/bin/bash
#SBATCH --job-name=yen_binarize
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=scratch/logs/yen_binarize_%A.out
#SBATCH --error=scratch/logs/yen_binarize_%A.err

# ============================================================
# Example submit:
# sbatch --export=ALL,INDIR=clouds/cloud_data/cleaned,OUTDIR=scratch/clouds_yen,PROJECT_ROOT=/u/mbiju2 clouds/data_processing_scripts/thresholding/yen_binarize.sbatch
# ============================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"

INDIR="${INDIR:-data/raw_clouds}"
OUTDIR="${OUTDIR:-scratch/clouds_yen}"

# --- Environment setup ---
source clouds/venv/bin/activate

mkdir -p "${OUTDIR}" scratch/logs

echo "[INFO] Input dir: ${INDIR}"
echo "[INFO] Output dir: ${OUTDIR}"

srun -u python -u -m clouds.data_processing_scripts.thresholding.yen_binarize \
  --indir "${INDIR}" \
  --outdir "${OUTDIR}"
