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
# sbatch --export=ALL,INDIR=data/raw_clouds,OUTDIR=scratch/clouds_yen,PROJECT_ROOT=/u/mbiju2 \
#        clouds/utils/yen_binarize.sbatch
# ============================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"

INDIR="${INDIR:-data/raw_clouds}"
OUTDIR="${OUTDIR:-scratch/clouds_yen}"

# --- Environment
module purge || true
source clouds/venv/bin/activate

mkdir -p "${OUTDIR}" scratch/logs

echo "[INFO] Input dir: ${INDIR}"
echo "[INFO] Output dir: ${OUTDIR}"

srun -u python -u clouds/utils/yen_binarize.py \
  --indir "${INDIR}" \
  --outdir "${OUTDIR}"
