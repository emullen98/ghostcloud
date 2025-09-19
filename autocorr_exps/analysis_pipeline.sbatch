#!/bin/bash
#SBATCH --job-name=fit_corr
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes  # CPU partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/fit_%A_%a.out
#SBATCH --error=logs/fit_%A_%a.err

set -euo pipefail

# --- Set your project root explicitly (override at submit time if needed)
PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"

# --- Config
CORR_DIR="${CORR_DIR:-scratch/correlations}"  # Directory with raw correlation data
OUTDIR="${OUTDIR:-scratch/fits}"
PREFIX="${PREFIX:-wk}"

# --- Environment (no CUDA needed)
source clouds/venv/bin/activate

mkdir -p logs "${OUTDIR}"

# Process all correlation files for this array task
python -u -m clouds.autocorr_exps.analysis_per_lattice \
  --input-dir "${CORR_DIR}" \
  --output-dir "${OUTDIR}" \
  --array-task "${SLURM_ARRAY_TASK_ID}" \
  --array-size "${SLURM_ARRAY_TASK_COUNT}" \
  --prefix "${PREFIX}"