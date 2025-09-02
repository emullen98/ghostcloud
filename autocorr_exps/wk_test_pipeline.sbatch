#!/bin/bash
#SBATCH --job-name=wk_autocorr
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/wk_%A_%a.out
#SBATCH --error=logs/wk_%A_%a.err

# SAMPLE COMMAND:
# sbatch --export=ALL,LATTICE_SIZE=2000,FILL_PROB=0.4074,SEEDS_FILE=clouds/autocorr_exps/seeds_100.txt,OUTDIR=scratch/debug/,PREFIX=new_max_r clouds/autocorr_exps/wk_test_pipeline.sbatch

set -euo pipefail

# --- Set your project root explicitly (override at submit time if needed)
PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"   # <-- EDIT THIS
cd "$PROJECT_ROOT"

# --- Config (override with: sbatch --export=ALL,VAR=... ...)
LATTICE_SIZE="${LATTICE_SIZE:-2000}"
FILL_PROB="${FILL_PROB:-0.4074}"
SEEDS_FILE="${SEEDS_FILE:-seeds_100.txt}"   # one seed per line
OUTDIR="${OUTDIR:-scratch}"
PREFIX="${PREFIX:-wk}"

# --- Environment
module load cuda/12.8
source clouds/venv/bin/activate

mkdir -p logs "${OUTDIR}"

# --- Pick seed for this array index (0..99), sed is 1-indexed
if [[ ! -f "${SEEDS_FILE}" ]]; then
  echo "[ERR] Seeds file not found: ${SEEDS_FILE}" >&2
  exit 1
fi
SEED="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SEEDS_FILE}")"
if [[ -z "${SEED}" ]]; then
  echo "[ERR] No seed for index ${SLURM_ARRAY_TASK_ID} in ${SEEDS_FILE}" >&2
  exit 1
fi

echo "[INFO] LATTICE_SIZE=${LATTICE_SIZE} FILL_PROB=${FILL_PROB} SEED=${SEED}"

# --- Run module (unbuffered stdout)
#srun -u python -u -m clouds.autocorr_exps.wk_optimal \
#  --lattice-size "${LATTICE_SIZE}" \
#  --fill-prob "${FILL_PROB}" \
#  --seed "${SEED}" \
#  --outdir "${OUTDIR}" \
#  --prefix "${PREFIX}"

srun -u python -u -m clouds.autocorr_exps.annulus_example \
