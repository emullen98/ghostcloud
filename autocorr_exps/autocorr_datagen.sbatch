#!/bin/bash
#SBATCH --job-name=autocorr_datagen
#SBATCH --account=mbiju2-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --array=0-99
#SBATCH --output=scratch/logs/datagen_%A/datagen_%A_%a.out
#SBATCH --error=scratch/logs/datagen_%A/datagen_%A_%a.err

# ============================================================
# Example submit command:
# sbatch --export=ALL,LATTICE_SIZE=10000,FILL_PROB=0.4074,SEEDS_FILE=clouds/autocorr_exps/seeds_100.txt,OUTDIR=scratch/autocorr_datagen/,PREFIX=autocorr_datagen,MIN_AREA=1000,MAX_AREA=7500000 clouds/autocorr_exps/autocorr_datagen.sbatch
# ============================================================

set -euo pipefail

# --- Project root (override at submit with PROJECT_ROOT=... if needed)
PROJECT_ROOT="${PROJECT_ROOT:-/u/mbiju2}"
cd "$PROJECT_ROOT"

# --- Config (can be overridden via --export at submit)
LATTICE_SIZE="${LATTICE_SIZE:-10000}"
FILL_PROB="${FILL_PROB:-0.4074}"
SEEDS_FILE="${SEEDS_FILE:-seeds_100.txt}"   # file with one seed per line
OUTDIR="${OUTDIR:-scratch}"
PREFIX="${PREFIX:-datagen}"
MIN_AREA="${MIN_AREA:-1000}"
MAX_AREA="${MAX_AREA:-7500000}"

# --- Environment (adjust if your module names differ)
module load cuda/12.8
source clouds/venv/bin/activate

mkdir -p logs "${OUTDIR}"

# --- Pick seed for this array index (0..N-1)
if [[ ! -f "${SEEDS_FILE}" ]]; then
  echo "[ERR] Seeds file not found: ${SEEDS_FILE}" >&2
  exit 1
fi
SEED="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SEEDS_FILE}")"
if [[ -z "${SEED}" ]]; then
  echo "[ERR] No seed for index ${SLURM_ARRAY_TASK_ID} in ${SEEDS_FILE}" >&2
  exit 1
fi

echo "[INFO] LATTICE_SIZE=${LATTICE_SIZE} FILL_PROB=${FILL_PROB} SEED=${SEED} MIN_AREA=${MIN_AREA}"
echo "[INFO] Output dir: ${OUTDIR}"

# --- Run autocorr_datagen module
srun -u python -u -m clouds.autocorr_exps.autocorr_datagen \
  --lattice-size "${LATTICE_SIZE}" \
  --fill-prob "${FILL_PROB}" \
  --seed "${SEED}" \
  --outdir "${OUTDIR}" \
  --prefix "${PREFIX}" \
  --min-area "${MIN_AREA}" \
  --max-area "${MAX_AREA}"
