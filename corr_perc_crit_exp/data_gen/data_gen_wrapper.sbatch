#!/bin/bash
#SBATCH --job-name=cp_data_gen_run
#SBATCH --output=/scratch/mbiju2/logs/cp_data_gen_wrapper_%j.out
#SBATCH --error=/scratch/mbiju2/logs/cp_data_gen_wrapper_%j.err
#SBATCH --partition=physics
#SBATCH -t 24:00:00
#SBATCH --ntasks=1

# Source shared params
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/pipeline_params.sh"

# Use shared derived paths
OUTPUT_BASE="${BASE_RUN_DIR}"
LOG_BASE="${BASE_LOG_DIR}"
mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

# Log parameters
PARAM_LOG="$OUTPUT_BASE/parameters_log.txt"
{
  echo "DATE_TAG: ${DATE_TAG}"
  echo "LATTICE_SIZE: ${LATTICE_SIZE}"
  echo "MIN_CLOUD_AREA: ${MIN_CLOUD_AREA}"
  echo "NUM_SLICES: ${NUM_SLICES}"
  echo "MIN_SLICE_WIDTH: ${MIN_SLICE_WIDTH}"
  echo "NUM_LATTICES: ${NUM_LATTICES}"
  echo "FILL_THRESHOLDS: ${FILL_THRESHOLDS[*]}"
  echo "GAMMAS: ${GAMMAS[*]}"
} >> "$PARAM_LOG"

# Loop gamma × p from shared arrays
for GAMMA in "${GAMMAS[@]}"; do
  GAMMA_TAG="$(tagify g "$GAMMA")"   # [CHANGE]
  GAMMA_DIR="$OUTPUT_BASE/$GAMMA_TAG"
  LOG_GAMMA_DIR="$LOG_BASE/$GAMMA_TAG"
  mkdir -p "$GAMMA_DIR" "$LOG_GAMMA_DIR"

  for FILL_THRESH in "${FILL_THRESHOLDS[@]}"; do
    FILL_TAG="$(tagify p "$FILL_THRESH")"  # [CHANGE]
    FILL_DIR="$GAMMA_DIR/$FILL_TAG"
    LOG_FILL_DIR="$LOG_GAMMA_DIR/$FILL_TAG"
    mkdir -p "$FILL_DIR" "$LOG_FILL_DIR"

    sbatch --array=1-$NUM_LATTICES%100 \
           --cpus-per-task=1 \
           --time=4:00:00 \
           --job-name="cp_${GAMMA_TAG}_${FILL_TAG}" \
           --mem=20G \
           --output="${LOG_FILL_DIR}/job_%A_task_%a.out" \
           --error="${LOG_FILL_DIR}/job_%A_task_%a.err" \
           single_lattice_gen.sbatch \
           "$FILL_DIR" \
           "$LATTICE_SIZE" \
           "$FILL_THRESH" \
           "$GAMMA" \
           "$MIN_CLOUD_AREA" \
           "$NUM_SLICES" \
           "$MIN_SLICE_WIDTH"
  done
done
