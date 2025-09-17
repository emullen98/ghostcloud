#!/usr/bin/env bash
# ======================================
# Common pipeline config (source me!)
# ======================================

# NOTE: YOU MUST CHANGE THE --ARRAY VALUES IN THE SBATCH SCRIPTS
# IF YOU CHANGE THE GAMMAS, FILL_THRESHOLDS, OR NUM_SLICES

# --- Roots ---
OUTPUT_ROOT="/scratch/mbiju2/storm"
LOG_ROOT="/scratch/mbiju2/logs"

# You can override DATE_TAG on submit; default is now
: "${DATE_TAG:=$(date +'%Y%m%d_%H%M%S')}"

# --- Shared scientific knobs (USED BY BOTH DATA-GEN & ANALYSIS) ---
# [CHANGE] Single source of truth for gamma, p, slices:
#GAMMAS=(0.15 0.2 0.25 0.3 0.35 0.4 0.45)
#FILL_THRESHOLDS=(0.375 0.40 0.425 0.45 0.475 0.5 0.525)
GAMMAS=(0.15 0.2 0.25 0.3 0.35 0.4 0.45)
FILL_THRESHOLDS=(0.375 0.40 0.425 0.45)
NUM_SLICES=20  # number of *slice segments* used during slicing (e.g., 20 or 30)

# [CHANGE][OPTIONAL] Analysis convention for "full cloud" row
INCLUDE_FULL_CLOUD=1   # 1 = include a "-1" slice_id, 0 = don’t

# --- Data-gen knobs ---
LATTICE_SIZE=15000
MIN_CLOUD_AREA=8000
MIN_SLICE_WIDTH=3
NUM_LATTICES=50

# --- Derived paths for this run ---
BASE_RUN_DIR="${OUTPUT_ROOT}/cp_crit_exp_${DATE_TAG}"
BASE_LOG_DIR="${LOG_ROOT}/cp_crit_exp_${DATE_TAG}"

# --- Helpers ---
tagify() {  # 0.475 -> p_0p475 ; 0.2 -> g_0p2
  local prefix="$1"; local value="$2"; local t="${value/./p}"; echo "${prefix}_${t}"
}

# [CHANGE][OPTIONAL] Build tag arrays for analysis from the shared floats
build_tag_arrays() {
  GAMMA_TAGS=()
  for g in "${GAMMAS[@]}"; do GAMMA_TAGS+=("$(tagify g "$g")"); done
  P_TAGS=()
  for p in "${FILL_THRESHOLDS[@]}"; do P_TAGS+=("$(tagify p "$p")"); done

  # NUM_SLICES_TOTAL = NUM_SLICES (+1 if we include full cloud)
  if [[ "${INCLUDE_FULL_CLOUD}" -eq 1 ]]; then
    NUM_SLICES_TOTAL=$(( NUM_SLICES + 1 ))
  else
    NUM_SLICES_TOTAL=$(( NUM_SLICES ))
  fi
}
