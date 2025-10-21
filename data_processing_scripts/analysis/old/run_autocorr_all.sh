#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  run_autocorr_all.sh — One command to generate all Autocorr aggregates
#
#  Must be executed from your $HOME directory (which contains scratch/)
#  and the script autocorr.py must be in the same directory
#  as this bash script.
#
#  USAGE:
#    bash run_autocorr_all.sh "<WINDOWS_CSV>" "<SITEPERC_P>" "<WHICH>"
#
#  EXAMPLE:
#    bash run_autocorr_all.sh \
#      "03:00-04:00,04:00-05:00,05:00-06:00,07:00-08:00,08:00-09:00,09:00-10:00,11:00-12:00,alltimes" \
#      0.4074 \
#      all
#
#  This runs:
#    - PNG (all thresholds) for each window + alltimes
#    - PNG (argmax-clouds) for each window + alltimes
#    - SITEPERC (single pass)
#
#  Outputs go to: $HOME/scratch/all_clouds_data/analysis/autocorr/
# ============================================================

WINDOWS_CSV="${1:-alltimes}"
SITEPERC_P="${2:-all}"
WHICH_PAIR="${3:-all}"  # all | bnd

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/autocorr.py"

# ---- sanity checks ----
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] autocorr.py not found next to run_autocorr_all.sh"
  echo "        Expected: $SCRIPT_PATH"
  exit 1
fi

if [[ ! -d "$HOME/scratch" ]]; then
  echo "[ERROR] Expected 'scratch/' directory under \$HOME but not found."
  echo "        Current \$HOME: $HOME"
  exit 1
fi

IFS=',' read -r -a WINDOWS <<< "$WINDOWS_CSV"

# helper: "03:00-04:00" -> "0300", "alltimes" -> "alltimes"
window_token() {
  local w="$1"
  if [[ "$w" == "alltimes" ]]; then
    echo "alltimes"
  else
    echo "${w%%-*}" | tr -d ':'
  fi
}

run_one() {
  echo "[RUN]" "$@"
  "$PYTHON_BIN" "$SCRIPT_PATH" "$@"
  echo
}

# ============================================================
#  PASS 1: PNG — all thresholds
# ============================================================
for W in "${WINDOWS[@]}"; do
  TOKEN=$(window_token "$W")
  SUFFIX="PNG_allthr_${TOKEN}"
  TIME_ARG="$([[ $W == alltimes ]] && echo all || echo "$W")"

  run_one \
    --source png \
    --threshold_policy all \
    --time_window "$TIME_ARG" \
    --which "$WHICH_PAIR" \
    --suffix_tag "$SUFFIX"
done

# ============================================================
#  PASS 2: PNG — argmax-clouds
# ============================================================
for W in "${WINDOWS[@]}"; do
  TOKEN=$(window_token "$W")
  SUFFIX="PNG_maxthr_${TOKEN}"
  TIME_ARG="$([[ $W == alltimes ]] && echo all || echo "$W")"

  run_one \
    --source png \
    --threshold_policy argmax_clouds \
    --time_window "$TIME_ARG" \
    --which "$WHICH_PAIR" \
    --suffix_tag "$SUFFIX"
done

# ============================================================
#  PASS 3: SITEPERC
# ============================================================
p_token="all"
if [[ "$SITEPERC_P" != "all" ]]; then
  p_token="p$(echo "$SITEPERC_P" | sed -E 's/[^0-9]//g')"
fi
SUFFIX="SITEPERC_${p_token}"

run_one \
  --source siteperc \
  --p_vals "$SITEPERC_P" \
  --which "$WHICH_PAIR" \
  --suffix_tag "$SUFFIX"

echo "[DONE] All autocorr runs complete."
