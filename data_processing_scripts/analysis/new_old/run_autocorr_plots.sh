#!/usr/bin/env bash
set -euo pipefail
# One-shot: make separate plots for all autocorr outputs (PNG + SITEPERC).
# Produces one PNG per CSV for: num, den, Cr (log–log axes).
#
# Usage:
#   bash analysis/run_autocorr_plots.sh
#   # (optional) override python: PYTHON_BIN=/path/to/venv/bin/python bash analysis/run_autocorr_plots.sh
#
# Assumes repo layout:
#   <REPO_TOP>/
#     clouds/
#       data_processing_scripts/
#         analysis/
#           plotter.py  (importable as clouds.data_processing_scripts.analysis.plotter)
#     scratch/all_clouds_data/analysis/autocorr/*.csv   (aggregator outputs)

PY="${PYTHON_BIN:-python}"

# Locate repo roots
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../clouds/data_processing_scripts/analysis
DPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                           # .../clouds/data_processing_scripts
CLOUDS_DIR="$(cd "${DPS_DIR}/.." && pwd)"                           # .../clouds
REPO_TOP="$(cd "${CLOUDS_DIR}/.." && pwd)"                          # parent of clouds/

MODULE="clouds.data_processing_scripts.analysis.plotter"
OUTDIR="${HOME}/scratch/all_clouds_data/analysis/autocorr"

# Helper: run plotter as module from REPO_TOP with correct PYTHONPATH
run_plotter() {
  ( cd "$REPO_TOP" && PYTHONPATH="$REPO_TOP${PYTHONPATH+:$PYTHONPATH}" "$PY" -m "$MODULE" "$@" )
}

# Gather CSV sets; skip gracefully if any set is empty
shopt -s nullglob
NUM_CSVS=("${OUTDIR}"/*__autocorr_num.csv)
DEN_CSVS=("${OUTDIR}"/*__autocorr_den.csv)
CR_CSVS=("${OUTDIR}"/*__autocorr_Cr.csv)
shopt -u nullglob

echo "[INFO] Output directory: $OUTDIR"

# 1) Aggregated Numerator
if (( ${#NUM_CSVS[@]} )); then
  echo "[PLOT] Autocorr numerator → separate plots (${#NUM_CSVS[@]} files)"
  run_plotter autocorr \
    --csv "${NUM_CSVS[@]}" \
    --field y \
    --logx --logy \
    --legend stem \
    --mode separate \
    --title "Autocorr — Aggregated Numerator" \
    --out-template "${OUTDIR}/{stem}.png"
else
  echo "[SKIP] No *__autocorr_num.csv files found"
fi

# 2) Aggregated Denominator
if (( ${#DEN_CSVS[@]} )); then
  echo "[PLOT] Autocorr denominator → separate plots (${#DEN_CSVS[@]} files)"
  run_plotter autocorr \
    --csv "${DEN_CSVS[@]}" \
    --field y \
    --logx --logy \
    --legend stem \
    --mode separate \
    --title "Autocorr — Aggregated Denominator" \
    --out-template "${OUTDIR}/{stem}.png"
else
  echo "[SKIP] No *__autocorr_den.csv files found"
fi

# 3) C(r) = num/den
if (( ${#CR_CSVS[@]} )); then
  echo "[PLOT] Autocorr C(r) → separate plots (${#CR_CSVS[@]} files)"
  run_plotter autocorr \
    --csv "${CR_CSVS[@]}" \
    --field y \
    --logx --logy \
    --legend stem \
    --mode separate \
    --title "Autocorr — C(r)" \
    --out-template "${OUTDIR}/{stem}.png"
else
  echo "[SKIP] No *__autocorr_Cr.csv files found"
fi

echo "[DONE] Separate plots generated where inputs existed."
