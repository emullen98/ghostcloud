#!/usr/bin/env bash
set -euo pipefail
# Must run from $HOME. Runs analysis/autocorr.py as a module to satisfy package imports.

WINDOWS_CSV="${1:-defaults}"
SITEPERC_P="${2:-defaults}"
WHICH_PAIR="${3:-all}"  # all | bnd

PY="${PYTHON_BIN:-python}"

# This script lives at .../clouds/data_processing_scripts/analysis/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # .../clouds/data_processing_scripts/analysis
DPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                             # .../clouds/data_processing_scripts
CLOUDS_DIR="$(cd "${DPS_DIR}/.." && pwd)"                             # .../clouds
REPO_TOP="$(cd "${CLOUDS_DIR}/.." && pwd)"                            # **parent of clouds/** (e.g., /u/mbiju2)

MODULE="clouds.data_processing_scripts.analysis.autocorr"

CFG="${SCRIPT_DIR}/config.yaml"
CFGCLI="${SCRIPT_DIR}/cfg_cli.py"

[[ -f "${SCRIPT_DIR}/autocorr.py" ]] || { echo "[ERR] autocorr.py not found under: ${SCRIPT_DIR}"; exit 1; }
[[ -f "$CFGCLI" ]] || { echo "[ERR] cfg_cli.py not found: $CFGCLI"; exit 1; }
[[ -d "$HOME/scratch" ]] || { echo "[ERR] scratch/ not found under \$HOME"; exit 1; }

# Read defaults from config (ensure PYTHONPATH points to REPO_TOP so imports work inside cfg_cli too)
if [[ "$WINDOWS_CSV" == "defaults" ]]; then
  WINS="$(PYTHONPATH="$REPO_TOP${PYTHONPATH+:$PYTHONPATH}" "$PY" "$CFGCLI" --config "$CFG" --getcsv png.time_windows || true)"
  [[ -z "$WINS" ]] && WINS="all"
else
  WINS="$WINDOWS_CSV"
fi
IFS=',' read -r -a WINDOWS <<< "$WINS"

if [[ "$SITEPERC_P" == "defaults" ]]; then
  PV="$(PYTHONPATH="$REPO_TOP${PYTHONPATH+:$PYTHONPATH}" "$PY" "$CFGCLI" --config "$CFG" --getcsv siteperc.p_vals || true)"
  [[ -z "$PV" ]] && PV="all"
else
  PV="$SITEPERC_P"
fi

window_token() {
  local w="$1"
  if [[ "$w" == "all" || "$w" == "alltimes" ]]; then
    echo "alltimes"
  else
    echo "${w%%-*}" | tr -d ':'
  fi
}

run_one() {
  echo "[RUN]" "$@"
  ( cd "$REPO_TOP" && PYTHONPATH="$REPO_TOP${PYTHONPATH+:$PYTHONPATH}" "$PY" -m "$MODULE" "$@" )
  echo
}

# PNG — thresholds = ALL
for W in "${WINDOWS[@]}"; do
  TKN=$(window_token "$W")
  PREF="PNG_allthr_${TKN}"
  TWARG="$([[ "$W" == "all" || "$W" == "alltimes" ]] && echo all || echo "$W")"
  run_one --source png --threshold_policy all --time_window "$TWARG" --which "$WHICH_PAIR" --prefix "$PREF"
done

# PNG — thresholds = ARGMAX_CLOUDS
for W in "${WINDOWS[@]}"; do
  TKN=$(window_token "$W")
  PREF="PNG_maxthr_${TKN}"
  TWARG="$([[ "$W" == "all" || "$W" == "alltimes" ]] && echo all || echo "$W")"
  run_one --source png --threshold_policy argmax_clouds --time_window "$TWARG" --which "$WHICH_PAIR" --prefix "$PREF"
done

# SITEPERC
PTOK="all"; [[ "$PV" != "all" ]] && PTOK="p$(echo "$PV" | sed -E 's/[^0-9]//g')"
PREF="SITEPERC_${PTOK}"
run_one --source siteperc --p_vals "$PV" --which "$WHICH_PAIR" --prefix "$PREF"

echo "[DONE] Autocorr runs complete (module mode)."
