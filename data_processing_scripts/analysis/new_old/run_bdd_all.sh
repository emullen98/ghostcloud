#!/usr/bin/env bash
set -euo pipefail
# Must run from $HOME. Expects bnd_dd.py next to this script.

WINDOWS_CSV="${1:-defaults}"
SITEPERC_P="${2:-defaults}"

PY="${PYTHON_BIN:-python}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${DIR}/bnd_dd.py"
CFG="${DIR}/config.yaml"
CFGCLI="${DIR}/cfg_cli.py"

[[ -f "$SCRIPT" ]] || { echo "[ERR] bnd_dd.py not found: $SCRIPT"; exit 1; }
[[ -d "$HOME/scratch" ]] || { echo "[ERR] scratch/ not found under \$HOME"; exit 1; }

if [[ "$WINDOWS_CSV" == "defaults" ]]; then
  WINS="$($PY "$CFGCLI" --config "$CFG" --getcsv png.time_windows)"
  [[ -z "$WINS" ]] && WINS="all"
else
  WINS="$WINDOWS_CSV"
fi
IFS=',' read -r -a WINDOWS <<< "$WINS"

if [[ "$SITEPERC_P" == "defaults" ]]; then
  PV="$($PY "$CFGCLI" --config "$CFG" --getcsv siteperc.p_vals)"
  [[ -z "$PV" ]] && PV="all"
else
  PV="$SITEPERC_P"
fi

window_token(){ local w="$1"; [[ "$w" == "all" || "$w" == "alltimes" ]] && echo "alltimes" || echo "${w%%-*}" | tr -d ':'; }
run_one(){ echo "[RUN]" "$@"; "$PY" "$SCRIPT" "$@"; echo; }

# PNG all thresholds
for W in "${WINDOWS[@]}"; do
  TKN=$(window_token "$W"); PREF="PNG_allthr_${TKN}"
  TWARG="$([[ "$W" == "all" || "$W" == "alltimes" ]] && echo all || echo "$W")"
  run_one --source png --time_window "$TWARG" --threshold_policy all --prefix "$PREF"
done

# PNG argmax
for W in "${WINDOWS[@]}"; do
  TKN=$(window_token "$W"); PREF="PNG_maxthr_${TKN}"
  TWARG="$([[ "$W" == "all" || "$W" == "alltimes" ]] && echo all || echo "$W")"
  run_one --source png --time_window "$TWARG" --threshold_policy argmax_clouds --prefix "$PREF"
done

# SITEPERC
PTOK="all"; [[ "$PV" != "all" ]] && PTOK="p$(echo "$PV" | sed -E 's/[^0-9]//g')"
PREF="SITEPERC_${PTOK}"
run_one --source siteperc --p_vals "$PV" --prefix "$PREF"
echo "[DONE] BDD runs complete."
