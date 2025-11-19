#!/usr/bin/env bash
# Serialized runner for expanded configs with nice progress lines.
# Usage:
#   bash run_all.sh --master MASTER.yaml
#
# Notes:
# - No parallelism.
# - Works on macOS default Bash (no `mapfile` needed).
# - Aggregators are invoked as Python modules via `-m`.
# - Run this from your repository root (e.g., $HOME) so that the `clouds` package is importable.
#   If needed, you can export PYTHONPATH to include the repo root:
#     export PYTHONPATH="${PYTHONPATH:-$HOME}"
#
# Environment overrides:
#   PYTHON=python3.12 bash run_all.sh --master MASTER.yaml

# bash clouds/data_processing_scripts/analysis/run_all.sh --master clouds/data_processing_scripts/analysis/configs/orchestrator.yaml

set -euo pipefail

# ------------------- EDIT THESE if module paths differ --------------
PYTHON="${PYTHON:-python3}"
# Module names (no ".py"—these are import paths!)
MOD_FD="clouds.data_processing_scripts.analysis.fd_aggregate"
MOD_AC="clouds.data_processing_scripts.analysis.autocorr_aggregate"
MOD_BD="clouds.data_processing_scripts.analysis.bdd_aggregate"
# -------------------------------------------------------------------

if [[ "${1:-}" != "--master" || -z "${2:-}" ]]; then
  echo "Usage: bash run_all.sh --master MASTER.yaml" >&2
  exit 2
fi

MASTER="$2"

# --- Helper: read a value from the MASTER yaml via a Python expr on 'm' ---
# Usage: py_get 'm["paths"]["output_root"]'
py_get() {
  local expr="$1"
  "$PYTHON" - "$MASTER" "$expr" <<'PY'
import sys, yaml, json
master_path, expr = sys.argv[1], sys.argv[2]
with open(master_path, "r") as f:
    m = yaml.safe_load(f)
val = eval(expr, {"__builtins__": {}}, {"m": m})
if isinstance(val, (dict, list)):
    print(json.dumps(val))
else:
    print("" if val is None else str(val))
PY
}

OUT_ROOT="$(py_get 'm["paths"]["output_root"]')"
PROFILE="$(py_get 'm.get("profile", "default")')"
# Remove spaces from profile for folder names
PROFILE_SAFE="${PROFILE// /}"
CFG_DIR="$OUT_ROOT/configs/$PROFILE_SAFE"

if [[ ! -d "$CFG_DIR" ]]; then
  echo "ERROR: configs dir not found: $CFG_DIR" >&2
  exit 2
fi

# Collect and count configs (portable: no mapfile, no -print0/-z)
CONFIG_LIST_CMD="find \"$CFG_DIR\" -type f -name '*.yaml' | sort"
# Get total count
N=$(eval "$CONFIG_LIST_CMD" | wc -l | awk '{print $1}')

if [[ "$N" -eq 0 ]]; then
  echo "No configs found under $CFG_DIR"
  exit 0
fi

echo "Running $N configs (profile=$PROFILE) from: $CFG_DIR"
ok=0
fail=0
i=0

# Iterate serialized over each config path
eval "$CONFIG_LIST_CMD" | while IFS= read -r cfg; do
  i=$((i+1))

  # Extract a few fields for progress printing using a literal heredoc
  read -r metric source win p tag <<EOF
$("$PYTHON" - "$cfg" <<'PY'
import sys,yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    c = yaml.safe_load(f)
metric = c.get("metric","?")
source = c.get("source","?")
filt = c.get("filters",{}) or {}
# time_windows is always a list in single-run; siteperc uses ["all"]
win = (filt.get("time_windows") or ["?"])[0]
pvs = filt.get("p_vals")
if isinstance(pvs, list) and pvs:
    p = pvs[0]
elif pvs == "all":
    p = "all"
else:
    p = ""
tag = c.get("output",{}).get("TAG","")
print(metric, source, win, p, tag)
PY
)"
EOF

  # Choose aggregator module
  case "$metric" in
    fractal_dimension) module="$MOD_FD" ;;
    autocorr)          module="$MOD_AC" ;;
    bdd)               module="$MOD_BD" ;;
    *)
      echo "[$i/$N] SKIP unknown metric in $cfg: '$metric'"
      fail=$((fail+1))
      continue
      ;;
  esac

  printf '[%d/%d] %-18s src=%-9s win=%-10s p=%-10s TAG=%s\n' "$i" "$N" "$metric" "$source" "$win" "$p" "$tag"

  # Per-metric logs directory
  logdir="$OUT_ROOT/logs/$PROFILE_SAFE/$metric"
  mkdir -p "$logdir"
  stem="$(basename "$cfg" .yaml)"
  log="$logdir/${stem}.out"

  # Run serialized; capture output to per-config log
  set +e
  "$PYTHON" -m "$module" --config "$cfg" >"$log" 2>&1
  code=$?
  set -e

  if [[ $code -ne 0 ]]; then
    echo "  -> [FAIL] code=$code | log: $log"
    fail=$((fail+1))
  else
    echo "  -> [OK]   log: $log"
    ok=$((ok+1))
  fi
done

# Simple portable summary:
TOTAL="$N"
OK_COUNT=$(find "$OUT_ROOT/logs/$PROFILE_SAFE" -type f -name '*.out' -exec grep -l -m1 '^\[OK\]' /dev/null {} + 2>/dev/null | wc -l | awk '{print $1}')
FAIL_COUNT=$(( TOTAL - OK_COUNT ))

echo "Done. OK=$OK_COUNT FAIL=$FAIL_COUNT TOTAL=$TOTAL"

# Exit non-zero if any failed
if [[ "$FAIL_COUNT" -ne 0 ]]; then
  exit 1
fi
exit 0
