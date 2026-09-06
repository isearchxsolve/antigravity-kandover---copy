#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# THE CONFRONTATION CLIP — MASTER EXECUTION SCRIPT
# One file. One command. End to end.
#
# USAGE:
#   bash MASTER.sh                  # DRY RUN (free — audits, cost-projects, holds)
#   bash MASTER.sh --render         # FULL RUN (stills → motion, the blueprint path)
#   bash MASTER.sh --stills-only    # FULL RUN (comic-book path, stills only, fastest)
#   bash MASTER.sh --motion-only    # FULL RUN (old text-to-motion path, no blueprint)
#
# GUARDRAILS (optional, prepend before the command):
#   MAX_BUDGET_USD=8 bash MASTER.sh --render
#   MAX_HOURS=6 bash MASTER.sh --render
#   AUTO_SHUTDOWN=1 bash MASTER.sh --render
#   RATE_PER_GPU_HR=0 bash MASTER.sh --render   # if on free Lightning hours
#
# WHAT IT DOES (in order):
#   1.  Verify the kit (free audit, zero GPU)
#   2.  Boot check (models, workflows, deps, GPU, disk)
#   3.  Forge character identities (idempotent)
#   4.  Cost meter (1 sample shot, real speed, project cost)
#   5.  Budget gate (abort if over ceiling)
#   6.  Render — stills first, then animate each still (blueprint path, 1% GPU load)
#   7.  Voice (Kokoro TTS — Jack wounded-lion, Julie regal-shattered)
#   8.  Score (original orchestral — prelude, search, confrontation)
#   9.  Sound design (water/seepage, welding, sparks, gunfire, body-fall, hall echo)
#   10. Finale (the film's own images curdle into the joke)
#   11. Stitch + master (video + score + ambience + VO + finale → THE_CONFRONTATION_CLIP.mp4)
#
# OUTPUT:
#   output_confrontation/THE_CONFRONTATION_CLIP.mp4
#   output_confrontation/THE_CONFRONTATION_CLIP_delivery.zip
# ===================================================================

cd "$(dirname "$0")"
export KIT_CONFIG="config_confrontation.yaml"
export PYTHONPATH="src:${PYTHONPATH:-}"
PY="${PYTHON3:-python3}"

# --- Parse mode ---
MODE="dry"
RENDER_MODE="motion_stills"   # default: blueprint path
if [[ "${1:-}" == "--render" ]]; then
    MODE="full"
    RENDER_MODE="motion_stills"
elif [[ "${1:-}" == "--stills-only" ]]; then
    MODE="full"
    RENDER_MODE="stills"
elif [[ "${1:-}" == "--motion-only" ]]; then
    MODE="full"
    RENDER_MODE="motion"
elif [[ -n "${1:-}" ]]; then
    echo "[error] unknown argument: $1"
    echo "Usage: bash MASTER.sh [--render | --stills-only | --motion-only]"
    exit 1
fi

echo "================================================================"
echo "  THE CONFRONTATION CLIP — MASTER SCRIPT"
echo "================================================================"
echo "  mode        : $MODE"
echo "  render path : $RENDER_MODE"
if [[ "$MODE" == "full" ]]; then
    echo "  budget      : ${MAX_BUDGET_USD:-none}"
    echo "  max hours   : ${MAX_HOURS:-none}"
    echo "  auto-shutdown: ${AUTO_SHUTDOWN:-0}"
    echo "  GPU rate    : ${RATE_PER_GPU_HR:-unset}"
fi
echo "================================================================"

# --- 1. Free audit (zero GPU) ---
echo -e "\n[1/11] Deep audit (free, zero GPU)..."
$PY /data/validate.py || { echo "[FAIL] audit failed — fix errors before spending GPU."; exit 1; }

# --- 2-5. run.py dry run (preflight + forge + cost meter + gates) ---
echo -e "\n[2/11] Preflight + cost projection..."
if [[ "$MODE" == "full" ]]; then
    export CONFIRM_FULL_RUN=1
    export KIT_RENDER_MODE="$RENDER_MODE"
    # If free Lightning hours, set rate to 0 so the budget gate always passes
    if [[ -n "${RATE_PER_GPU_HR:-}" ]]; then
        # Patch the config rate on the fly
        $PY -c "
import yaml, os
cfg = yaml.safe_load(open('$KIT_CONFIG'))
cfg['rate_per_gpu_hr'] = float(os.environ.get('RATE_PER_GPU_HR', 0))
yaml.dump(cfg, open('$KIT_CONFIG','w'), default_flow_style=False, sort_keys=False)
"
    fi
else
    export CONFIRM_FULL_RUN=0
fi

# --- 3-11. Run the full pipeline (or dry run if not --render) ---
$PY run.py

# --- Done ---
echo -e "\n================================================================"
echo "  DONE"
echo "  output: output_confrontation/THE_CONFRONTATION_CLIP.mp4"
echo "  zip:    output_confrontation/THE_CONFRONTATION_CLIP_delivery.zip"
echo "================================================================"
if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
    echo "[shutdown] stopping instance to prevent idle burn..."
    ${KIT_SHUTDOWN_CMD:-sudo shutdown -h now} || true
fi
