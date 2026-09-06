#!/usr/bin/env bash
# Local / Lightning AI entry point for THE OBSERVER (Film II).
# On Kaggle use kaggle_run_observer.ipynb instead (adds 2x T4 hardening + cost gate).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export KIT_CONFIG="config_observer.yaml"

# 0) workflows exist (includes the charforge text-to-image graph)
[ -d workflows ] && [ -n "$(ls -A workflows 2>/dev/null)" ] || python3 src/make_workflows.py

# 1) start ComfyUI headless if not already up
COMFY="${COMFYUI:-$ROOT/comfyui}"
COMFY_FLAGS="$(python3 -c 'import sys;sys.path.insert(0,"src");from cfgutil import load_cfg;print(load_cfg(".").get("comfy_flags","--cache-none"))' 2>/dev/null || echo --cache-none)"
if ! curl -s http://127.0.0.1:8188/ >/dev/null 2>&1; then
  if [ -d "$COMFY" ]; then
    echo "[run] starting ComfyUI headless..."
    ( cd "$COMFY" && python3 main.py --listen 127.0.0.1 --port 8188 $COMFY_FLAGS \
        > "$ROOT/comfyui_observer.log" 2>&1 & echo $! > "$ROOT/comfyui.pid" )
    for i in $(seq 1 60); do curl -s http://127.0.0.1:8188/ >/dev/null 2>&1 && break; sleep 3; done
  else
    echo "[run] NOTE: ComfyUI not found at $COMFY. Set \$COMFYUI. Continuing (stages that"
    echo "      need the server will report and resume once it is up)."
  fi
fi

# 2) run the Film II pipeline (resumable; forges the character, then holds at the cost meter)
python3 src/filmforge_observer.py
echo "[run] done -> output_observer/THE_OBSERVER.mp4"
