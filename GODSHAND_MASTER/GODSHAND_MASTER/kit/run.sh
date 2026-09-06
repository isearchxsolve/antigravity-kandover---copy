#!/usr/bin/env bash
# Local / Lightning AI entry point. Starts ComfyUI headless, then runs the pipeline.
# On Kaggle use kaggle_run.ipynb instead (adds the 2x T4 hardening).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 0) workflows exist
[ -d workflows ] && [ -n "$(ls -A workflows 2>/dev/null)" ] || python3 src/make_workflows.py

# 1) start ComfyUI headless if not already up (expects comfyui/ alongside kit or $COMFYUI)
COMFY="${COMFYUI:-$ROOT/comfyui}"
COMFY_FLAGS="$(python3 -c 'import sys;sys.path.insert(0,"src");from cfgutil import load_cfg;print(load_cfg(".").get("comfy_flags","--cache-none"))' 2>/dev/null || echo --cache-none)"
if ! curl -s http://127.0.0.1:8188/ >/dev/null 2>&1; then
  if [ -d "$COMFY" ]; then
    echo "[run] starting ComfyUI headless..."
    ( cd "$COMFY" && python3 main.py --listen 127.0.0.1 --port 8188 $COMFY_FLAGS \
        > "$ROOT/comfyui.log" 2>&1 & echo $! > "$ROOT/comfyui.pid" )
    for i in $(seq 1 60); do curl -s http://127.0.0.1:8188/ >/dev/null 2>&1 && break; sleep 3; done
  else
    echo "[run] NOTE: ComfyUI not found at $COMFY. Set \$COMFYUI. Continuing (stages that"
    echo "      need the server will report and resume once it is up)."
  fi
fi

# 2) run the pipeline (resumable)
python3 src/filmforge.py
echo "[run] done -> output/THE_MOMENT.mp4"
