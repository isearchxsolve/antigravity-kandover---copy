#!/usr/bin/env bash
# assemble.sh [ROOT]  --  100% CPU, OFFLINE, ZERO GPU SPEND.
# Build the final film / clip deliverables from ALREADY-rendered per-shot clips.
# Run this ON YOUR LAPTOP. The paid GPU box should only ever GENERATE; all
# concatenation, music bed, voice mux, ducking, mastering and encoding happen
# here for free (see stitch.py -- it is pure ffmpeg).
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
export POST_MODE=cpu    # if any postprocess runs here, force the free ffmpeg-only path

echo "[assemble] CPU-only offline assembly (POST_MODE=cpu). No GPU required."
command -v ffmpeg  >/dev/null 2>&1 || { echo "[FAIL] ffmpeg not found  (apt/brew install ffmpeg)"; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "[FAIL] ffprobe not found (apt/brew install ffmpeg)"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[FAIL] python3 not found"; exit 1; }

# Fail fast & friendly if the rendered clips were not downloaded from the GPU box.
WORK="$(python3 - "$ROOT" <<'PY'
import sys, os
root = sys.argv[1]; sys.path.insert(0, os.path.join(root, "src"))
from cfgutil import load_cfg
print(load_cfg(root)["paths"]["work"])
PY
)"
if [ ! -d "$WORK" ] || [ -z "$(find "$WORK" -name final.mp4 2>/dev/null | head -n1)" ]; then
  echo "[FAIL] no rendered clips (final.mp4) under $WORK"
  echo "       -> download the per-shot clips from the GPU instance into that folder first."
  exit 1
fi

python3 "$ROOT/src/stitch.py" --root "$ROOT"
echo "[assemble] done -> see $ROOT/output/"
