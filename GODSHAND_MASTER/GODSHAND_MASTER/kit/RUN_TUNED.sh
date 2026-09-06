#!/usr/bin/env bash
# Tuned proof-scene launcher: 4-shot confrontation beat on Kaggle 2x Tesla T4.
# Changes vs stock: kaggle_2xt4 profile, denoise 0.55, shot-to-shot morph ON,
# face-composite OFF, softened temporal denoise, codeformer 0.9->0.5.
set -euo pipefail
cd "$(dirname "$0")"
export KIT_CONFIG=config_confrontation.yaml
export KIT_PROFILE=kaggle_2xt4      # Kaggle free GPUs = 2x Tesla T4 (there is NO L4 on Kaggle)
export KIT_MOTION_DENOISE=0.55      # motion freedom (was 0.25)
export KIT_FLF2V_ENDFRAME=1         # shot-to-shot morph ON
export KIT_FACE_ENHANCE=0           # skip plasticky face composite
SHOTS="confr_caught confr_allblades confr_dare confr_dontouch"
echo "[tuned] Kaggle 2xT4 | profile=kaggle_2xt4 | 4-shot confrontation proof"
echo "[tuned] Stage 1/2: pixel-perfect stills"
for s in $SHOTS; do python3 src/render_stills.py --only "$s"; done
echo "[tuned] Stage 2/2: animate stills -> motion"
for s in $SHOTS; do python3 src/render_motion.py --only "$s"; done
echo "[tuned] DONE. On Kaggle: click 'Stop session' to free your weekly GPU quota."
