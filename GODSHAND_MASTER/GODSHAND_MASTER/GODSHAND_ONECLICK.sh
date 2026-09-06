#!/usr/bin/env bash
###############################################################################
# GODSHAND -- ONE-CLICK end-to-end runner  (Lightning AI L4 / any CUDA box)
#
#   unzip -> repoint paths -> RE-ASSERT every fix+tuning (idempotent) ->
#   VERIFY (no-compromise guard: aborts if ANY fix/hardening is missing) ->
#   install ComfyUI + custom nodes -> download WAN 2.2 models ->
#   launch ComfyUI headless -> render the tuned 4-shot proof scene ->
#   auto-shutdown (to stop billing).
#
# Usage:
#   bash GODSHAND_ONECLICK.sh                 # auto-detect kit / unzip GODSHAND_FINAL.zip
#   bash GODSHAND_ONECLICK.sh some_kit.zip    # explicit archive
#
# Env knobs:
#   GODSHAND_FULL=1          render ALL shots of the active story (KIT_CONFIG); default is 4 proof shots
#   GODSHAND_NO_SHUTDOWN=1   leave the box running at the end (default: shutdown)
#   GODSHAND_BASE=/path      where models/work live (default: Lightning studio dir or <kit>/_run)
#   GODSHAND_ANCHORS=/path   folder of character reference images to ingest
#   HF_TOKEN=hf_xxx          only if a model repo needs auth
###############################################################################
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
log(){ printf '\n\033[1;36m[oneclick]\033[0m %s\n' "$*"; }

# --------------------------------------------------------------- 0) locate kit
RUNPY="$(find . -maxdepth 5 -name run.py 2>/dev/null | grep -v -e node_modules -e ComfyUI | head -1 || true)"
if [ -z "$RUNPY" ]; then
  ZIP="${1:-}"
  [ -z "$ZIP" ] && ZIP="$(ls -1t GODSHAND_FINAL.zip GODSHAND*.zip 2>/dev/null | head -1 || true)"
  [ -z "$ZIP" ] && { echo "[FATAL] no run.py and no GODSHAND*.zip found in $HERE"; exit 1; }
  log "unzipping $ZIP"
  rm -rf _kit && mkdir -p _kit && unzip -q "$ZIP" -d _kit
  RUNPY="$(find _kit -maxdepth 6 -name run.py | grep -v ComfyUI | head -1 || true)"
fi
[ -z "$RUNPY" ] && { echo "[FATAL] run.py not found after unzip"; exit 1; }
KIT="$(cd "$(dirname "$RUNPY")" && pwd)"; cd "$KIT"
log "kit root: $KIT"

# ------------------------------------------------- 1) choose a writable base
if [ -n "${GODSHAND_BASE:-}" ]; then BASE="$GODSHAND_BASE"
elif [ -d /teamspace/studios/this_studio ]; then BASE=/teamspace/studios/this_studio/godshand
else BASE="$KIT/_run"; fi
mkdir -p "$BASE"
export COMFYUI="$BASE/ComfyUI"
log "work base: $BASE   ComfyUI: $COMFYUI"

# ---------- 2+3) repoint Kaggle paths + RE-ASSERT all fixes/tuning (idempotent)
log "repointing paths + re-asserting every fix and tuning value..."
python3 - "$KIT" "$BASE" "$COMFYUI" <<'PYFIX'
import sys, re, os
kit, base, comfy = sys.argv[1], sys.argv[2], sys.argv[3]
cfgp = os.path.join(kit, "config_confrontation.yaml"); s = open(cfgp).read()
s = s.replace("/kaggle/working/confrontation_work", base + "/confrontation_work")
s = s.replace("/kaggle/temp/models", comfy + "/models").replace("/kaggle/working/models", comfy + "/models")
s = re.sub(r'^active_profile:.*$', 'active_profile: lightning_l4', s, flags=re.M)
s = re.sub(r'^codeformer_fidelity:.*$', 'codeformer_fidelity: 0.5', s, flags=re.M)
s = re.sub(r'^denoise:.*$', 'denoise: 0.55', s, flags=re.M)
s = re.sub(r'^interpolate_fps:\s*(\d+)', lambda m: 'interpolate_fps: %d' % min(16, int(m.group(1))), s, flags=re.M)
open(cfgp, 'w').write(s)
ppp = os.path.join(kit, "src", "postprocess.sh"); p = open(ppp).read()
p = p.replace("hqdn3d=4.0:3.0:6.0:3.0", "hqdn3d=2.0:1.5:3.0:2.0")
if "KIT_FACE_ENHANCE" not in p:
    p = p.replace(
        '# This fixes the #1 visual complaint (blurry faces) at near-zero cost\nif command -v python3 >/dev/null 2>&1; then',
        '# This fixes the #1 visual complaint (blurry faces) at near-zero cost\nif [ "${KIT_FACE_ENHANCE:-0}" = "1" ] && command -v python3 >/dev/null 2>&1; then')
open(ppp, 'w').write(p)
print("   paths repointed + fixes/tuning re-asserted")
PYFIX

# ----------------------------- 4) NO-COMPROMISE VERIFY (abort if any missing)
log "verifying ALL fixes + hardening + tuning are intact..."
CFG=config_confrontation.yaml; fail=0
chk(){ if eval "$2" >/dev/null 2>&1; then printf '   \033[32m OK \033[0m %s\n' "$1"; else printf '   \033[31mMISS\033[0m %s\n' "$1"; fail=1; fi; }
chk "FIX: NEED_INTERP interpolation guard"    "grep -q NEED_INTERP src/postprocess.sh"
chk "FIX: vectorized face_enhance mask"        "grep -q VECTORIZED src/face_enhance.py"
chk "HARDEN: interpolate_fps <= 16"            "awk '/^interpolate_fps:/{exit !(\$2+0<=16)}' $CFG"
chk "TUNE: L4 profile active"                  "grep -q '^active_profile: lightning_l4' $CFG"
chk "TUNE: denoise 0.55"                       "grep -q '^denoise: 0.55' $CFG"
chk "TUNE: codeformer_fidelity 0.5"            "grep -q '^codeformer_fidelity: 0.5' $CFG"
chk "TUNE: softened hqdn3d"                    "grep -q 'hqdn3d=2.0:1.5:3.0:2.0' src/postprocess.sh"
chk "TUNE: face-composite gated OFF by default" "grep -q KIT_FACE_ENHANCE src/postprocess.sh"
if bash -n src/postprocess.sh 2>/dev/null; then printf '   \033[32m OK \033[0m %s\n' "postprocess.sh bash syntax"; else printf '   \033[31mMISS\033[0m %s\n' "postprocess.sh syntax"; fail=1; fi
[ "$fail" = 1 ] && { echo; echo "[FATAL] NO-COMPROMISE CHECK FAILED -- refusing to run."; exit 2; }
log "all fixes + hardening + tuning verified."

# ----------------------------------------- 5) install ComfyUI + custom nodes
if [ ! -f "$COMFYUI/main.py" ]; then
  log "installing ComfyUI + custom nodes (first run only)"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI"
  pip install -q -r "$COMFYUI/requirements.txt"
  CN="$COMFYUI/custom_nodes"; mkdir -p "$CN"
  for repo in \
    https://github.com/city96/ComfyUI-GGUF.git \
    https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    https://github.com/cubiq/ComfyUI_InstantID.git \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git ; do
    ( cd "$CN" && git clone --depth 1 "$repo" ) || echo "[warn] node clone failed: $repo"
  done
  pip install -q gguf sentencepiece protobuf einops hf_transfer "huggingface_hub>=0.23"
else
  log "ComfyUI already present at $COMFYUI"
fi

# ------------------------------------------------------- 6) download WAN models
log "downloading WAN 2.2 models (first run only; tens of GB over the network)"
COMFYUI="$COMFYUI" HF_TOKEN="${HF_TOKEN:-}" python3 "$KIT/l4_model_setup.py"
# shellcheck disable=SC1091
source "$KIT/wan_env.sh"

# ------------------------- 7) OPTIONAL character reference images (NOT required)
# forge_character.py auto-GENERATES a consistent face/body set for each target
# (the active story's cast, from KIT_CONFIG) during the GPU session -- no photos needed.
# To PIN specific real faces instead: drop images into kit/refs/<name>/face (and
# /body) BEFORE running, or point GODSHAND_ANCHORS at a folder laid out as
# <name>/face/*.jpg -- it will be copied into kit/refs/ and forge will reuse it.
mkdir -p "$COMFYUI/input"
# Story is selectable: KIT_CONFIG picks which film to build (default: confrontation).
export KIT_CONFIG="${KIT_CONFIG:-config_confrontation.yaml}"
# Derive THIS story's characters from its config (fallback: jack julie).
TARGETS="$(cd "$KIT" && python3 -c 'import sys;sys.path.insert(0,"src");from cfgutil import load_cfg;c=load_cfg(".");print(" ".join((c.get("character_forge") or {}).get("targets",[]) or []))' 2>/dev/null)"
[ -z "$TARGETS" ] && TARGETS="jack julie"
log "story: $KIT_CONFIG | characters: $TARGETS"
if [ -n "${GODSHAND_ANCHORS:-}" ] && [ -d "$GODSHAND_ANCHORS" ]; then
  cp -R "$GODSHAND_ANCHORS"/* "$KIT/refs/" 2>/dev/null || true
  log "copied your reference images from $GODSHAND_ANCHORS into kit/refs/"
fi
for t in $TARGETS; do
  if ls "$KIT"/refs/"$t"/face/*.jpg "$KIT"/refs/"$t"/face/*.jpeg "$KIT"/refs/"$t"/face/*.png "$KIT"/refs/"$t"/face/*.webp >/dev/null 2>&1; then
    log "identity '$t': using your photos in kit/refs/$t/face"
  else
    log "identity '$t': no photos in kit/refs/$t/face -> will be AUTO-FORGED this session (consistent invented face)"
  fi
done

# --------------------------------------------------- 8) launch ComfyUI headless
if ! curl -s http://127.0.0.1:8188/ >/dev/null 2>&1; then
  log "launching ComfyUI headless on 127.0.0.1:8188"
  ( cd "$COMFYUI" && nohup python3 main.py --listen 127.0.0.1 --port 8188 --cache-none \
      > "$BASE/comfyui.log" 2>&1 & echo $! > "$BASE/comfyui.pid" )
  for i in $(seq 1 90); do curl -s http://127.0.0.1:8188/ >/dev/null 2>&1 && break; sleep 3; done
fi
if ! curl -s http://127.0.0.1:8188/ >/dev/null 2>&1; then
  echo "[FATAL] ComfyUI did not come up. Last 40 log lines:"; tail -n 40 "$BASE/comfyui.log" 2>/dev/null || true; exit 5
fi
log "ComfyUI is up."

# --------------------------------------------------------------- 9) render
# KIT_CONFIG was already exported in step 7 (defaults to config_confrontation.yaml).
export KIT_MOTION_DENOISE=0.55
export KIT_FLF2V_ENDFRAME=1
export KIT_FACE_ENHANCE=0
python3 src/make_workflows.py || true
log "forging character identities (idempotent)"
python3 src/forge_character.py || echo "[warn] forge_character reported an issue (continuing)"
if [ "${GODSHAND_FULL:-0}" = "1" ]; then
  log "FULL render: all shots in $KIT_CONFIG"
  python3 src/render_stills.py
  python3 src/render_motion.py
else
  # Proof shots: GODSHAND_SHOTS overrides; else the first 4 shot ids of this story.
  SHOTS="${GODSHAND_SHOTS:-$(cd "$KIT" && python3 -c 'import json,sys;sys.path.insert(0,"src");from cfgutil import load_cfg;c=load_cfg(".");d=json.load(open(c["shots_file"]));print(" ".join(s["id"] for s in d["shots"][:4]))' 2>/dev/null)}"
  [ -z "$SHOTS" ] && SHOTS="confr_caught confr_allblades confr_dare confr_dontouch"
  log "PROOF render: $SHOTS"
  for s in $SHOTS; do log "still  -> $s"; python3 src/render_stills.py --only "$s"; done
  for s in $SHOTS; do log "motion -> $s"; python3 src/render_motion.py --only "$s"; done
fi

# ------------------------------------------------------ 10) report + shutdown
WORK="$(python3 -c 'import sys; sys.path.insert(0,"src"); from cfgutil import load_cfg; print(load_cfg(".")["paths"]["work"])' 2>/dev/null || echo "$BASE/confrontation_work")"
log "DONE. Per-shot outputs at: $WORK/<shot_id>/final.mp4"
sync
if [ "${GODSHAND_NO_SHUTDOWN:-0}" = "1" ]; then
  log "GODSHAND_NO_SHUTDOWN=1 -> leaving the box running. STOP the Studio yourself to avoid idle billing."
else
  log "auto-shutdown to stop billing (set GODSHAND_NO_SHUTDOWN=1 to skip)"
  sudo shutdown -h now || echo "[warn] shutdown failed -- STOP THE STUDIO MANUALLY to avoid charges."
fi
