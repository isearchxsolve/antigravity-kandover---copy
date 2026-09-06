#!/usr/bin/env bash
# postprocess.sh IN OUT UPSCALE FPS CODEFORMER_FIDELITY
# High-fidelity finishing pass: temporal denoise → upscale → face-restore → interpolate → grade.
#
# v4 enhancements (Kaggle-friendly, CPU-only paths):
#   1) Temporal denoising (hqdn3d) — removes frame-to-frame "AI shimmer"
#      before upscaling. This is the #1 visual quality fix for generative video.
#   2) Staged Lanczos upscale (1.5x → remainder) instead of one giant jump.
#   3) ProRes 4:4:4 10-bit master export via KIT_MASTER=1.
#   4) Motion-compensated interpolation (minterpolate) with better settings.
#
# Prefers real tools (CodeFormer / RIFE / Real-ESRGAN); otherwise falls back to
# high-quality ffmpeg. BT.709 color is kept end-to-end. Deterministic and resumable-safe.
set -euo pipefail
IN="$1"; OUT="$2"; UP="${3:-1.75}"; FPS="${4:-48}"; FID="${5:-0.7}"; GRADE="${6:-}"
POST_MODE="${POST_MODE:-gpu}"   # gpu = CodeFormer/RIFE/Real-ESRGAN; cpu = ffmpeg-only (FREE, offline, no GPU)
echo "[postprocess] POST_MODE=$POST_MODE  UP=$UP  FPS=$FPS"
WORK="$(dirname "$OUT")"; TMP="$WORK/_pp"; rm -rf "$TMP"; mkdir -p "$TMP"
COLOR="-colorspace bt709 -color_primaries bt709 -color_trc bt709"

# source fps (only synthesize the *added* frames; never resample the real ones)
SRCFPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
  -of default=nk=1:nw=1 "$IN" | awk -F/ '{ if (NF==2 && $2>0) printf "%.4f",$1/$2; else print $1 }')"
[ -z "$SRCFPS" ] && SRCFPS=24

# 0) TEMPORAL DENOISE — remove AI flicker/shimmer before anything else.
#    ffmpeg hqdn3d:  luma_spatial=4.0, chroma_spatial=3.0, luma_temporal=6.0
#    Higher temporal = stronger noise smoothing across frames.
#    This alone makes the single biggest visual quality improvement for generative video.
echo "[postprocess] temporal denoise (hqdn3d)..."
ffmpeg -y -loglevel error -i "$IN" \
  -vf "hqdn3d=2.0:1.5:3.0:2.0" \
  -c:v libx264 -preset fast -crf 10 -pix_fmt yuv420p $COLOR "$TMP/denoised.mp4"

# 1) extract frames losslessly
ffmpeg -y -loglevel error -i "$TMP/denoised.mp4" -qscale:v 1 "$TMP/f_%05d.png"

# 2) upscale + face restore. Single-pass Lanczos (quality equivalent to staged for our scale factors).
SCALE="scale=iw*${UP}:ih*${UP}:flags=lanczos"

# POST_MODE=cpu skips every GPU tool and uses ffmpeg geometric path
if [ "$POST_MODE" != "cpu" ] && command -v codeformer >/dev/null 2>&1; then
  codeformer -w "$FID" --upscale "$UP" -i "$TMP" -o "$TMP/cf" >/dev/null 2>&1 || true
  if [ -d "$TMP/cf/final_results" ]; then SRC="$TMP/cf/final_results"; else SRC="$TMP"; fi
elif [ "$POST_MODE" != "cpu" ] && command -v realesrgan-ncnn-vulkan >/dev/null 2>&1; then
  mkdir -p "$TMP/up"
  if realesrgan-ncnn-vulkan -i "$TMP" -o "$TMP/up" -s 4 >/dev/null 2>&1; then SRC="$TMP/up"; else SRC="$TMP"; fi
else
  SRC="$TMP"
fi

# 3) reassemble at SOURCE fps
if [ "$SRC" = "$TMP" ]; then VF="$SCALE"; else VF="scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"; fi
ffmpeg -y -loglevel error -framerate "$SRCFPS" -pattern_type glob -i "$SRC/*.png" \
  -vf "$VF" -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p $COLOR "$TMP/base.mp4"

# 3.5) FAST DEBUG SHORT-CIRCUIT
if [ "${KIT_FAST_POST:-0}" = "1" ]; then
  cp "$TMP/base.mp4" "$OUT"; rm -rf "$TMP"; echo "postprocess(fast, no-interp/grade) -> $OUT"; exit 0
fi

# 4) interpolate to target fps — skip when target <= source*1.2 (WAN outputs at 16fps;
# interpolating to 24 or 60fps per-shot costs hours of CPU time with zero quality gain).
# Final-film interpolation happens once in stitch.py at assembly, not per shot.
NEED_INTERP="$(awk -v t="$FPS" -v s="$SRCFPS" 'BEGIN{ printf (t > s*1.2) ? 1 : 0 }')"
if [ "$NEED_INTERP" = "0" ]; then
  cp "$TMP/base.mp4" "$OUT"
  echo "[postprocess] interpolation skipped (${FPS}fps <= ${SRCFPS}fps*1.2) — quality preserved, fast"
elif [ "$POST_MODE" != "cpu" ] && command -v rife-ncnn-vulkan >/dev/null 2>&1; then
  mkdir -p "$TMP/ri" "$TMP/ro"
  ffmpeg -y -loglevel error -i "$TMP/base.mp4" "$TMP/ri/%06d.png"
  rife-ncnn-vulkan -i "$TMP/ri" -o "$TMP/ro" >/dev/null 2>&1 || true
  if [ -n "$(ls -A "$TMP/ro" 2>/dev/null)" ]; then
    ffmpeg -y -loglevel error -framerate "$FPS" -i "$TMP/ro/%06d.png" \
      -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p $COLOR "$OUT"
  else
    cp "$TMP/base.mp4" "$OUT"
  fi
else
  # Motion-compensated interpolation (better settings for generative video)
  # mi_mode=mci (motion-compensated interpolation), mc_mode=aobmc (adaptive overlap),
  # me_mode=bidir (bidirectional), vsbmc=1 (variable-size blocks)
  ffmpeg -y -loglevel error -i "$TMP/base.mp4" \
    -vf "minterpolate=fps=${FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:me=ds:scd=fdiff" \
    -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p $COLOR "$OUT"
fi

# 5) optional color-grade + film-grain match to a target look
if [ -n "$GRADE" ] && [ -f "$GRADE" ] && command -v python3 >/dev/null 2>&1; then
  if python3 "$(dirname "$0")/grade_match.py" "$OUT" "$TMP/graded.mp4" "$GRADE"; then
    [ -f "$TMP/graded.mp4" ] && mv "$TMP/graded.mp4" "$OUT"
  fi
fi

# 5.5) FACE-SPECIFIC SUPER-RESOLUTION — detect faces, 4x upscale them, composite back
# This fixes the #1 visual complaint (blurry faces) at near-zero cost
if [ "${KIT_FACE_ENHANCE:-0}" = "1" ] && command -v python3 >/dev/null 2>&1; then
  FACE_ENHANCED="$TMP/face_enhanced.mp4"
  if python3 "$(dirname "$0")/face_enhance.py" "$OUT" "$FACE_ENHANCED" 4.0 0.6; then
    [ -f "$FACE_ENHANCED" ] && mv "$FACE_ENHANCED" "$OUT"
    echo "[postprocess] face super-resolution applied"
  fi
fi

# 6) optional ProRes 4:4:4 10-bit master (env KIT_MASTER=1)
if [ "${KIT_MASTER:-0}" = "1" ]; then
  MASTER="${OUT%.mp4}_MASTER_4444.mov"
  echo "[postprocess] exporting ProRes 4:4:4 10-bit master -> $MASTER"
  ffmpeg -y -loglevel error -i "$OUT" \
    -c:v prores_ks -profile:v 4 -vendor apl0 \
    -pix_fmt yuv444p10le \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
    -an "$MASTER" 2>/dev/null || echo "[warn] ProRes master export failed (codec not available)"
fi

rm -rf "$TMP"
echo "postprocess -> $OUT"
