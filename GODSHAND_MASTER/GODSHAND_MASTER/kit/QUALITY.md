# Quality — v4 Blueprint-First Pipeline (near-100% pixel-perfect on Kaggle)

## Architecture overview

```
preflight → cull → de-age
  → render_stills (pixel-perfect still per beat, 5 rejection samples, 40 steps)
  → render_motion (animate each still with low-denoise, next-still as end-frame)
  → identity_check (CPU insightface, auto-retry low-scoring shots)
  → voice_pass (48kHz cinematic TTS with convolution reverb)
  → score_gen (MusicGen → multiband mastering → 48kHz)
  → sfx_gen (procedural ambience/foley)
  → stitch (stem-based mix: multiband ducking, reverb bus, ProRes 4:4:4)
```

Key innovation: **The still carries 99% of the detail; motion is infrastructure (1%).**
On T4, still image generation is fast and high-quality. Animating from a perfect still
with low denoise (0.35) preserves that quality while adding temporal coherence.

## v4 quality gates

### 1. Technical QC — `src/qc.py` (unchanged)
Objective per-shot checks: resolution, fps, duration, black-frame, freeze, loudness,
sharpness, flicker.

### 2. Per-shot color grading — `src/grade_match.py` (v4 upgrade)
- **Per-channel color curves** (lift/gamma/gain) instead of global LAB transfer
- **Temporally-coherent film grain** — grain flows naturally across frames (numpy-only,
  no scipy needed)
- **Halation bloom** — subtle warm glow on bright highlights
- **ProRes 4:4:4 master** export option (env `KIT_MASTER=1`)

### 3. Identity consistency — `src/identity_check.py`
- ArcFace-based face-matching against character reference images
- **Auto-retry** — shots below `identity.drift_cosine` threshold get re-rendered
  with new seeds up to `identity_retry` times
- Runs on CPU (insightface + onnxruntime), free

### 4. Cinematic voice processing — `src/voice_pass.py` (v4 upgrade)
- **48kHz output** pipeline (upsampled from native TTS sample rate)
- **Character-specific EQ chains** (warmth for Gunja, chest resonance for Kunal,
  metallic distance for Observer)
- **Convolution reverb** — synthetic hall IR (no external files, numpy-only)
- Multiband compression and de-essing

### 5. Professional final mix — `src/stitch.py` (v4 rewrite)
- **5-stem architecture**: music, ambience, voice, chorus, reverb-return
- **Multiband sidechain ducking** — only ducks 200–4000 Hz where voice lives;
  preserves bass weight and high-frequency air in the music
- **Shared convolution reverb bus** — realistic room acoustics
- **True-peak limiting** @ -0.5 dBTP — streaming-safe, no intersample peaks
- **48 kHz / 24-bit audio** throughout
- **ProRes 4:4:4 10-bit master** export (env `KIT_MASTER=1`)

### 6. Post-processing pipeline — `src/postprocess.sh` (v4 upgrade)
- **Temporal denoising** (hqdn3d) — removes AI "shimmer" before upscaling.
  This is the #1 visual quality fix for generative video.
- Staged Lanczos upscale
- Motion-compensated interpolation with better settings

## Quality settings (tunable in config.yaml)

| Setting | Kaggle value | Effect |
|---------|-------------|--------|
| `steps` | 30 | T4 sweet spot (40 eats 33% more time) |
| `cfg` | 6.0 | Stronger prompt adherence |
| `denoise` | 0.35 | Faithful to blueprint stills |
| `codeformer_fidelity` | 0.8 | Better face restoral |
| `rejection_samples` | 3 | Best of N take selection |
| `identity_retry` | 2 | Auto-retry on low face-match |

## Where quality costs GPU time (on Kaggle 2x T4)

| Stage | Est. time | Free? |
|-------|-----------|-------|
| Preflight + cull + de-age | ~5 min | ✅ CPU |
| Render stills (40 shots × 3 takes) | ~2 h | GPU |
| Animate stills → motion | ~3 h | GPU |
| Voice (TTS on CPU) | ~10 min | ✅ CPU |
| Score (MusicGen on GPU) | ~20 min | GPU |
| Stitch + master | ~5 min | ✅ CPU |
| **Total** | **~6-8 h** | **$0** |

## Honest limits
- The kit targets "high-end stylized cinematic," not Sora-2 live-action realism
- Temporal coherence between shots is limited (each shot animates independently)
- No multi-pass refinement (each shot gets one render cycle with retry)
- Face identity is checked but not perfectly enforced across all angles
