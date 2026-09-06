# EXPERIMENTAL: True >8-bit architecture (EXR capture -> 10-bit 4:4:4 master)

The delivery chain is 8-bit / 4:2:0 end to end. No mux flag fixes that, because
the frames are already 8-bit by the time they reach `grade_match`/`stitch`. To
get genuine >8-bit you must change the ARCHITECTURE at the source: capture the
Wan VAE output as scene-linear 16-bit-float EXR, then master with a
color-managed 10-bit encode.

**Status:** the encode + color management are validated in-repo (`exr_master.py`).
The ComfyUI Save-EXR capture is NOT testable outside the GPU session; treat the
graph change below as untested until you run it on Kaggle.

## 1. Capture EXR from the VAE (ComfyUI graph change)

Every render graph ends in a `VAEDecode` (see `src/make_workflows.py`: the s2v,
phantom, flf2v, and t2i graphs all list `VAEDecode`). ComfyUI's stock
`SaveImage` writes 8-bit PNG and destroys the float precision immediately.

Add an EXR-capable save node fed by the SAME `VAEDecode` image output:
- Install a custom node that writes 32-bit float EXR, e.g. `ComfyUI-saveimage-plus`
  or the WAS Node Suite `Save Image (EXR)`.
- Wire `VAEDecode.IMAGE -> SaveEXR.images`, `format = EXR`, `bit_depth = 32f`,
  no color conversion (keep it scene-linear).
- Output naming: `hero_%05d.exr` into a per-shot dir.

Keep the existing 8-bit path too (it still feeds the fast H.264 preview/culling).
EXR roughly 10-20x the disk of PNG, so only enable it for hero shots.

## 2. Master with color management (validated)

```bash
python3 src/exr_master.py /path/to/exr_dir out/hero_MASTER.mov 24 out/master_48k.wav
```

What it does (and why):
- EXR is scene-LINEAR. It applies a linear->Rec.709 transfer via `zscale`
  (`t=linear` then `transfer=bt709`). Skipping this makes the master dark/flat.
- Encodes ProRes 4444 (`-profile:v 4 -pix_fmt yuv444p10le`), NCLC 1-1-1 tags,
  optional 48 kHz PCM 24-bit audio.

### Hard ceiling (unchanged)
- `prores_ks` real precision caps at **10-bit** (accepts only yuv444p10le). The
  file probes as `yuv444p12le` but that is 10-bit data in a 12-bit container.
  True 12-bit needs Resolve Studio / Apple Compressor.
- 4:4:4 + 10-bit removes chroma subsampling and banding, and preserves whatever
  the VAE produced -- but it cannot add detail the 14B model never generated.

## 3. Native ~44.1 kHz voice (Fish Speech) -- code hook

`src/voice_pass.py` now honors `voice_backend` in config:
- `kokoro` (default): 24 kHz, as before.
- `fish`: routes English lines to `synth_english_fish()`, which is a hook that
  currently falls back to Kokoro until you wire Fish Speech inference in the GPU
  session (needs the `fish-speech` package + weights). Fish Speech's Firefly-GAN
  vocoder is ~44.1 kHz, lifting the voice Nyquist from 12 kHz toward ~22 kHz.
- NOTE: CosyVoice2 is 24 kHz, so it would NOT remove upsampling; Fish Speech is
  the right pick for genuine high-frequency voice detail.
