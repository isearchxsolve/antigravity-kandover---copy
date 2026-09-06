# Maxing out local Wan2.2 realism — documented pathways

Honest ceiling: your reference clip was made by **OiiOii**, a hosted service that
runs frontier closed models (Sora 2 / Seed Dance 2.0 / DreamActor-M1). A quantized
Wan2.2 on 2xT4 cannot equal that. The steps below are the best-documented ways to
narrow the gap on local hardware. They are ordered by impact.

## 1. Model + precision
- Use the **14B** Wan2.2 variants (S2V / FLF2V / Phantom), not the 5B TI2V. The 5B
  (Wan2.2-TI2V-5B) runs on a single 4090 and is the fast/light option; the 14B
  models are the quality tier. See HF: Wan-AI/Wan2.2-TI2V-5B and the Wan2.2 blog.
- On 2xT4 (16GB each) you must quantize. Prefer the **highest GGUF quant that fits**
  (Q8 > Q6 > Q5). city96 GGUF builds are the standard. Q8 is already set.
- Native ComfyUI Wan2.2 workflow (source of truth for graph structure):
  https://docs.comfy.org/tutorials/video/wan/wan2_2

## 2. Sampling settings (quality-first, not speed-first)
- Community consensus for MAX quality: **do NOT rely on 4-step distill LoRAs**
  (Lightx2v / CausVid / FusionX). They cut render time massively but visibly
  "deteriorate motion, clarity and overall quality." Keep them OFF for hero shots.
  (r/comfyui: "My Wan2.2 generation settings").
- Use the Wan2.2 **two-expert (high-noise + low-noise) split**: high-noise model for
  the first portion of steps, low-noise for the rest. This is the MoE design of
  Wan2.2 and is what the native workflow does.
- Steps 30-40, CFG ~5-6, sampler uni_pc / scheduler beta are reasonable quality
  settings (kit `kaggle_2xt4_max` now uses steps 40).
- Lightx2v distill best-practice comparisons (if you DO need speed):
  https://www.reddit.com/r/StableDiffusion/comments/1nc8hcu/ and
  https://huggingface.co/lightx2v/Wan2.2-Distill-Loras

## 3. Prompting
- Natural-language, structured prompts outperform tag soup on Wan2.2. Many users
  auto-expand prompts with an LLM into a fixed template (subject / action / camera /
  lighting / lens). The kit now appends a global realism token block
  (`realism.positive` / `realism.negative` in config).

## 4. Character consistency (your kunal_young / gunja targets)
- **Phantom** (reference-image conditioned) holds identity better than plain I2V, but
  degrades when the face is small/far — keep faces reasonably framed.
- For stubborn drift: generate a strong first frame with a character LoRA / reference,
  then drive I2V/FLF2V from it. (r/comfyui character-consistency threads.)

## 5. Temporal consistency + finishing (biggest "looks pro" lever locally)
- Do a **cleanup/fix pass, THEN upscale to 1080p, THEN optionally to 4K** — staged,
  not one giant jump. (comfy.org upscaling handbook.)
- Best-quality video upscalers (documented): **SeedVR2** and **Topaz Astra Creative**
  for cinematic AI-video fix; FlashVSR for speed. Real-ESRGAN is the free fallback
  the kit already uses. https://docs.comfy.org/tutorials/utility/video-upscale
- Frame interpolation (RIFE) for smoother motion; kit falls back to motion-compensated
  ffmpeg `minterpolate` when RIFE isn't present.
- **Color-grade + grain match** to the reference look: `src/compute_grade.py` builds
  `refs/grade_target.json` from any reference clip; `src/grade_match.py` applies a
  temporally-stable LAB transfer + subtle grain as the final postprocess step.

## 6. Reality check
Even with all of the above, expect "high-end stylized cinematic," not Sora-2
live-action realism. The remaining gap is model weights + training data + datacenter
compute, which no local setting or library closes. For true parity, generate the
hero shots through a frontier hosted API and keep this pipeline for everything else.

### Sources
- ComfyUI Wan2.2 native workflow: https://docs.comfy.org/tutorials/video/wan/wan2_2
- Wan2.2 model card: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- lightx2v distill LoRAs: https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- Settings/quality thread: r/comfyui "My Wan2.2 generation settings"
- Lightx2v A/B tests: r/StableDiffusion "Testing Wan2.2 Best Practices for I2V Part 2"
- Upscaling: https://docs.comfy.org/tutorials/utility/video-upscale and https://blog.comfy.org/p/upscaling-in-comfyui
