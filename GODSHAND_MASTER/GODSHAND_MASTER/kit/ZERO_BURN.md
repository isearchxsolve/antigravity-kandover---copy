# ZERO-POCKET-BURN PLAYBOOK

The rule: **a metered GPU should only ever run diffusion generation (+ CodeFormer + RIFE).
Everything else is free.** Every rupee you spend fixing code, installing nodes, waiting on
downloads, stitching, or idling is waste. This kit now enforces that.

---

## The two-box model

| Box | Cost | Does |
|-----|------|------|
| **Free box** (Kaggle 2xT4 / Colab / your laptop) | 0 | debug, install, validate, pre-stage models, **assemble** (stitch/audio/master/crop) |
| **Paid box** (E2E spot A100 / Jarvislabs) | metered | ONLY: diffusion generation + CodeFormer + RIFE |

Never debug on the paid box. Never stitch on the paid box.

---

## Debug loop = ZERO cost (do this on free hardware first)

1. Bring up ComfyUI + all custom nodes on **Kaggle 2xT4 / Colab** (free). Shake out every
   install failure, path mismatch, and OOM here where the meter reads 0.
2. Render the 10-second **`confrontation`** cut end-to-end -- it exercises the whole pipeline
   (graph -> ComfyClient -> pick_best -> postprocess -> stitch) for free.
3. Only when a full shot completes clean do you touch a paid instance.

```bash
# FREE box: full functional debug
KIT_CONFIG=config.yaml python3 src/preflight.py         # fails in ~5s if anything's missing
python3 src/render_film.py --only confrontation_shot     # one free smoke shot on T4
```

> Caveat: T4 can't do bf16/fp8 like the A100, so a *few* errors are hardware-specific.
> Budget exactly **one** cheap validation shot on the paid box (next section) before the full run.

---

## Paid box = 4 guardrails, all built in

### 1. `preflight.py` -- fail fast for ~0 rupees
Run it the instant the paid instance boots. Missing model / node / disk / GPU => exit non-zero
in seconds, before a single denoise step is billed.
```bash
KIT_REQUIRE_GPU=1 python3 src/preflight.py && \
python3 src/render_film.py --smoke        # 2. cheap validation: 1 shot, 1 take, low steps, no upscale
```
The `&&` guarantees the render never starts if preflight fails.

### 2. `--smoke` -- one cheap real-hardware check
Validates the *paid* environment (bf16/fp8 paths, VRAM, nodes) with 1 shot at low steps and no
upscale. Costs a few rupees, not a few hundred. Passes => launch the real run.

### 3. Resume / skip-completed -- preemption & crash insurance
`render_film.py` writes `work/<shot>/final.mp4` per shot and **auto-skips** any shot already
done. A spot preemption or crash costs you only the *one* in-flight shot; just re-run the same
command and it continues.
```bash
python3 src/render_film.py --max-hours 6 --shutdown
```

### 4. `--shutdown` + `--max-hours` -- kill the idle meter
- `--shutdown` stops the instance the instant the render loop ends (a finished render idling on
  a live GPU is the #1 pocket-burner). Set `KIT_SHUTDOWN_CMD` to your provider's stop/terminate
  API call for spot instances; otherwise it does a best-effort OS poweroff.
- `--max-hours N` refuses to START a new shot past your wall-clock budget (resumable).

---

## Assemble OFFLINE on CPU = free (stitch never touches the GPU)

Download the per-shot `final.mp4` clips from the paid box, then on your laptop:
```bash
bash src/assemble.sh              # POST_MODE=cpu, pure ffmpeg: concat + score + mux + master + zip
```
`stitch.py` is 100% ffmpeg (no GPU). `postprocess.sh` honors `POST_MODE=cpu` to force the
ffmpeg-only path if you ever finish a clip locally (lower quality: no AI face-restore/RIFE).

Re-stitch as many times as you like (music tweaks, cut order, captions, vertical crops) --
all free. Zero GPU seconds spent on encoding.

---

## End-to-end money-safe recipe

```bash
# --- FREE box: debug until confrontation renders clean ---
python3 src/preflight.py && python3 src/render_film.py --only confrontation_shot

# --- Pre-stage models to a volume (cheap CPU box), NOT on the A100 ---

# --- PAID box (E2E spot A100), models already mounted ---
KIT_REQUIRE_GPU=1 python3 src/preflight.py && \
  python3 src/render_film.py --smoke && \
  python3 src/render_film.py --max-hours 8 --shutdown     # generates + CodeFormer + RIFE, then self-stops

# --- download work/<shot>/final.mp4 clips, then FREE laptop assembly ---
bash src/assemble.sh
```

Result: the paid meter runs for generation only. On E2E spot (~Rs.70/hr) the whole ~8-min film
stays inside the Rs.2,000 credit, and debugging/stitching/re-runs cost nothing.

---

## Optimised CPU debug cycle (run these, in order, all FREE)

Layer the checks fastest-first so most edits are validated in ~1 second:

```bash
python3 src/selftest.py --quick   # ~1s : compiles + graph/_bind + config/shots + deps
python3 src/selftest.py --fast    # ~seconds : full mock pipeline, skips slow interpolation
python3 src/selftest.py           # full mock pipeline + cpu post + clips + stitch + qc gate
python3 src/qc.py                 # objective quality report on rendered clips (PASS/WARN/FAIL)
```

- `--quick` is your inner loop -- run after every code edit; it never touches ffmpeg.
- `KIT_FAST_POST=1` (what `--fast` sets) skips interpolation/grade in postprocess for a fast
  debug render; drop it for final quality.
- `COMFY_MOCK=1` makes the client synthesize clips with ffmpeg so the whole orchestration runs
  with no GPU and no ComfyUI.
- `qc.py` is the publish gate: black/freeze detection, resolution/fps/duration, loudness, and
  sharpness/flicker per shot. Exit code is non-zero on any FAIL so you can block a bad upload.

Only three things genuinely need real hardware and cannot be caught here: ComfyUI model loading,
GPU-specific fp8/bf16/VRAM behaviour, and the optional CodeFormer/RIFE/c2patool binaries. Debug
the first two on a FREE GPU (Kaggle/Colab) and confirm with one `--smoke` shot on the paid box.

