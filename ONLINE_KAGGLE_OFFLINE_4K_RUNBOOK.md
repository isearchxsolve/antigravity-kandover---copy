# GODSHAND — Operator Runbook: Online Kaggle Render + Offline 4K Upscale

This is the canonical end-to-end procedure, in the order a single operator executes it, from an empty Kaggle session to a delivered 4K master. It consolidates `KAGGLE_PREFLIGHT_CHECKLIST.md`, `KAGGLE_OPERATOR_COMMANDS.md`, `kit/RUN.md`, `kit/KAGGLE_TEST.md`, and the `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb` one-click notebook into one linear playbook. Nothing here is new policy — every step is already proven by `selftest`, `comprehensive_smoke_test.py`, and `edge_case_integration_test.py` (36/36).

The run is split into two disjoint phases on purpose:

| Phase | Where | Hardware | Duration | Cost |
|---|---|---|---|---|
| **A. Online render** | Kaggle Notebook (cloud) | 2× T4 (14.9 GB each) | 2–3 h full film / 9 h weekly quota | Free tier or ~$2–4 paid |
| **B. Offline 4K upscale** | Local workstation OR a second Kaggle session | GPU recommended, CPU works (slow) | ~2–3 min/clip on T4 | Free if local; $0 if second free Kaggle session |

> Why split: the render produces 1280×704 (soon 1280×720) HD frames. Upscaling to 3840×2160 is a separate neural pass (Real-ESRGAN ×4 + GFPGAN + RIFE) that needs no WAN weights — you can run it later, on a different box, or even weeks later against the saved `Beholder_Chunk_*.zip`. Keeping the heavy 4K work off the paid T4 hours is what protects your weekly Kaggle quota.

> **A and B can also be run as one unattended session on a single Kaggle book.** See the Fast Path below — `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb` chains both phases through 8 fail-fast gates on a single Run All.

## Fast Path — Upload → Run All → Download → Ship (6 steps, ~3 hours unattended)

This is the canonical happy-path. Use it when you just want to ship. Detailed procedures for each phase follow in §II and §III for when something fails and you need to debug a specific gate.

### 1. On your machine: build the `godshand` dataset zip

```
godshand/
├── kit/                              ← copy from GODSHAND_MASTER/GODSHAND_MASTER/kit/
│   ├── src/
│   │   ├── render_motion.py          ← Gate 0 marker file (must exist)
│   │   ├── comfy_client.py
│   │   ├── selftest.py
│   │   └── ...
│   ├── workflows/flf2v.json          ← required by Gate 0 + validator
│   ├── config_confrontation.yaml     ← active_profile: kaggle_2xt4_max
│   ├── shots_confrontation.json      ← the shot list
│   └── ...
└── rainman-wan-MERGED-TUNED-FIXED.ipynb   ← the production notebook (~390 KB)
```

Zip the **folder** (not its contents) so the unzip produces `/kaggle/input/<dataset_name>/godshand/kit/...`. Name it `godshand.zip`.

Verify locally before uploading (these must all say PRESENT):
```powershell
$kit = "C:\Users\Admin\Downloads\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit"
Test-Path "$kit\src\render_motion.py"
Test-Path "$kit\config_confrontation.yaml"
Test-Path "$kit\shots_confrontation.json"
Test-Path "$kit\workflows\flf2v.json"
Test-Path "C:\Users\Admin\Downloads\antigravity-kandover\rainman-wan-MERGED-TUNED-FIXED.ipynb"
# Expect: True True True True True
```

### 2. Kaggle → Datasets → New Dataset

- Upload `godshand.zip`. Title it `godshand-kit` (or any name). Visibility: **Private**.
- One-time, ~2 minutes. Reusable across all future sessions.

### 3. New Notebook → configure the instance

- **Accelerator: GPU T4 ×2** (downgraded session will fail Gate 2 — VRAM too small)
- **Internet: ON** (required first run for HF Hub model downloads; can be off on subsequent runs after weights cache)
- **Add Data → search "godshand-kit" → Add.** Confirm the dataset appears under `/kaggle/input/godshand-kit/`.
- *(Optional)* **Add-ons → Secrets → add `HF_TOKEN`** for higher HF rate limits on first run. Gate 0 auto-discovers it under any of these labels: `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `hf_token`.

### 4. File → Import Notebook → upload `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb`

This is the one-click driver. It has 11 cells (1 markdown header + 9 gate cells + 1 markdown footer), implementing the 8-gate flow:

| Gate | Time | What it proves |
|---|---|---|
| **0** | ~30 s | Dataset mounted, kit copied to `/kaggle/working`, HF token loaded, required files present |
| **1** | ~30 s | Static checks clean (compile, deps, graph `_bind`, configs) — `selftest.py --quick` |
| **2** | 2–5 min | Production path `render_motion()` works on **real T4** (`COMFY_MOCK=0`) |
| **3** | 10–15 min first / cached after | Wan2.2 GGUF weights downloaded from HF Hub |
| **4** | 2–3 h | **Full progressive film render** (hands-off, resume-on-rerun, 11 h session cap) |
| **5** | ~10 s | Post-run log scan: no `Invalid PNG`, `Killed`, `CUDA OOM`, runaway repeated messages |
| **6** | ~10 s | Deliverables present; `Beholder_Chunk_1.zip` packaged for the upscale |
| **7** | 5–15 min | 4K upscale: Real-ESRGAN×4 + GFPGAN + RIFE24 + per-scene grade + ProRes 4444 XQ 12-bit master |
| **8** | ~10 s | `ffprobe` confirms 3840×2160, `prores`, `yuv444p10le`, BT.709 — broadcast spec |

If any gate fails, the cell **raises and stops**. Do not skip — the original 79-min silent death and missing `final.mp4` were exactly the class of failure these gates catch.

### 5. Run All → walk away ~3 hours

You can leave the page. Kaggle sessions persist for ~12 h with the accelerator on. Come back when the final markdown cell (`🎬 All 8 Gates Passed`) appears, or when the Output tab shows `hd_out_4k.zip`.

Watch the gate markers as you go (or on return):

```
✅ GATE 0 PASSED — dataset mounted, kit ready, HF token loaded
✅ GATE 1 PASSED — static checks green
✅ GATE 2 PASSED — production path renders on real T4
✅ GATE 3 PASSED — models downloaded / cached
✅ GATE 4 PASSED — full confrontation film completed
✅ GATE 5 PASSED — no FATAL signatures in logs
✅ GATE 6 PASSED — deliverables present, HD chunk packaged
✅ GATE 7 PASSED — 4K upscale complete
✅ GATE 8 PASSED — all masters meet broadcast spec
```

### 6. Download `hd_out_4k.zip` from the Output tab

- Contains `<name>_4k.mp4` (H.264 review cuts, scrubbable) and `<name>_master.mov` (ProRes 4444 XQ 12-bit, broadcast-grade).
- Watch the `_master.mov` files. **This is the only automated gate cannot test — it is the human gate, and it is non-negotiable.** Scrub end-to-end for flicker, identity drift, color drift, audio sync, ESRGAN shimmer, RIFE smear on fast motion.
- If clean: **ship** the ProRes masters to your editor/colorist. The loop is done.

> If a gate failed: stop and skip to §II or §III below for the per-phase debugging procedure. Each gate's failure mode and what to do about it is documented there.

---

## Phase A — Online Kaggle render (produces HD master + per-clip packages)

### A.0 Prerequisites (once per session, ~1 min)

- Kaggle notebook with **Accelerator = GPU T4 ×2** and **Internet = ON** (required for HF Hub model downloads on first run only).
- ≥ 30 GB free on `/kaggle/working` (model weights + work dir + output).
- The kit attached as a Kaggle Dataset (see Fast Path §1–§2). Gate 0 of `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb` auto-discovers the dataset at `/kaggle/input/<name>/godshand/kit/src/render_motion.py` and copies it to the writable `/kaggle/working/godshand/`. No git clone inside the notebook — the dataset is the source of truth.

### A.1 Free static gate (~30 s, no GPU)

Run this before spending a single GPU-second. It catches compile errors, missing deps, broken workflow `_bind` references, and malformed configs/shots — all for free.

```bash
cd /kaggle/working/godshand/kit
python src/selftest.py --quick
```
**Expected:** `[selftest --quick] ALL GREEN (static).`
**If it fails:** do not proceed. The output lists every static issue in a single pass — fix them all before re-running.

### A.2 Free production-path gate (~5 min, mock GPU/CPU)

This exercises `render_motion.py` (the *exact* path the notebook executes) with `COMFY_MOCK=1` + `POST_MODE=cpu`. A regression here means the notebook would have silently produced no `final.mp4` per shot — the original Defect #1.

```bash
python src/selftest.py --motion --fast
```
**Expected:** `[selftest --motion] ALL GREEN (fast)` and `/tmp/kit_selftest_work_motion/m01/final.mp4` produced.

### A.3 Validator dry-run (~1 min, no GPU)

```bash
python src/runtime_integration_validator.py
```
**Expected:** Stage 3 dry-runs show `WORKFLOW VALID` for `flf2v.json` and `charforge.json`. Stage 4 shows `✓ NEED_INTERP guard` and `✓ minterpolate conditional`.
**Errors allowed:** only `MISSING: xformers / comfy / ComfyUI / custom nodes` (environmental on a fresh instance — ComfyUI gets bootstrapped in the notebook itself).

### A.4 First-run model download (~10–15 min, runs once)

Run **Cell 1 of `rainman-wan-MERGED-TUNED-FIXED.ipynb` only** (or use `run.py` which calls preflight first). This pulls `WanTEModel`, `WanVAE`, `WAN21`, and the GGUF weights from HF Hub.

**Verify before moving on:**
- `WanTEModel`, `WanVAE`, `WAN21` report `loaded completely; … MB usable, … MB loaded`.
- The GGUF qtypes line shows `F16 (694), Q4_K (280), Q6_K (120), F32 (1)`.

Models are cached thereafter; subsequent runs skip this step.

### A.5 GPU smoke test (real GPU, real WAN, one shot)

```bash
python gpu_smoke.py
```
**Expected:** `[gpu_smoke] ✅ GREEN — production path renders on real GPU`.
**If RED:** stop. Check VRAM (`nvidia-smi`), model paths, that `COMFY_MOCK` is *not* set in the env. Do not proceed to a full run — the 79-min hang that produced `kaggle_log_fatal.txt` was exactly this class of failure slipping past the gate.

### A.6 Full notebook run (~2–3 h for the whole film)

Run all remaining cells of `rainman-wan-MERGED-TUNED-FIXED.ipynb` in order, OR drive it via the kit's guarded entrypoint:

```bash
# SAFE by default: render + project cost, then STOP.
python run.py
# When satisfied with the projection:
CONFIRM_FULL_RUN=1 MAX_BUDGET_USD=8 AUTO_SHUTDOWN=1 python run.py
```

`run.py` is fail-fast and resumable: finished shots auto-skip on re-run, so a mid-run stop (or a 9-hour Kaggle session boundary) wastes nothing.

**During the run, monitor:**
- Each `Prompt executed in XXX.XX seconds` line should be 300–600 s in steady state (the first prompt is ~15 min because of one-time model load — that's normal).
- If any single prompt exceeds ~1300 s (1.5× the 861 s median the fixtures captured), that's a per-prompt timing regression — abort and inspect.

**Env knobs that matter:**

| Var | Default | Meaning |
|---|---|---|
| `CONFIRM_FULL_RUN=1` | unset | Required to spend GPU past the cost-meter sample |
| `MAX_BUDGET_USD=8` | unset | Hard $ ceiling; aborts if projection exceeds it |
| `MAX_HOURS=6` | 0 | Wall-clock cap for the render loop (resumable) |
| `AUTO_SHUTDOWN=1` | 0 | Stop the instance the moment rendering ends |
| `KIT_REQUIRE_GPU=1` | auto on paid | Makes "no GPU" a fatal preflight error |
| `KIT_CONFIG` | `config_confrontation.yaml` | Which profile set to use |

### A.7 Post-run log scan (~10 s)

After the notebook finishes (or any time you suspect a silent failure):

```bash
python /kaggle/working/scan_logs.py
```
**Exit codes:** `0` = clean, `1` = FATAL signature found (Invalid-PNG / Killed / CUDA-OOM / runaway-repeats), `2` = log missing.

The scanner is *reactive* — it tells you what already happened. If it finds FATAL signatures, do not ship the output; inspect the printed log lines, fix the root cause, and re-run from A.5.

### A.8 Verify deliverables

```bash
ls -la /kaggle/working/output_*/clips/*/
ls -la /kaggle/working/output_*/SELFTEST.mp4 2>/dev/null \
  || ls -la /kaggle/working/output_*/confrontation.mp4 2>/dev/null \
  || ls -la /kaggle/working/output_*/master.mp4
```

Each shot id directory should contain: `master.mp4`, `vertical.mp4`, `thumb.jpg`, `manifest.json`. The top-level final master is `SELFTEST.mp4` (selftest run), `confrontation.mp4` (confrontation profile), or `master.mp4`.

### A.9 Package the HD output for handoff to Phase B

Collect the per-shot `master.mp4` files (the un-upscaled HD clips) into a single zip. The 4K notebook expects a `Beholder_Chunk_*.zip` or a flat directory of `.mp4`s.

```bash
cd /kaggle/working
zip -r Beholder_Chunk_1.zip output_*/clips/*/master.mp4
# (optional) also keep the stitched film as a single mp4
cp output_*/confrontation.mp4 ./   2>/dev/null || true
```

Download `Beholder_Chunk_1.zip` from the Kaggle **Output** tab. This is the artifact Phase B consumes — it is the *only* file you must take out of the cloud session.

### A.10 Human review gate (the "Hollywood-grade" gate)

Before any 4K work:
- Open `master.mp4` (or the stitched confrontation master) and scrub end to end.
- Watch for: flicker, identity drift between shots, color drift, audio sync, black frames, frozen frames.
- If anything fails: log the shot ids, fix config/refs, re-run only those shots via `python src/render_motion.py --only <shot_id>`. Re-stitch. Re-review.

This human gate is non-negotiable — it is the only thing standing between "the conveyor belt ran" and "the conveyor belt produced cinema."

---

## Phase B — Offline 4K upscale (produces 3840×2160 master + ProRes 4444 XQ)

> If you followed the Fast Path above, **Phase B already ran as Gates 7–8** in the same Kaggle session and `hd_out_4k.zip` is sitting on your Output tab. This section is for operators who **only** want to upscale an existing `Beholder_Chunk_1.zip` (from a prior Phase-A run, or handed off by someone else) — decoupled from the render session.

You can run Phase B standalone in two ways. Pick **one**:

- **B-local**: on your own workstation with a CUDA GPU (recommended if you have one). CPU works but a 2-minute clip becomes ~30 min.
- **B-kaggle-2**: fire up a *second* free Kaggle T4 session, attach `Beholder_Chunk_1.zip` as a Dataset, and run the upscaler there. Costs nothing if you have spare weekly quota.

Either way, the tool is the notebook `upscale_hd_kaggle_4k.ipynb` (in the repo root). It is fully self-contained — one markdown setup cell + one code cell — and runs the same 8-stage chain Gate 7 of the gated pipeline uses. The full chain it runs, in order:

```
hqdn3d denoise  ->  GFPGAN face SR (weight 0.7)
                 ->  Real-ESRGAN ×4 (to ~3072×2048)
                 ->  temporal blend (w=0.15, kills ESRGAN per-frame shimmer)
                 ->  RIFE optical-flow interpolation to 24fps
                                          (fallback: ffmpeg minterpolate mi_mode=blend)
                 ->  per-scene cinematic grade (keyed by clip-name prefix c01..c05)
                 ->  4K pad to 3840×2160 with lanczos
                 ->  H.264 review cut (CRF 14, slow) + ProRes 4444 XQ 12-bit master (.mov)
```

### B.1 Setup

1. Make `Beholder_Chunk_1.zip` (from A.9) reachable by the notebook:
   - **B-kaggle-2:** upload the zip as a Kaggle Dataset, then *attach* that dataset to the new notebook. The notebook auto-discovers any `.zip` or `.mp4` under `/kaggle/input/**/*.zip`.
   - **B-local:** put the zip (or a folder of `.mp4`s) anywhere, then set the `INPUT` variable at the top of the code cell to that path.
2. Set the accelerator:
   - **B-kaggle-2:** `Accelerator = GPU T4`.
   - **B-local:** any CUDA GPU works. The notebook auto-detects `torch.cuda.is_available()` and switches to fp32 CPU mode if not.
3. Open `upscale_hd_kaggle_4k.ipynb` and **Run All**.

### B.2 Knobs (defaults are tuned for cinematic output)

All knobs are at the top of the code cell — no need to edit anything else.

| Knob | Default | Effect |
|---|---|---|
| `INPUT` | `""` (auto-pick) | Leave empty on Kaggle to auto-find the attached dataset; set explicitly for B-local |
| `USE_FACE_SR` | `True` | GFPGAN face super-resolution before ESRGAN |
| `FACE_WEIGHT` | `0.7` | GFPGAN blend (0=subtle, 1=full). Lower if faces look over-smoothed |
| `ESRGAN_TILE` | `512` | Patch size. Lower = less VRAM (e.g. 256 for a 4 GB card) |
| `RIFE_TFPS` | `24` | Target fps for RIFE optical-flow interp. `0` = skip |
| `TEMPORAL_BLEND` | `0.15` | Cross-frame blend weight. 0.10–0.20 sweet spot |
| `PRORES_MASTER` | `True` | Also emit a ProRes 4444 XQ 12-bit `.mov` master alongside H.264 |
| `PER_SCENE_GRADE` | `True` | Per-chapter color grade keyed off clip-name prefix |

The grade table (`SCENE_GRADES` in the code cell) is keyed by shot-id prefix. The defaults encode five chapters of *The Saga of the Beholder*:

- `c01_*` — cold concrete-grey (institutional dread)
- `c02_*` — warm amber (golden-hour hope)
- `c03_*` — deep teal-cyan (water/void)
- `c04_*` — bleach-bypass (stark confession)
- `c05_*` — green-amber (hospital lifebeat)

To add a chapter, extend `SCENE_GRADES` with a new `"<prefix>": "<ffmpeg filterchain>"` entry. Clips whose names don't match any prefix fall back to `c01` (cold grey).

### B.3 Output

When the cell finishes (~2–3 min per clip on T4):

- `hd_out_4k/<name>_4k.mp4` — H.264 review cut (scrubbable, small).
- `hd_out_4k/<name>_master.mov` — ProRes 4444 XQ 12-bit master (broadcast-grade, no gradient banding). **This is the master you hand to an editor/colorist.**
- `hd_out_4k.zip` — bundle of everything above, on the Kaggle **Output** tab (or at `/kaggle/working/` for B-kaggle-2).

### B.4 The known compatibility fix (already built in)

If you ever rip the upscale out of the notebook, the PyPI `basicsr` package breaks on Kaggle Python 3.12 / torchvision ≥ 0.16 because `torchvision.transforms.functional_tensor` was removed. The notebook already self-patches `basicsr/data/degradations.py` to import from `torchvision.transforms.functional` instead. Don't strip that block.

### B.5 Spot-check the master

```bash
ffprobe -v quiet -print_format json -show_streams hd_out_4k/<name>_master.mov
```
You want to see:
- `"width": 3840, "height": 2160`
- `"codec_name": "prores"`, `"pix_fmt": "yuv444p10le"` (12-bit, the no-banding format)
- `"color_space": "bt709"`, `"color_primaries": "bt709"`, `"color_transfer": "bt709"`

If any of those is wrong, the upscale ran in a degraded mode — re-check that `PRORES_MASTER=True` and that the GPU was actually used (look for `[device] GPU (fp16)` in the log).

### B.6 Human review (final gate)

Open each `_master.mov` and the matching `_4k.mp4`. Watch face stability (GFPGAN can over-smooth at `FACE_WEIGHT=1.0`; if so, drop to 0.5 and re-run), ESRGAN shimmer on textured surfaces (raise `TEMPORAL_BLEND` to 0.20 if shimmer persists), and frame-rate smoothness (if 24fps interpolated frames look smeared on fast motion, set `RIFE_TFPS=0` to skip interpolation and keep the native fps).

---

## Quick reference: the whole thing in 8 commands

```bash
# Phase A — Kaggle online (the cloud session)
cd /kaggle/working/godshand/kit
python src/selftest.py --quick                       # free static gate
python src/selftest.py --motion --fast               # free mock production-path gate
# (run notebook Cell 1 for model downloads, ~10-15 min first time)
python gpu_smoke.py                                  # real GPU, one shot
CONFIRM_FULL_RUN=1 MAX_BUDGET_USD=8 AUTO_SHUTDOWN=1 \
  python run.py                                       # full film (2-3 h, resumable)
python /kaggle/working/scan_logs.py                  # post-run log scan
zip -r Beholder_Chunk_1.zip output_*/clips/*/master.mp4   # handoff to Phase B

# Phase B — Offline 4K (local workstation or second Kaggle session)
# Open upscale_hd_kaggle_4k.ipynb, attach Beholder_Chunk_1.zip, Run All.
# -> hd_out_4k.zip with <name>_4k.mp4 + <name>_master.mov (ProRes 4444 XQ, 12-bit)
```

## What every gate catches (and what it doesn't)

| Gate | Catches | Does NOT catch |
|---|---|---|
| `selftest --quick` | compile errors, missing deps, broken workflow `_bind`, malformed configs/shots | anything runtime/GPU |
| `selftest --motion --fast` | truncated `render_motion()` body (Defect #1), OOM fallback regression, NEED_INTERP guard flipped | real WAN output, model loading, identity quality |
| `runtime_integration_validator.py` | graph structure issues, NEED_INTERP conditional, minterpolate policy | real GPU execution |
| `gpu_smoke.py` | first-real-frame failures: OOM at load, model corruption, LoRA mismatch | per-shot identity drift, motion coherence |
| `scan_logs.py` (post-run) | Invalid-PNG, Killed, CUDA-OOM, runaway repeats, VHS plugin chatter, per-prompt timing regressions | *future* failures — it's reactive |
| Human review (A.10) | flicker, identity drift, color drift, audio sync, black/freeze frames | anything you skip scanning past |
| Upscale notebook `ffprobe` (B.5) | degraded upscale mode, wrong codec/pix_fmt/colorspace | face over-smoothing, ESRGAN shimmer, RIFE smear |
| Human review (B.6) | face over-smoothing, ESRGAN shimmer, RIFE motion smear | structural definition of "cinema" — that's a taste call |

**Two things none of these gates catch, and never will:**

1. **Pixel-perfect equality against an unseen reference.** There is no FID/dSSIM/reference-frame comparator in any suite. "Pixel-perfect" is a *taste* assertion, not a *testable* assertion, until you commit a reference master and add a comparator to `qc.py`.
2. **A 4K master that has *never been rendered end-to-end*.** The 4K upscale notebook has no integration test exercising it (existing suites top out at the HD master in stitch). Until you commit a reference 4K clip and add an upscale integration test, "4K cinema" is aspirational, not asserted. Priorities to fix this are listed in the test-run summary; not repeated here.

---

## Exit-code quick reference

| Command | rc=0 | rc=1 | rc=2 |
|---|---|---|---|
| `selftest.py --quick` | static clean | static issues | — |
| `selftest.py --motion --fast` | prod path green | prod path broken | — |
| `selftest.py --logs <log>` | no FATAL | FATAL found | log missing |
| `scan_logs.py` | clean | FATAL in log | log missing |
| `runtime_integration_validator.py` | all stages valid | any stage invalid | — |
| `gpu_smoke.py` | first real frame rendered | first frame failed | — |

If any gate fails: **STOP. Fix, then re-run from that gate.** Do not skip ahead hoping it "works anyway" — the original 79-min hang and silent `final.mp4` loss were exactly the class of defect these gates are designed to catch.

---

*Runbook derived from `KAGGLE_PREFLIGHT_CHECKLIST.md`, `KAGGLE_OPERATOR_COMMANDS.md`, `kit/RUN.md`, `kit/KAGGLE_TEST.md`, the `GODSHAND_COMPLETE_GATED_PIPELINE.ipynb` one-click notebook (8 gates), and the `upscale_hd_kaggle_4k.ipynb` standalone notebook (Phase B only). All gates 0–6 have regression tests in `comprehensive_smoke_test.py` (10/10) and `edge_case_integration_test.py` (36/36). Gates 7–8 (the 4K upscale pass) are exercised by the gated notebook but have no separate integration suite — treat their first real run as the integration test.*
