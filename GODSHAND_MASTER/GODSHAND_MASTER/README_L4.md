# GODSHAND -- Tuned + Hardened Kit (Lightning L4)

This package is the **union of every fix and hardening done to date** on the kit,
plus the confrontation-scene tuning, plus a single end-to-end runner.

## What's inside
- `GODSHAND_ONECLICK.sh` -- single script: unzip -> harden -> tune -> verify -> setup -> render -> shutdown.
- `kit/` -- the tuned + hardened kit (`run.py`, `src/`, `config_confrontation.yaml`, `l4_model_setup.py`, ...).
- `notebooks/` -- the hardened notebooks, **preserved untouched** (`wan-latest.ipynb`, `rainman-wan-multichar-FIXED.ipynb`).

## No-compromise guarantee (verified at build AND at run time)
The runner refuses to start unless ALL of these are present:

| Category | Item | Value |
|---|---|---|
| Permanent fix | NEED_INTERP interpolation guard (`postprocess.sh`) | present |
| Permanent fix | Vectorized `face_enhance.py` edge-mask (numpy) | present |
| Hardening | `interpolate_fps` clamped `<= 16` (never triggers degrading minterpolate) | 16 |
| Tuning | Active profile | `lightning_l4` |
| Tuning | Motion denoise (more freedom for WAN) | `0.55` |
| Tuning | CodeFormer fidelity (less over-smoothing) | `0.5` |
| Tuning | Temporal denoise hqdn3d (gentler) | `2.0:1.5:3.0:2.0` |
| Tuning | Face super-res composite | gated OFF (`KIT_FACE_ENHANCE`, default 0) |
| Tuning | End-frame morph for single-char flf2v shots | ON (`KIT_FLF2V_ENDFRAME=1`) |

> `interpolate_fps: 16` is deliberately kept **instead of 24**: WAN outputs 16fps, so 16 guarantees
> the NEED_INTERP guard always skips optical interpolation (the artifact/morph trap). 24 would re-enable it.

## Run it (on a fresh Lightning L4 Studio)
```bash
# put GODSHAND_ONECLICK.sh next to GODSHAND_FINAL.zip, then:
bash GODSHAND_ONECLICK.sh
```
That single command unzips, hardens, verifies, installs ComfyUI + the 5 custom nodes
(GGUF, WanVideoWrapper, InstantID, VideoHelperSuite, UltimateSDUpscale), downloads the
WAN 2.2 I2V MoE GGUF + UMT5 + VAE, launches ComfyUI, renders the 4 proof shots, and shuts down.

### Options
```bash
GODSHAND_FULL=1        bash GODSHAND_ONECLICK.sh   # render all 26 confrontation shots
GODSHAND_NO_SHUTDOWN=1 bash GODSHAND_ONECLICK.sh   # keep the box alive at the end
GODSHAND_ANCHORS=/path bash GODSHAND_ONECLICK.sh   # OPTIONAL: copy real face photos into kit/refs/<name>/
```

## Prerequisites you must supply
- A CUDA GPU box (Lightning **L4** recommended; the profile targets fp8/Q6, 24 steps, 1152x648).
- Network access for the first-run model download (tens of GB).
- **Character reference images are OPTIONAL.** `forge_character.py` auto-generates a consistent
  face/body set for each target (`jack`, `julie`) during the GPU session -- no photos required.
  To pin *specific real faces*, drop images into **`kit/refs/<name>/face/`** (and `/body/`) before running,
  or point `GODSHAND_ANCHORS` at a folder laid out as `<name>/face/*.jpg` (it is copied into `kit/refs/`).
  There is **no `kit/anchors/` folder** -- the real location is `kit/refs/<name>/`.
- `HF_TOKEN` only if a model repo requires auth (usually not needed for these public repos).

## Cost / safety notes
- The runner auto-shuts down when finished (prevents idle burn). Lightning also auto-sleeps after ~10 min idle.
- Proof scene (4 shots) is cheap; use it as the before/after against the T4 output before committing to the full clip.
- The 4 proof shots: `confr_caught`, `confr_allblades`, `confr_dare`, `confr_dontouch`.
