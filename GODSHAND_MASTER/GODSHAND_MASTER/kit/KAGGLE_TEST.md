# THE LAST LAUGH — Kaggle 2×T4 (BLUEPRINT PATH: stills → motion)

**Winning approach:** lock a pixel-perfect STILL per beat (face + dignity approved),
then ANIMATE each still into real motion. Final output = **full cinematic motion**,
not stills. This is the most reliable, most dignity-safe, and cheapest path.

Easiest way to run this: open **`kaggle_run_lastlaugh.ipynb`** and run the cells
top to bottom. The manual commands below do the same thing.

Profile is set to `kaggle_2xt4_max` (2 GPU · Q8 · best-of-4 rejection · upscale 2.0 · $0/hr).

---

## ⚠️ The one rule that prevents the big mistake
Both passes treat a beat as "done" when `final.mp4` exists. The **stills pass writes
a freeze `final.mp4`**. So BEFORE the motion pass you must delete those freeze files
(**keep `still.png`**) or the motion pass will skip every beat and animate nothing:
```bash
find /kaggle/working/lastlaugh_work -name final.mp4 -delete   # keeps still.png, drops the freeze
```

---

## 0. Kaggle setup
- Accelerator: **GPU T4 ×2**, Internet: **ON**.
- Upload the kit as a Dataset or clone it; model weights download on first run.
```bash
pip install -q torch transformers xformers
cd kit && export KIT_CONFIG=config_lastlaugh.yaml
```

## 1. Free validation (no spend)
```bash
KIT_CONFIG=config_lastlaugh.yaml python3 src/selftest.py --quick
KIT_CONFIG=config_lastlaugh.yaml python3 src/preflight.py
COMFY_MOCK=1 KIT_CONFIG=config_lastlaugh.yaml python3 src/render_stills.py --smoke
```

## 2. Real score on the T4 (free)
```bash
KIT_CONFIG=config_lastlaugh.yaml python3 src/score_gen.py --root .
```

## 3. PASS 1 — lock the 4 hero stills
```bash
for S in gift_firstlight mask_becoming confr_real confr_smile; do
  KIT_CONFIG=config_lastlaugh.yaml python3 src/render_stills.py --only "$S"
done
```
Review the stills at `/kaggle/working/lastlaugh_work/<beat>/still.png` — **approve the faces first.**

## 4. PASS 2 — animate the approved stills into motion
```bash
for S in gift_firstlight mask_becoming confr_real confr_smile; do
  rm -f /kaggle/working/lastlaugh_work/$S/final.mp4          # keep still.png, drop the freeze
  KIT_CONFIG=config_lastlaugh.yaml KIT_MOTION_DENOISE=0.45 python3 src/render_motion.py --only "$S"
done
```

## 5. Preview the motion
```bash
cd /kaggle/working/lastlaugh_work
for S in gift_firstlight mask_becoming confr_real confr_smile; do echo "file '$S/final.mp4'"; done > preview_list.txt
ffmpeg -y -f concat -safe 0 -i preview_list.txt -c copy /kaggle/working/PREVIEW_4BEATS_MOTION.mp4
```

---

## The full film (after the 4 beats satisfy you)
```bash
# A) stills for all 74 beats (resumable — rerun each session):
KIT_CONFIG=config_lastlaugh.yaml python3 src/render_stills.py --max-hours 8
# B) review every still.png, then drop the freeze mp4s (keep stills):
find /kaggle/working/lastlaugh_work -name final.mp4 -delete
# C) animate all beats into motion (resumable):
KIT_CONFIG=config_lastlaugh.yaml KIT_MOTION_DENOISE=0.45 python3 src/render_motion.py --max-hours 8
# D) assemble for free on CPU:
POST_MODE=cpu KIT_CONFIG=config_lastlaugh.yaml bash src/assemble.sh   # -> output_lastlaugh/THE_LAST_LAUGH.mp4
```

## Honest caveats
- **No one can promise 100% pixel-perfect.** Stills→motion gets you the CLOSEST
  by far and removes the worst failure (face morphing), because you approve every
  face as a still before it ever moves. The human gate IS the guarantee.
- Free Kaggle = ~9h/session + weekly quota; the full film will span several
  sessions (auto-resumes). Motion is heavier than stills — plan for it.
- The finale currently ends on a 1.4s freeze of the smile (`finale.freeze_hold`).
  That is one deliberate held frame. Set it to 0 in `config_lastlaugh.yaml` if you
  want pure motion all the way to black.
