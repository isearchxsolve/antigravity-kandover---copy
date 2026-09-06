# THE MOMENT — AI Film Kit (v3: Constrained + Hardened)

Generate the short film **"THE MOMENT"** on **free GPUs (Kaggle 2x T4)** from **photos + typed dialogue only** — no filmed footage, no paid services, no voice sample.

Built for three hard constraints:
1. No credit card / free-tier only.
2. No driving/motion video available.
3. Only photos of the two people (Kunal + Gunja).

---

## Quick start (6 steps)

### 1. Prepare inputs
- **Photos** — the easy way: dump ALL your photos into `refs/kunal/_inbox/` and `refs/gunja/_inbox/`. The preflight culler picks the best and files them into `face/ body/ costume/` and writes `selection_report.csv`. (Or hand-place them; see `refs/README.txt`.)
- **Dialogue** — already drafted in `script/lines.csv`. Edit any line if you like.
- **Music (optional)** — drop 3 files in `audio/bg/`: `descent.mp3`, `love.mp3`, `close.mp3`. Leave empty to auto-compose.

### 2. One-time Kaggle setup (cannot be scripted)
- Create a Kaggle account.
- Settings -> verify phone + verify identity (unlocks internet).
- Notebook -> Session Options -> Accelerator = **GPU T4 x2**, Internet = **On**.

### 3. Upload inputs as a private Kaggle Dataset
Upload `refs/`, `script/`, `audio/` as a private Dataset (mounts read-only, persists, doesn't eat the 20 GB working quota).

### 4. Build the model cache (run once)
Open `build_model_dataset.ipynb` -> Run All. Downloads + checksums all weights, saves as a reusable Dataset. ~30-60 min first time; ~2 min every session after.

### 5. Run the film
Open `kaggle_run.ipynb` -> **Save & Run All** (batch = no idle timeout). Pipeline:
`preflight -> cull photos -> build young-Kunal reference set (de-age) -> render each shot -> upscale/face-restore/interpolate -> voice + lip-sync -> stitch + mux audio`.
Resumable — a dropped session resumes from the last per-shot checkpoint.

### 6. Download the film
`THE_MOMENT.mp4` (plus per-shot files) lands zipped in `/kaggle/working/`.

---

## What you provide (exactly)
| Input | Where | Required? |
|-------|-------|-----------|
| Your photos | `refs/kunal/_inbox/` | Yes |
| Her photos | `refs/gunja/_inbox/` | Yes |
| Dialogue text | `script/lines.csv` | Pre-filled |
| 3 song files | `audio/bg/` | Optional (auto-score if empty) |
| Kaggle account + verification | (external) | Yes |

Everything else — models, code, workflows — the kit handles itself.

## Local / Lightning AI
Run `bash run.sh` instead of the Kaggle notebook. Same pipeline.

**Pick your card in one line** -- edit `active_profile` in `config.yaml`:
- `kaggle_2xt4` (default) -- FREE on Kaggle, full quality, $0 at risk.
- `lightning_l4` -- single L4 24GB (Ada, fp8), cost-cut so the whole film fits a few dollars / your free monthly L4 hours.

**Cost meter (never blind-burn credits):** the run renders ONE sample shot, prints the *measured* time + projected $ cost, then HOLDS. Only when you set `CONFIRM_FULL_RUN=1` does the whole film render. See `COST_AND_GPU.md`.

See `TROUBLESHOOTING.md` for the failure -> recovery map, and the Notion doc for the full rationale.
