# GODSHAND — Kaggle Pre-Flight Checklist
**Run this entire list on the Kaggle 2×T4 instance BEFORE starting the full notebook.**

---

## 0. Prerequisites (once per session)

- [ ] **GPU type confirmed:** 2× T4 (14.9 GB VRAM each) — verify in notebook first cell: `nvidia-smi`
- [ ] **Disk space:** ≥ 30 GB free on `/kaggle/working` (model weights + work dir + output)
- [ ] **Internet access:** Enabled (required for HF Hub model downloads on first run)

---

## 1. Repository Sync

```bash
# In a fresh Kaggle notebook cell:
!git clone https://github.com/<your-repo>/godshand.git /kaggle/working/godshand
# OR if using local upload:
# Upload the entire GODSHAND_MASTER/GODSHAND_MASTER/kit folder to /kaggle/working/godshand
```

- [ ] `kit/` folder present at `/kaggle/working/godshand/kit`
- [ ] `rainman-wan-MERGED-TUNED-FIXED.ipynb` present in repo root or `/kaggle/working`

---

## 2. Static Selftest (30 sec, no GPU)

```bash
cd /kaggle/working/godshand/kit
python src/selftest.py --quick
```

**Expected:** `[selftest --quick] ALL GREEN (static).`  
**If FAIL:** Do not proceed — fix static issues first.

---

## 3. Production-Path Selftest (~5 min, mock GPU)

```bash
python src/selftest.py --motion --fast
```

**Expected:** `[selftest --motion] ALL GREEN (fast)` + `work/m01/final.mp4` produced.  
**Why this matters:** This exercises `render_motion.py` (the exact path the notebook uses), not the legacy `render_film.py`. A FAIL here means the production path has a regression that `--fast` would miss.

---

## 4. Verify Notebook Graph Integrity

```bash
python -c "
import json
nb = json.load(open('rainman-wan-MERGED-TUNED-FIXED.ipynb'))
src = ''.join(nb['cells'][1]['source']) if isinstance(nb['cells'][1]['source'], list) else nb['cells'][1]['source']
vhs = src.count('VHS_VideoCombine')
si  = src.count('SaveImage')
print(f'VHS_VideoCombine: {vhs} (must be 0)')
print(f'SaveImage nodes : {si} (must be >= 2)')
assert vhs == 0 and si >= 2, 'Notebook graph integrity FAIL'
print('OK')
"
```

**Expected:** `VHS_VideoCombine: 0`, `SaveImage nodes: 2` (WAN + LTXV).

---

## 5. Validator Dry-Run (1 min, no GPU)

```bash
python src/runtime_integration_validator.py
```

**Expected:** Stage 3 dry-runs show `WORKFLOW VALID` for both `flf2v.json` and `charforge.json`. Stage 4 shows `✓ NEED_INTERP guard` and `✓ minterpolate conditional`. Errors allowed: only "MISSING: xformers / comfy / ComfyUI / custom nodes" (environmental on fresh instance).

---

## 6. Operator Log-Scanner Wrapper (copy to `/kaggle/working/scan_logs.py`)

```python
#!/usr/bin/env python3
"""Run after ANY Kaggle ComfyUI session to catch silent failures."""
import sys, subprocess, os

LOG = "/kaggle/working/comfyui.log"  # or comfyui_gpu0.log / download.txt
if not os.path.exists(LOG):
    print(f"[scan] Log not found: {LOG}")
    sys.exit(2)

# Uses the hardened parser from the kit
sys.path.insert(0, "/kaggle/working/godshand/kit/src")
import kaggle_log_validator as klv

findings = klv.scan_path(LOG)
any_fatal = any(f["severity"] == "FATAL" for f in findings)

print(f"\n=== Scan of {LOG} ===")
for f in findings:
    sev = "FATAL" if f["severity"] == "FATAL" else "WARN "
    mult = f"({f['count']}x)" if f.get("count", 1) > 1 else ""
    print(f"  [{sev}] L{f['line']:>5} {mult:<7} {f['id']}")

if any_fatal:
    print("\n[scan] FAIL — FATAL signatures detected. Inspect log before continuing.")
    sys.exit(1)
else:
    print("\n[scan] OK — no FATAL signatures.")
    sys.exit(0)
```

**Usage after each Kaggle session:**
```bash
python /kaggle/working/scan_logs.py
# rc=0 = clean, rc=1 = fatal found, rc=2 = log missing
```

---

## 7. First-Run Model Download (expect 10–15 min)

- [ ] Run the notebook **Cell 1 only** (model setup / HF Hub downloads)
- [ ] Verify: `WanTEModel`, `WanVAE`, `WAN21` all show `loaded completely; ... MB usable, ... MB loaded`
- [ ] Verify: GGUF qtypes line shows `F16 (694), Q4_K (280), Q6_K (120), F32 (1)`

---

## 8. Full Notebook Run (2–3 hours)

- [ ] Run **all remaining cells** in order
- [ ] Monitor: each `Prompt executed in XXX.XX seconds` should be 300–600s steady state
- [ ] After completion: run `python /kaggle/working/scan_logs.py`
- [ ] Verify outputs: `/kaggle/working/output_*/clips/*/master.mp4`, `vertical.mp4`, `thumb.jpg`, `manifest.json`
- [ ] Verify final: `SELFTEST.mp4` (or `confrontation.mp4` / `master.mp4` per config) exists and plays

---

## 9. Post-Run Human Review (the "Hollywood-grade" gate)

- [ ] Open `master.mp4` — inspect for flicker, identity drift, color consistency, audio sync
- [ ] QC pass: no black frames, freezes, loudness within EBU R128 target
- [ ] If all clear: **ship it**. If not: log findings, iterate config, re-run.

---

## Quick Reference: Exit Codes

| Command | rc=0 | rc=1 | rc=2 |
|---------|------|------|------|
| `selftest.py --quick` | static clean | static issues | — |
| `selftest.py --motion --fast` | prod path green | prod path broken | — |
| `selftest.py --logs <log>` | no FATAL | FATAL found | log missing |
| `scan_logs.py` | clean | FATAL in log | log missing |
| `runtime_integration_validator.py` | all stages valid | any stage invalid | — |

---

**If any gate above fails: STOP. Fix, then re-run from that gate.**  
Do not skip to the full notebook run hoping it "works anyway" — the original 79-min hang and silent `final.mp4` loss were exactly the class of defect these gates catch.

---

*Generated from GODSHAND_FULL_AUDIT_AND_FIXES_REPORT.md + log-validator suite. All gates have regression tests in `comprehensive_smoke_test.py` (10/10) and `edge_case_integration_test.py` (36/36).*