# Quick-Reference: Exact Commands for Kaggle Operator

Copy-paste each block into a separate Kaggle notebook cell. Run in order.

---

## Cell 1 — Clone / Sync Repo
```bash
!git clone https://github.com/<your-repo>/godshand.git /kaggle/working/godshand
# OR if uploaded manually: verify /kaggle/working/godshand/kit exists
```

---

## Cell 2 — Static Selftest (30 sec)
```bash
%cd /kaggle/working/godshand/kit
!python src/selftest.py --quick
# EXPECT: [selftest --quick] ALL GREEN (static).
```

---

## Cell 3 — Model Downloads (run notebook Cell 1, or this equivalent)
```bash
# If using the notebook: just run Cell 1 of rainman-wan-MERGED-TUNED-FIXED.ipynb
# It downloads WanTEModel, WanVAE, WAN21, GGUF weights from HF Hub.
# Expect 10–15 min on first run; cached thereafter.
```

---

## Cell 4 — Production-Path GPU Smoke Test (run AFTER model downloads)
```bash
!python gpu_smoke.py
# EXPECT: [gpu_smoke] ✅ GREEN — production path renders on real GPU
# If RED: stop, debug, do not proceed to full run.
```

---

## Cell 5 — Full Notebook Run
```bash
# Run ALL remaining cells of rainman-wan-MERGED-TUNED-FIXED.ipynb in order
# This takes 2–3 hours for a full confrontation film.
# Monitor: each "Prompt executed in XXX.XX seconds" should be 300–600s steady.
```

---

## Cell 6 — Post-Run Log Scan (after notebook completes)
```bash
!python scan_logs.py
# EXPECT: [scan] OK — no FATAL signatures.
# If FAIL: inspect the log lines printed above before shipping output.
```

---

## Cell 7 — Verify Deliverables
```bash
!ls -la /kaggle/working/output_*/clips/*/
!ls -la /kaggle/working/output_*/SELFTEST.mp4 2>/dev/null || \
     ls -la /kaggle/working/output_*/confrontation.mp4 2>/dev/null || \
     ls -la /kaggle/working/output_*/master.mp4
# Should show: master.mp4, vertical.mp4, thumb.jpg, manifest.json per clip
# And a final master/SELFTEST.mp4 or confrontation.mp4
```

---

## If Any Step Fails

| Step | Action |
|------|--------|
| Cell 2 (static) | Fix code/config, re-run Cell 2 |
| Cell 4 (GPU smoke) | Check GPU VRAM, model paths, `COMFY_MOCK=0`; re-run Cell 4 |
| Cell 6 (log scan) | Open the failing log, fix root cause, re-run from Cell 4 |
| Cell 7 (deliverables) | Re-run failed clips via `render_motion.py --only <shot_id>` |

---

## One-Liner Cheat Sheet (for terminal)

```bash
# All-in-one verification after a Kaggle session:
cd /kaggle/working/godshand/kit && \
python src/selftest.py --quick && \
python gpu_smoke.py && \
python /kaggle/working/scan_logs.py && \
echo "ALL GATES GREEN"
```