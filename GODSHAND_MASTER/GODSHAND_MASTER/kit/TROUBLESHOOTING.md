# TROUBLESHOOTING — failure -> recovery map

The kit is idempotent, resumable, verified, fail-loud. One failure = one retry, never the whole run.

| Symptom | Cause | Fix |
|---|---|---|
| `preflight: No internet` | Kaggle internet off / unverified | Verify phone + identity, then Session Options -> Internet = On |
| `preflight: No nvidia-smi` | GPU not attached | Session Options -> Accelerator = GPU T4 x2 |
| Only 1 GPU seen | Single-GPU session | Kit still runs; falls back to Q5/Q6 low-VRAM. Prefer T4 x2 |
| Tunnel (Pinggy/Cloudflare) dies | Free tunnels drop | Not needed — kit renders via headless API. Tunnel optional with 3 fallbacks |
| Disk quota 20 GB / models vanish | `/kaggle/working` cap | Models live in read-only Dataset at `/kaggle/input`; scratch in `/tmp` |
| HF download stuck / 403 / partial | Flaky mirror | Auto: `HF_HUB_ENABLE_HF_TRANSFER=1`, resume, 3x retry, checksum. Re-run the build cell |
| Custom node import fails | Version drift | Pinned commits + post-install import assert; re-run build cell |
| Torch/CUDA/attention mismatch | T4 = Turing sm_75 | No SageAttention/fp8; xformers + GGUF; match preinstalled CUDA |
| OOM / RAM not released | VRAM pressure | `--cache-none`, block-swap, tiled VAE, subprocess-per-shot (auto) |
| Session timeout / disconnect | 9-12 h wall / idle | Use **Save & Run All** (batch). Per-shot checkpoints resume |
| Re-downloads 20 GB each session | No cached Dataset | Build the model Dataset once (Step 4); reused every session |
| Young Kunal face drifts | Too few youth refs | Add 3-5 clear youth photos; re-run `deage.py --force`; QA `refs/kunal_young/` |
| `cull_photos: no images` | Empty `_inbox` | Drop photos in `refs/<name>/_inbox/` OR hand-place in face/body/costume |
| Song sounds chopped | Wrong file lengths | Kit crossfades on cuts + shared pad; ensure files are full tracks, not clips |
| A face is smiling in a serious shot | Culler kept a smiling pick | Edit `selection_report.csv`, swap in a neutral photo, re-run render for that shot |

## Resume after a crash
Just re-run `kaggle_run.ipynb` (or `bash run.sh`). Finished shots are skipped
(checkpoint = `/tmp/moment_work/<shot>/final.mp4`); it picks up where it stopped.

## Force a clean rebuild of one shot
Delete `/tmp/moment_work/<shot_id>/` and re-run.
