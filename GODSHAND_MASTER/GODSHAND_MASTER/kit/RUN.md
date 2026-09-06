# THE CONFRONTATION CLIP -- run guide

One script runs everything with every guardrail in front of the money: `run.py`.

## The whole flow, one command

```bash
# 1) SAFE by default: audit + render ONE sample + project cost, then STOP.
python3 run.py

# 2) When you're satisfied with the projection, render the full clip:
CONFIRM_FULL_RUN=1 MAX_BUDGET_USD=8 AUTO_SHUTDOWN=1 python3 run.py
```

`run.py` never spends beyond a single sample shot unless `CONFIRM_FULL_RUN=1`.

## Stages (fail-fast, each gates the next)
1. **preflight** (free, ~5s) -- missing model/dep/GPU/disk aborts for ~0 rupees.
2. **forge** -- build character identities (idempotent).
3. **cost meter** -- render ONE real sample, measure true speed, project GPU-hours + $.
4. **budget gate** -- if projected cost > `MAX_BUDGET_USD`, ABORT before the full render.
5. **hold gate** -- stops unless `CONFIRM_FULL_RUN=1`.
6. **render** -- resumable; finished shots auto-skip; optional `--max-hours` + auto-shutdown.
7-10. **voice -> score -> lipsync(skips) -> stitch** -> `output_confrontation/THE_CONFRONTATION_CLIP.mp4`

## Env knobs
| var | meaning | default |
|---|---|---|
| `CONFIRM_FULL_RUN=1` | required to spend GPU past the cost-meter sample | unset (dry-run) |
| `MAX_BUDGET_USD=8` | hard $ ceiling; abort if projection exceeds it | unset (no ceiling) |
| `MAX_HOURS=6` | wall-clock cap for the render loop (resumable) | 0 (no cap) |
| `AUTO_SHUTDOWN=1` | stop the instance the moment rendering ends | 0 |
| `KIT_REQUIRE_GPU=1` | 'no GPU' becomes a fatal preflight error | auto-set on paid profiles |

## Profiles (`config_confrontation.yaml` -> `active_profile`)
- `lightning_h100_max` (default) -- bf16, 50 steps, upscale 2.0, 1280x720 base.
- `lightning_a100_max` -- same, 80GB A100.
- `lightning_a100_40gb` -- VRAM-safe: 45 steps, MusicGen-medium auto-selected by run.py.

Rates in each profile are on-demand estimates. **If you're on free GPU hours, set that profile's `rate_per_gpu_hr: 0.0`** and the cost meter reports $0.

## Resolution note
Base render is 1280x720, upscaled x2.0 -> **1440p** master (not true 4K). To push higher, raise the base or the upscale factor (costs more GPU).

## Recommended path on Lightning ($ safe)
```bash
python3 run.py                                   # see the real projection first
CONFIRM_FULL_RUN=1 MAX_BUDGET_USD=4 AUTO_SHUTDOWN=1 python3 run.py
```
Start on one account's free hours; the render is resumable across sessions/accounts (finished shots are skipped), so a mid-run stop never wastes a rupee.
