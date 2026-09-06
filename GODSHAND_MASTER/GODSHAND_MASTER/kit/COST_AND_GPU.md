# Cost & GPU guide -- THE MOMENT

Verified pricing (Lightning AI, 2026): **L4 = $0.48 / GPU-hr** on-demand; **1 credit = $1**;
the free tier advertises ~**31 free L4 GPU-hours/month** + ~15 credits (verify in your own
billing screen). Kaggle: **free**, ~30 GPU-hr/week on 2x T4. Render-time figures below are
engineering estimates; the built-in cost meter prints the REAL measured time after one shot.

## One-line card switch
Edit `active_profile` in `config.yaml`:
- `kaggle_2xt4` (default) -- FREE, full quality (Q8, 3 takes, 720p), $0 at risk.
- `lightning_l4` -- single L4 24GB (Ada, fp8), cost-cut (Q6, 1 take, 24 steps, 1152x648) so
  the whole film fits a few dollars / your free monthly L4 hours.

## The cost-meter gate (why you can never blind-burn credits)
`filmforge` runs: preflight -> cull -> de-age -> **render ONE sample shot** -> print the
measured per-shot time + projected total GPU-hours and $ cost -> then **HOLD**. It renders
the whole film ONLY when you set `CONFIRM_FULL_RUN=1`. The sample shot is checkpointed and
reused, so the measurement costs nothing extra.

```bash
# Kaggle (free):
python3 src/filmforge.py                        # STEP 1: see the numbers, then it stops
CONFIRM_FULL_RUN=1 python3 src/filmforge.py     # STEP 2: render the whole film

# Lightning L4 (cost-cut), $4 safety ceiling for the warning:
MOMENT_PROFILE=lightning_l4 BUDGET_USD=4 python3 src/filmforge.py
MOMENT_PROFILE=lightning_l4 CONFIRM_FULL_RUN=1 python3 src/filmforge.py
```

## Rough estimates (confirm with the meter)
| Plan | Card | Takes | Est. time | Est. cost |
|---|---|---|---|---|
| Full film, full quality | 2x T4 (Kaggle) | 3 | ~8-11 h wall / 2-3 free weekly sessions | $0 |
| Full film, cost-cut | 1x L4 (Lightning) | 1 | ~6-10 GPU-hr | ~$3-5 (or free L4 hours) |

Recommended: render free on Kaggle; keep the $4 L4 as a booster / insurance. Everything is
resumable -- a dropped session continues from the last finished shot.
