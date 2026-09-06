#!/usr/bin/env python3
"""Cost meter: render ONE sample shot, measure real wall-clock time, then project
total GPU-hours and $ cost for the whole film under the active profile. Writes
work/cost_report.json. Does NOT start the full render -- filmforge holds until you
set CONFIRM_FULL_RUN=1. You see the true number after one shot, not after the money
is gone. The sample shot is checkpointed and reused, so nothing is wasted."""
import os, sys, json, time, subprocess

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    shots = json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]
    n = len(shots)
    first = shots[0]["id"]
    work = cfg["paths"]["work"]; os.makedirs(work, exist_ok=True)
    final = os.path.join(work, first, "final.mp4")
    elapsed = None
    if os.path.exists(final):
        print("[cost-meter] sample shot '%s' already rendered; cannot time a cached render." % first)
    else:
        print("[cost-meter] rendering ONE sample shot '%s' to measure real speed..." % first)
        # Use the correct renderer based on KIT_RENDER_MODE (default: motion_stills)
        render_mode = os.environ.get("KIT_RENDER_MODE", "motion_stills")
        if render_mode == "stills":
            render_script = "render_stills.py"
        elif render_mode == "motion_stills":
            render_script = "render_motion.py"
        else:
            render_script = "render_film.py"
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(root, "src", render_script),
                            "--only", first], cwd=root)
        if r.returncode != 0:
            print("[cost-meter] sample render FAILED -- fix the errors above before the full run.")
            sys.exit(r.returncode)
        elapsed = time.time() - t0
    per_shot_min = (elapsed / 60.0) if elapsed else 18.0
    tag = "measured" if elapsed else "fallback estimate (sample was cached)"
    ngpu = int(cfg.get("num_gpus", 1))
    rate = float(cfg.get("rate_per_gpu_hr", 0.0))
    takes = int(cfg.get("rejection_samples", 1))
    wall_hr = per_shot_min * n / 60.0
    gpu_hr = wall_hr * ngpu
    cost = gpu_hr * rate
    budget = float(os.environ.get("BUDGET_USD", "0") or 0)
    report = {"profile": cfg.get("_active_profile"), "shots": n, "sample_shot": first,
              "rejection_samples": takes, "num_gpus": ngpu,
              "per_shot_min": round(per_shot_min, 1),
              "projected_wall_hr": round(wall_hr, 1),
              "projected_gpu_hr": round(gpu_hr, 1),
              "rate_per_gpu_hr": rate, "projected_cost_usd": round(cost, 2),
              "budget_usd": budget}
    json.dump(report, open(os.path.join(work, "cost_report.json"), "w"), indent=2)
    print("")
    print("================= COST METER =================")
    print("  profile             : %s" % report["profile"])
    print("  shots               : %d  (rejection_samples=%d, gpus=%d)" % (n, takes, ngpu))
    print("  per-shot time       : %s min  [%s]" % (report["per_shot_min"], tag))
    print("  projected wall-clock: %s h" % report["projected_wall_hr"])
    print("  projected GPU-hours : %s h" % report["projected_gpu_hr"])
    if rate == 0:
        print("  projected cost      : $0.00  (free tier -- no money at risk)")
    else:
        print("  rate                : $%s / GPU-hr" % rate)
        print("  projected cost      : $%s" % report["projected_cost_usd"])
        if budget and cost > budget:
            print("  [!] EXCEEDS your budget of $%.2f -- it would stop before finishing." % budget)
            print("      Every shot is checkpointed, so you resume later with ZERO waste.")
            print("      Cheaper: keep lightning_l4, or set active_profile: kaggle_2xt4 (free).")
        elif budget:
            print("  [ok] within your budget of $%.2f." % budget)
    print("==============================================")
    print("  Sample shot saved and will be reused in the full render.")

if __name__ == "__main__":
    main()
