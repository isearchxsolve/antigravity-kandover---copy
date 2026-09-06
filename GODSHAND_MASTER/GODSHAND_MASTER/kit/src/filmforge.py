#!/usr/bin/env python3
"""filmforge — v4: BLUEPRINT-FIRST PIPELINE (near-100% pixel-perfect on Kaggle).
  preflight -> cull photos -> build young-Kunal set (de-age)
  -> render pixel-perfect STILLS per beat (render_stills)
  -> ANIMATE each still into motion with low denoise (render_motion)
  -> identity check + auto-retry (CPU insightface)
  -> voice (cinematic TTS with reverb bus)
  -> score (original dramatic score)
  -> sfx (ambience/foley)
  -> stitch + master (ProRes 4:4:4 optional)

Each stage is idempotent; a dropped session re-runs and skips finished work.
Set CONFIRM_FULL_RUN=1 after reviewing the cost meter to start the full render.

Env knobs:
  KIT_RENDER_MODE=motion     full motion render (text->video, no blueprint)
  KIT_RENDER_MODE=stills     comic-book path: freeze-held stills only
  KIT_MASTER=1               export ProRes 4:4:4 10-bit master in addition to H.264
                             (runs in postprocess.sh + grade_match.py + stitch.py)
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PY = sys.executable


def stage(title, *cmd):
    print(f"\n===== {title} =====")
    r = subprocess.run(list(cmd), cwd=ROOT)
    if r.returncode != 0:
        print(f"[FAIL] stage '{title}' exited {r.returncode}")
        sys.exit(r.returncode)


def main():
    # Determine render mode (default: blueprint path = stills-first)
    render_mode = os.environ.get("KIT_RENDER_MODE", "motion_stills")

    stage("1/7 preflight", PY, os.path.join(SRC, "preflight.py"))
    for person in ("refs/kunal", "refs/gunja"):
        stage(f"2/7 cull {person}", PY, os.path.join(SRC, "cull_photos.py"),
              os.path.join(ROOT, person))
    stage("3/7 build young-Kunal set (de-age)", PY, os.path.join(SRC, "deage.py"),
          "--refs", os.path.join(ROOT, "refs"))

    # Cost meter renders ONE sample shot using the selected mode
    stage("4/7 cost meter (1 sample shot + projection)", PY, os.path.join(SRC, "cost_meter.py"))

    if os.environ.get("CONFIRM_FULL_RUN") != "1":
        print("")
        print("===== HELD AT COST METER =====")
        print("Review the projection above (saved to work/cost_report.json).")
        print("Nothing else rendered; no budget committed.")
        print("To render the whole film: set CONFIRM_FULL_RUN=1 and run again.")
        print("The sample shot is kept and reused -- zero waste.")
        return

    # 5) Render: Blueprint-First path (default)
    if render_mode == "motion_stills":
        # Step A: pixel-perfect stills
        stage("5a/8 render blueprint stills", PY, os.path.join(SRC, "render_stills.py"))
        # Step B: animate each still into motion
        stage("5b/8 animate stills (blueprint -> motion)", PY, os.path.join(SRC, "render_motion.py"))
    elif render_mode == "stills":
        stage("5/8 render stills (comic-book path)", PY, os.path.join(SRC, "render_stills.py"))
    else:
        stage("5/8 render shots (full motion)", PY, os.path.join(SRC, "render_film.py"))

    # 6) Identity consistency check (CPU insightface)
    stage("6/8 identity check (CPU)", PY, os.path.join(SRC, "identity_check.py"))

    # 7) Audio pipeline (all CPU-side, runs while GPU is free)
    stage("7a/8 cinematic voice", PY, os.path.join(SRC, "voice_pass.py"), "--root", ROOT)
    stage("7b/8 original score", PY, os.path.join(SRC, "score_gen.py"), "--root", ROOT)
    stage("7c/8 sound design + ambience", PY, os.path.join(SRC, "sfx_gen.py"), "--root", ROOT)

    # 8) Final assembly
    stage("8/8 stitch + master", PY, os.path.join(SRC, "stitch.py"), "--root", ROOT)

    print("\nDONE -> output/THE_MOMENT.mp4" +
          (" + ProRes 4:4:4 master" if os.environ.get("KIT_MASTER") == "1" else ""))


if __name__ == "__main__":
    main()
