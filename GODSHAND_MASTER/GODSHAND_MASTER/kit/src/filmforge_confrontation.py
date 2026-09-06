#!/usr/bin/env python3
"""Standalone confrontation clip pipeline.
Applies the same upgraded video, cinematic voice, original score, ducking, and final mastering used by THE ONE.
"""
import os, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PY = sys.executable
os.environ["KIT_CONFIG"] = "config_confrontation.yaml"

def stage(title, *cmd):
    print(f"\n===== {title} =====")
    r = subprocess.run(list(cmd), cwd=ROOT, env={**os.environ})
    if r.returncode != 0:
        print(f"[FAIL] stage '{title}' exited {r.returncode}")
        sys.exit(r.returncode)

def main():
    stage("1/6 preflight", PY, os.path.join(SRC, "preflight.py"))
    stage("2/6 forge confrontation characters", PY, os.path.join(SRC, "forge_character.py"))
    stage("3/6 cost meter for confrontation clip", PY, os.path.join(SRC, "cost_meter.py"))
    if os.environ.get("CONFIRM_FULL_RUN") != "1":
        print("")
        print("===== HELD AT COST METER =====")
        print("Review the projection above (saved to /tmp/confrontation_work/cost_report.json).")
        print("To render the confrontation video: set CONFIRM_FULL_RUN=1 and run again.")
        return
    stage("4/6 render confrontation video", PY, os.path.join(SRC, "render_film.py"))
    stage("5/6 dramatic voice", PY, os.path.join(SRC, "voice_pass.py"), "--root", ROOT)
    stage("5b/6 original score", PY, os.path.join(SRC, "score_gen.py"), "--root", ROOT)
    stage("5c/6 lip-sync align (optional)", PY, os.path.join(SRC, "lipsync.py"), "--root", ROOT)
    stage("6/6 stitch + cinematic mix", PY, os.path.join(SRC, "stitch.py"), "--root", ROOT)
    print("\nDONE -> output_confrontation/THE_CONFRONTATION_CLIP.mp4")
if __name__ == "__main__":
    main()
