#!/usr/bin/env python3
"""filmforge (Film II - THE OBSERVER) - single entry point, resumable.
  preflight -> FORGE invented character -> cost meter (HOLD) -> render -> voice -> stitch.
No photo cull and no de-age (the protagonist is generated, not photographed).
Forces KIT_CONFIG=config_observer.yaml so every stage sees the Film II config.
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PY = sys.executable
os.environ["KIT_CONFIG"] = "config_observer.yaml"

def stage(title, *cmd):
    print(f"\n===== {title} =====")
    r = subprocess.run(list(cmd), cwd=ROOT, env={**os.environ})
    if r.returncode != 0:
        print(f"[FAIL] stage '{title}' exited {r.returncode}")
        sys.exit(r.returncode)

def main():
    stage("1/6 preflight", PY, os.path.join(SRC, "preflight.py"))
    stage("2/6 forge character (generate the Observer)", PY, os.path.join(SRC, "forge_character.py"))
    stage("3/6 cost meter (1 sample shot + projection)", PY, os.path.join(SRC, "cost_meter.py"))
    if os.environ.get("CONFIRM_FULL_RUN") != "1":
        print("")
        print("===== HELD AT COST METER =====")
        print("Review the projection above (saved to work/cost_report.json).")
        print("Nothing else rendered; no budget committed.")
        print("To render the whole film: set CONFIRM_FULL_RUN=1 and run again.")
        print("The sample shot is kept and reused -- zero waste.")
        return
    stage("4/6 render shots", PY, os.path.join(SRC, "render_film.py"))
    stage("5/6 voice", PY, os.path.join(SRC, "voice_pass.py"), "--root", ROOT)
    stage("6/6 stitch + mux", PY, os.path.join(SRC, "stitch.py"), "--root", ROOT)
    print("\nDONE -> output_observer/THE_OBSERVER.mp4")

if __name__ == "__main__":
    main()
