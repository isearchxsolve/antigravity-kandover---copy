#!/usr/bin/env python3
"""filmforge (THE ONE - v4 Blueprint-First) - resumable.
  preflight -> forge every face -> cull -> cost meter (HOLD)
  -> render perfect stills -> animate with low denoise
  -> identity check -> voice -> score -> sfx -> stitch.
Forces KIT_CONFIG=config_one.yaml. Fully generated: no real photos required.
Uses Blueprint-First pipeline (still-first) for maximum quality on T4.
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PY = sys.executable
os.environ["KIT_CONFIG"] = "config_one.yaml"

def stage(title, *cmd):
    print(f"\n===== {title} =====")
    r = subprocess.run(list(cmd), cwd=ROOT, env={**os.environ})
    if r.returncode != 0:
        print(f"[FAIL] stage '{title}' exited {r.returncode}")
        sys.exit(r.returncode)

def main():
    render_mode = os.environ.get("KIT_RENDER_MODE", "motion_stills")

    stage("1/9 preflight", PY, os.path.join(SRC, "preflight.py"))
    stage("2/9 forge faces (protagonist both ages + creature + her)", PY, os.path.join(SRC, "forge_character.py"))
    stage("3/9 cull any supplied photos (no-op when fully generated)", PY, os.path.join(SRC, "cull_photos.py"), os.path.join(ROOT, "refs/gunja"))
    stage("4/9 cost meter (1 sample shot + projection)", PY, os.path.join(SRC, "cost_meter.py"))
    if os.environ.get("CONFIRM_FULL_RUN") != "1":
        print("\n===== HELD AT COST METER =====")
        print("Review the projection above. To render: set CONFIRM_FULL_RUN=1. Sample shot kept and reused.")
        return

    # 5) Blueprint-First render
    if render_mode == "motion_stills":
        stage("5a/9 render blueprint stills", PY, os.path.join(SRC, "render_stills.py"))
        stage("5b/9 animate stills -> motion", PY, os.path.join(SRC, "render_motion.py"))
    elif render_mode == "stills":
        stage("5/9 render stills (comic-book)", PY, os.path.join(SRC, "render_stills.py"))
    else:
        stage("5/9 render shots (full motion)", PY, os.path.join(SRC, "render_film.py"))

    stage("6/9 identity check (CPU)", PY, os.path.join(SRC, "identity_check.py"))
    stage("7/9 voice (lead + chorus)", PY, os.path.join(SRC, "voice_pass.py"), "--root", ROOT)
    stage("8/9 score (auto-composed, wordless)", PY, os.path.join(SRC, "score_gen.py"), "--root", ROOT)
    stage("9/9 stitch + master", PY, os.path.join(SRC, "stitch.py"), "--root", ROOT)
    print("\nDONE -> output_one/THE_ONE.mp4" +
          (" + ProRes 4:4:4 master" if os.environ.get("KIT_MASTER") == "1" else ""))

if __name__ == "__main__":
    main()
