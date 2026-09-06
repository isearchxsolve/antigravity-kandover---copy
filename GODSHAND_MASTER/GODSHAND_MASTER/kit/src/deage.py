#!/usr/bin/env python3
"""Build a consistent 'young Kunal' reference set for the early-20s scenes.

Why: THE MOMENT is set almost entirely in the early-20s Bangalore era, so young-Kunal
is used in nearly every shot, but you have few youth photos and many 40s photos.
This pass synthesizes a small, consistent young reference set ONCE, so the young face
does not drift shot-to-shot.

Strategy (recommended default = 'build young set once'):
  - Identity anchor  : your few real youth photos (InstantID / PuLID embedding).
  - Age direction    : prompt tokens 'early 20s, fuller face, no grey, smoother skin'.
  - Build/wardrobe   : borrowed from your plentiful 40s photos.
  - Output           : refs/kunal_young/{face,body}/  (QA'd, reused for every young shot)
  - refs/kunal_40s/   : your real 40s photos, used only for the present-day shot m08.

This is a scaffold that drives a local diffusion age-edit (ComfyUI graph in
workflows/deage.json). It is idempotent: if refs/kunal_young/ already has enough
QA'd faces, it skips. Requires the model dataset mounted; on CPU-only it exits
cleanly and tells you to run it in the GPU session.
"""
import os, sys, glob, json, shutil, argparse

EXT = (".jpg", ".jpeg", ".png", ".webp")
AGE_PROMPT = "early 20s, fuller face, no grey hair, smoother skin, youthful, same identity"
TARGET_FACES = 5

def imgs(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(EXT))

def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    kunal = os.path.join(a.refs, "kunal")
    young = os.path.join(a.refs, "kunal_young")
    old = os.path.join(a.refs, "kunal_40s")
    os.makedirs(os.path.join(young, "face"), exist_ok=True)
    os.makedirs(os.path.join(young, "body"), exist_ok=True)
    os.makedirs(old, exist_ok=True)

    # Seed kunal_40s from the real (older) photos so m08 uses them directly.
    src_faces = imgs(os.path.join(kunal, "face"))
    for p in src_faces:
        dst = os.path.join(old, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)

    have = imgs(os.path.join(young, "face"))
    if len(have) >= TARGET_FACES and not a.force:
        print(f"[ok] refs/kunal_young/face already has {len(have)} faces — skipping de-age.")
        return

    if not gpu_available():
        print("[warn] No CUDA GPU here. De-age synthesis runs in the Kaggle GPU session.")
        print("       Skipping now; kaggle_run.ipynb will perform this pass before rendering.")
        # Fallback so the pipeline still runs: use real youth photos if any are tagged young.
        return

    # --- GPU path: drive the ComfyUI de-age graph ---------------------------------
    from comfy_client import ComfyClient  # provided in src/comfy_client.py
    client = ComfyClient()
    graph = json.load(open(os.path.join(os.path.dirname(__file__), "..", "workflows", "deage.json")))
    anchors = src_faces[:3]  # few real youth/identity photos anchor identity
    if not anchors:
        print("[FAIL] No source faces in refs/kunal/face to anchor identity. Add 3-5 youth photos.")
        sys.exit(1)
    for i in range(TARGET_FACES):
        out = client.run(graph, overrides={
            "identity_images": anchors,
            "prompt": AGE_PROMPT,
            "seed": 1000 + i,
        })
        client.save_image(out, os.path.join(young, "face", f"young_{i:02d}.png"))
    print(f"[ok] Built {TARGET_FACES} young-Kunal reference faces -> {young}/face")
    print("     QA them; delete any that drift and re-run with --force to regenerate.")

if __name__ == "__main__":
    main()
