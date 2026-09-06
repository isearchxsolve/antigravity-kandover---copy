#!/usr/bin/env python3
"""Character forge (Film II) - generate reference stills for INVENTED characters.

The Observer is never photographed. His reference set (face angles + full body) is
generated ONCE from the prompts in <shots_file>.character_design, then filed into
refs/<target>/face and refs/<target>/body, so the rest of the pipeline treats him
exactly like a photographed character. No user photos are needed.

Idempotent: skips a target whose refs/<target>/face is already populated, unless
character_forge.overwrite = true. No ComfyUI reachable (e.g. CPU/local) -> exits
cleanly and reports which targets will be generated in the GPU session.
"""
import os, sys, json

EXT = (".jpg", ".jpeg", ".png", ".webp")

def load_cfg(root):
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg as _lc
    return _lc(root)

def populated(d):
    return os.path.isdir(d) and any(f.lower().endswith(EXT) for f in os.listdir(d))

def gpu_ready():
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8188/", timeout=5)
        return True
    except Exception:
        return False

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_cfg(root)
    fc = cfg.get("character_forge") or {}
    if not fc.get("enabled"):
        print("[forge] character_forge disabled in config; nothing to do.")
        return
    shots_json = json.load(open(os.path.join(root, cfg["shots_file"])))
    design_all = shots_json.get("character_design", {})
    tone = shots_json["globals"]
    targets = fc.get("targets", [])
    overwrite = bool(fc.get("overwrite"))
    wf_name = fc.get("workflow", "charforge")
    graph_path = os.path.join(root, "workflows", f"{wf_name}.json")
    if not os.path.exists(graph_path):
        import subprocess
        subprocess.run([sys.executable, os.path.join(root, "src", "make_workflows.py")], cwd=root)
    todo = []
    for t in targets:
        face_d = os.path.join(root, "refs", t, "face")
        if not overwrite and populated(face_d):
            print(f"[forge] {t}: reference set already present ({face_d}); skipping."
                  " (set character_forge.overwrite: true to rebuild)")
            continue
        todo.append(t)
    if not todo:
        print("[forge] all targets already forged.")
        return
    if not gpu_ready():
        print("[forge] ComfyUI not reachable at 127.0.0.1:8188 (likely CPU/local box).")
        print("        The character(s) will be generated automatically in the GPU session:")
        print("        awaiting forge -> " + ", ".join(todo))
        return
    from comfy_client import ComfyClient
    client = ComfyClient()
    graph = json.load(open(graph_path))
    for t in todo:
        design = design_all.get(t, {})
        prompts = design.get("forge_prompts", {})
        per = int(design.get("forge_count_per_prompt", 2))
        seed_base = int(design.get("seed_base", 42))
        face_d = os.path.join(root, "refs", t, "face")
        body_d = os.path.join(root, "refs", t, "body")
        os.makedirs(face_d, exist_ok=True)
        os.makedirs(body_d, exist_ok=True)
        i = 0
        for key, ptext in prompts.items():
            dest_dir = body_d if key.startswith("body") else face_d
            for n in range(per):
                i += 1
                dest = os.path.join(dest_dir, f"{t}_{key}_{n}.png")
                if os.path.exists(dest) and not overwrite:
                    continue
                out = client.run(graph, overrides={
                    "positive": ptext + ", " + tone["tone"],
                    "negative": tone["negative_prompt"],
                    "width": cfg["width"], "height": cfg["height"],
                    "steps": cfg["steps"], "cfg": cfg["cfg"],
                    "seed": seed_base + i,
                })
                client.save_image(out, dest)
                print(f"[forge] {t}: {os.path.basename(dest)}")
    print("[forge] character forge complete. Reference stills written under refs/<target>/.")

if __name__ == "__main__":
    main()
