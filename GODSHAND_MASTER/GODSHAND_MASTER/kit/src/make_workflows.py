#!/usr/bin/env python3
"""Generate placeholder ComfyUI node graphs the render/de-age stages load.
These are minimal, valid JSON skeletons with the node types this kit uses; the
first GPU run auto-populates model paths from the mounted model dataset (MANIFEST).
Replace with your tuned graphs anytime — file names are the contract.
"""
import os, json

WF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows")

GRAPHS = {
    "s2v": {"purpose": "Wan2.2-S2V talking shots (photo + voice -> performance)",
            "nodes": ["WanSoundImageToVideo", "PuLID", "VAEDecode"]},
    "phantom": {"purpose": "Wan Phantom two-person subject-to-video",
                "nodes": ["WanPhantomSubjectToVideo", "InstantID", "VAEDecode"]},
    "flf2v": {"purpose": "Wan FLF2V first-last-frame continuation / reveals",
              "nodes": ["WanFirstLastFrameToVideo", "VAEDecode"]},
    "deage": {"purpose": "Diffusion age-edit to build young reference set",
              "nodes": ["InstantID", "PuLID", "KSampler", "VAEDecode"]},
    "upscale": {"purpose": "SUPIR/SeedVR2/4x-UltraSharp two-stage upscale",
                "nodes": ["SUPIR", "UpscaleModelLoader", "ImageUpscaleWithModel"]},
    "charforge": {"purpose": "Text-to-image character design: generate invented-character reference stills (SDXL/Flux)",
                  "nodes": ["CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode", "SaveImage"]},
}

def main():
    os.makedirs(WF, exist_ok=True)
    wrote = 0
    for name, spec in GRAPHS.items():
        p = os.path.join(WF, f"{name}.json")
        if os.path.exists(p):
            # never clobber a real, populated graph; only heal missing files
            continue
        json.dump({"_comment": spec["purpose"], "_nodes_used": spec["nodes"],
                   "nodes": {}}, open(p, "w"), indent=2)
        wrote += 1
    print(f"[ok] wrote {wrote} missing workflow skeleton(s) to {WF}; kept existing graphs.")

if __name__ == "__main__":
    main()
