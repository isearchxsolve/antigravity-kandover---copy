#!/usr/bin/env python3
"""Optional lip-sync alignment stage.

Tightens each TALKING shot's mouth motion to its generated voice using LatentSync
(see MANIFEST: bytedance/LatentSync). This makes the dramatic voice land faithfully
on the lips instead of drifting.

Fully optional and NON-FATAL: if LatentSync / a GPU is unavailable it skips cleanly
(exit 0) so the pipeline always continues. When it runs it replaces
work/<shot>/final.mp4 in place, so stitch.py picks up the synced clip automatically.

Enable/disable via config: lipsync.enabled (default true). To activate on Kaggle,
clone the repo once:  git clone https://github.com/bytedance/LatentSync /kaggle/working/LatentSync
"""
import os, sys, glob, json, shutil, subprocess, argparse


def gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def find_latentsync():
    for base in ("/kaggle/working/LatentSync", "/kaggle/input", os.path.expanduser("~")):
        for cand in glob.glob(os.path.join(base, "**", "inference.py"), recursive=True):
            if "latentsync" in cand.lower():
                return cand
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args()
    root = os.path.abspath(a.root)
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    if not (cfg.get("lipsync") or {}).get("enabled", True):
        print("[lipsync] disabled in config; skipping.")
        return 0
    work = cfg["paths"]["work"]
    vdir = os.path.join(work, "voice")
    shots = json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]
    talking = [s for s in shots if s.get("talking")]
    if not talking:
        print("[lipsync] no talking shots; nothing to align.")
        return 0
    if not gpu():
        print("[lipsync] no GPU here; lip-sync runs in the GPU session. Skipping.")
        return 0
    infer = find_latentsync()
    if not infer:
        print("[lipsync] LatentSync not installed. To enable, clone bytedance/LatentSync")
        print("          into /kaggle/working/LatentSync. Skipping without failing.")
        return 0
    n_ok = 0
    for s in talking:
        sid = s["id"]
        vid = os.path.join(work, sid, "final.mp4")
        wav = os.path.join(vdir, sid + ".wav")
        if not (os.path.exists(vid) and os.path.exists(wav)):
            continue
        out = os.path.join(work, sid, "lipsync.mp4")
        cmd = [sys.executable, infer, "--video_path", vid, "--audio_path", wav,
               "--video_out_path", out]
        try:
            r = subprocess.run(cmd, cwd=os.path.dirname(infer))
            if r.returncode == 0 and os.path.exists(out):
                shutil.move(out, vid)
                n_ok += 1
                print(f"[lipsync] aligned {sid}")
            else:
                print(f"[lipsync] {sid}: LatentSync returned {r.returncode}; keeping original.")
        except Exception as e:
            print(f"[lipsync] {sid}: {e}; keeping original.")
    print(f"[lipsync] done ({n_ok} shot(s) aligned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
