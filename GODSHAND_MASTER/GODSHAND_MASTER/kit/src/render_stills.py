#!/usr/bin/env python3
"""render_stills.py -- THE COMIC-BOOK PATH.

One pixel-perfect STILL per beat (full resolution, full steps, full rejection
sampling), held for the beat's duration as a frozen-frame video clip.  The
stills hold; the sound moves.  This is the DC-origin approach: the still page
carries the intensity, the suspense, the dread -- motion is infrastructure.

Output is identical in shape to render_film.py: <work>/<shot_id>/final.mp4 per
beat, so stitch.py (voice + orchestra + ambience + finale) works unchanged.

Speed: instead of generating N seconds of motion per beat (the expensive path),
we generate ONE image per beat and freeze-hold it.  On a Lightning H100 this cuts
GPU time by roughly 5-8x versus full S2V/Phantom/FLF2V motion rendering.

Usage:
  KIT_CONFIG=config_confrontation.yaml python3 src/render_stills.py          # all beats
  KIT_CONFIG=config_confrontation.yaml python3 src/render_stills.py --only search_void  # one beat
  KIT_CONFIG=config_confrontation.yaml python3 src/render_stills.py --smoke   # 1 beat, 1 take, low steps

Env:
  COMFY_MOCK=1  synthesize placeholder stills with ffmpeg (free offline dry-run)
  KIT_STILL_HOLD  override hold duration in seconds (default: shot's dur)
  KIT_STILL_KENBURNS=1  add a very slow sub-1% zoom-in on the still (optional, off by default)
"""
import os, sys, json, glob, argparse, subprocess
import gc

try:
    import torch
except Exception:
    torch = None


def load_cfg(root):
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg as _lc
    return _lc(root)


def load_shots(root, cfg):
    return json.load(open(os.path.join(root, cfg["shots_file"])))


def refs_for(shot, char, cfg, root):
    io = cfg.get("identity_override", {}) or {}
    if io.get("enabled"):
        d = os.path.join(root, io.get("refs_dir", "refs/original_identity"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*"), recursive=True):
            return d
    return os.path.join(root, "refs", char)


def done(work, sid):
    return os.path.exists(os.path.join(work, sid, "final.mp4"))


def _take_metrics(path):
    try:
        import cv2, numpy as np
    except Exception:
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    gd = cv2.medianBlur(img, 3)
    return {"sharpness": float(cv2.Laplacian(gd, cv2.CV_64F).var()),
            "exposure_ok": 25.0 <= float(img.mean()) <= 230.0}


def pick_best_still(takes):
    try:
        import cv2  # noqa: F401
    except Exception:
        return takes[0]
    scored = []
    for t in takes:
        m = _take_metrics(t)
        if m:
            scored.append((t, m))
    if not scored:
        return takes[0]
    sh = [m["sharpness"] for _, m in scored]
    lo, hi = min(sh), max(sh)
    best, best_score = scored[0][0], float("-inf")
    for t, m in scored:
        norm = 0.0 if hi <= lo else (m["sharpness"] - lo) / (hi - lo)
        score = norm - (0.0 if m["exposure_ok"] else 1.0)
        if score > best_score:
            best, best_score = t, score
    return best


def _run_with_oom_fallback(client, graph, overrides, sid, attempt=0):
    """Run ComfyUI with automatic OOM recovery.

    Module-level peer (not nested) so the surrounding function body
    stays contiguous and the inner function cannot be orphaned by
    a code-splice edit.  Mirrors the identical function in render_motion.py.
    """
    fallbacks = [
        (1.0, 1.0, 1.0),   # original
        (0.8, 1.0, 1.0),   # -20% steps
        (0.8, 0.875, 0.875), # -12.5% resolution
        (0.7, 0.75, 0.75),   # -30% steps, -25% res
        (0.5, 0.625, 0.625), # minimal
    ]
    if attempt >= len(fallbacks):
        raise RuntimeError(f"[{sid}] OOM: exhausted all fallback tiers")

    steps_pct, w_pct, h_pct = fallbacks[attempt]
    local_overrides = dict(overrides)

    if "steps" in local_overrides:
        local_overrides["steps"] = max(8, int(local_overrides["steps"] * steps_pct))

    orig_w = local_overrides.get("width", 1280)
    orig_h = local_overrides.get("height", 704)
    if w_pct < 1.0:
        new_w = max(512, int(orig_w * w_pct) // 64 * 64)
        new_h = max(288, int(orig_h * h_pct) // 64 * 64)
        local_overrides["width"] = new_w
        local_overrides["height"] = new_h

    if attempt > 0:
        print(f"[oom-fallback] {sid}: tier {attempt} -> steps={local_overrides.get('steps')}, "
              f"res={local_overrides.get('width')}x{local_overrides.get('height')}")

    try:
        return client.run(graph, overrides=local_overrides)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
            gc.collect()
            if torch:
                torch.cuda.empty_cache()
            return _run_with_oom_fallback(client, graph, overrides, sid, attempt + 1)
        raise


def render_still(shot, cfg, root, work):
    from comfy_client import ComfyClient
    sid = shot["id"]
    outdir = os.path.join(work, sid)
    os.makedirs(outdir, exist_ok=True)
    if done(work, sid):
        print(f"[skip] {sid} already rendered.")
        return

    client = ComfyClient()
    tone = json.load(open(os.path.join(root, cfg["shots_file"])))["globals"]
    rp = cfg.get("realism", {}) or {}
    pos_suffix = (", " + rp["positive"]) if rp.get("positive") else ""
    neg_extra = (", " + rp["negative"]) if rp.get("negative") else ""
    thr = float((cfg.get("intensity") or {}).get("keep_glitch_threshold", 0.66))
    intensity = float(shot.get("intensity", 0.0) or 0.0)
    keep_glitch = intensity >= thr
    fx_suffix = (", " + shot["fx"]) if shot.get("fx") else ""
    base_neg = shot.get("negative", tone["negative_prompt"]) + neg_extra
    if keep_glitch:
        for bad in ["flickering", "temporal artifacts", "blurry", "motion blur",
                    "soft focus", "low detail", "low quality"]:
            base_neg = base_neg.replace(", " + bad, "")
    if keep_glitch:
        print(f"[stills] {sid}: directed-glitch beat (intensity={intensity}); keeping artifacts as design.")

    positive = (shot["prompt"]
               + (", " + shot["palette"] if shot.get("palette") else "")
               + ", " + shot.get("tone", tone["tone"])
               + fx_suffix + pos_suffix)

    ref_dirs = {}
    if shot.get("chars"):
        ref_dirs = {c: refs_for(shot, c, cfg, root) for c in shot["chars"]}

    wf_name = "s2v" if shot.get("talking") else ("phantom" if len(shot.get("chars", [])) > 1 else "flf2v")
    if not shot.get("chars"):
        wf_name = "flf2v"
    graph = json.load(open(os.path.join(root, "workflows", f"{wf_name}.json")))

    takes = []
    n_samples = int(shot.get("rejection_samples", cfg["rejection_samples"]))
    for n in range(n_samples):
        overrides = {
            "positive": positive,
            "negative": base_neg,
            "ref_dirs": ref_dirs if ref_dirs else None,
            "width": cfg["width"], "height": cfg["height"],
            "steps": cfg["steps"], "cfg": cfg["cfg"],
            "seconds": 1,
            "seed": 42 + n,
        }
        out = _run_with_oom_fallback(client, graph, overrides, sid)
        still_path = os.path.join(outdir, f"take_{n}.png")
        client.save_image(out, still_path)
        takes.append(still_path)

    best = pick_best_still(takes)
    best_final = os.path.join(outdir, "still.png")
    os.replace(best, best_final)
    print(f"[stills] {sid}: best still -> {best_final}")

    hold = float(os.environ.get("KIT_STILL_HOLD", shot.get("dur", 4)))
    fps = int(cfg.get("interpolate_fps", cfg.get("fps", 24)))
    kenburns = os.environ.get("KIT_STILL_KENBURNS", "0") == "1"

    vf = f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if kenburns:
        vf = (f"scale=trunc(iw/2)*2:trunc(ih/2)*2,"
              f"zoompan=z='min(zoom+0.0005,1.02)':d={int(hold*fps)}"
              f":s={cfg['width']}x{cfg['height']}:fps={fps}")

    final_mp4 = os.path.join(outdir, "final.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "image2", "-loop", "1", "-framerate", str(fps), "-i", best_final,
        "-t", str(hold),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "slow", "-crf", "14",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        final_mp4
    ], check=True)
    print(f"[ok] {sid} -> {final_mp4} (still held {hold}s @ {fps}fps)")


def _auto_shutdown():
    cmd = os.environ.get("KIT_SHUTDOWN_CMD", "sudo shutdown -h now")
    print(f"[shutdown] work complete -> stopping instance: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=False)
    except Exception as e:
        print(f"[shutdown] FAILED ({e}) -- STOP THE INSTANCE MANUALLY to avoid charges.")


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shutdown", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0.0)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    cfg = load_cfg(root); work = cfg["paths"]["work"]; os.makedirs(work, exist_ok=True)
    if a.smoke:
        cfg["rejection_samples"] = 1
        cfg["steps"] = min(int(cfg.get("steps", 30)), 12)
        print("[smoke] stills validation: 1 beat, 1 take, low steps (cheap check).")
    shots = load_shots(root, cfg)["shots"]
    if a.smoke:
        shots = shots[:1]
    t0 = time.time()
    try:
        for shot in shots:
            if a.only and shot["id"] != a.only:
                continue
            if a.max_hours and (time.time() - t0) / 3600.0 >= a.max_hours:
                print(f"[budget] reached --max-hours={a.max_hours}; stopping before {shot['id']}. "
                      "Re-run to resume -- finished beats are auto-skipped.")
                break
            render_still(shot, cfg, root, work)
        print("[ok] stills render loop finished.")
    finally:
        if a.shutdown:
            _auto_shutdown()


if __name__ == "__main__":
    main()
