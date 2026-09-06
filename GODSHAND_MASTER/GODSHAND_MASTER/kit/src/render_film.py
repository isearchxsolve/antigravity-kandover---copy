#!/usr/bin/env python3
"""Render each shot via ComfyUI (Wan 2.2-S2V / Phantom / FLF2V), then
upscale -> face-restore -> interpolate. Resumable: per-shot checkpoints in work dir.
Rejection sampling: generate N takes, auto-pick the sharpest/most-stable.
"""
import os, sys, json, glob, argparse, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vram_governor as vg
import gpu_scheduler
import thermal_monitor

def load_cfg(root):
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg as _lc
    return _lc(root)

def load_shots(root, cfg):
    return json.load(open(os.path.join(root, cfg["shots_file"])))

def refs_for(shot, char, cfg, root):
    """Pick era-correct refs: young shots -> refs/kunal_young, m08 -> refs/kunal_40s.
    If identity_override is enabled, ALL characters use your OWNED synthetic identity
    (refs/original_identity) instead of real-actor refs -> avoids personality-rights liability."""
    io = cfg.get("identity_override", {}) or {}
    if io.get("enabled"):
        d = os.path.join(root, io.get("refs_dir", "refs/original_identity"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*"), recursive=True):
            return d
    if char == "kunal":
        era = shot.get("era", "young")
        sub = "kunal_young" if era == "young" else "kunal_40s"
        d = os.path.join(root, "refs", sub)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*"), recursive=True):
            return d
    return os.path.join(root, "refs", char)

def done(work, sid):
    return os.path.exists(os.path.join(work, sid, "final.mp4"))

def render_shot(shot, cfg, root, work, gpu_idx=0):
    from comfy_client import ComfyClient
    sid = shot["id"]
    outdir = os.path.join(work, sid)
    os.makedirs(outdir, exist_ok=True)
    if done(work, sid):
        print(f"[skip] {sid} already rendered.")
        return
    client = ComfyClient()
    chars = shot.get("chars", [])
    wf_name = "s2v" if shot.get("talking") else ("phantom" if len(chars) > 1 else "flf2v")
    graph = json.load(open(os.path.join(root, "workflows", f"{wf_name}.json")))
    ref_dirs = {c: refs_for(shot, c, cfg, root) for c in shot["chars"]}
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
    flicker_weight = 0.0 if keep_glitch else 1.5
    if keep_glitch:
        print(f"[render] {sid}: directed-glitch beat (intensity={intensity}); keeping artifacts as design.")

    params = vg.resolve_params(cfg, gpu_idx, shot)
    print(f"[render] {sid} on GPU {gpu_idx}: regime={params['regime']} "
          f"free={params['free_mb']}MB steps={params['steps']} "
          f"dur={params['seconds']}s "
          f"offload={params['offload_level']} attn_slice={params['attention_slicing']} "
          f"vae_tile={params['vae_tiling']}/{params['vae_tile_size']} "
          f"fp8={params['fp8']}")
    if params["regime"] == "danger":
        os.environ["KIT_LOW_VRAM"] = "1"
    else:
        os.environ.pop("KIT_LOW_VRAM", None)

    thermal_monitor.snapshot(gpu_idx, f"before_{sid}")

    takes = []
    max_tries = int(os.environ.get("KIT_SHOT_RETRIES", "3"))
    for n in range(int(shot.get("rejection_samples", cfg["rejection_samples"]))):
        last_err = None
        for attempt in range(max_tries):
            try:
                out = client.run(graph, overrides={
                    "positive": shot["prompt"] + (", " + shot["palette"] if shot.get("palette") else "") + ", " + shot.get("tone", tone["tone"]) + fx_suffix + pos_suffix,
                    "negative": base_neg,
                    "ref_dirs": ref_dirs,
                    "width": params["width"], "height": params["height"],
                    "steps": params["steps"], "cfg": cfg["cfg"],
                    "seconds": min(shot["dur"], params["seconds"]),
                    "seed": 42 + n,
                })
                take = client.save_image(out, os.path.join(outdir, f"take_{n}.mp4"))
                takes.append(take)
                last_err = None
                break
            except Exception as e:
                last_err = e
                wait = 8 * (attempt + 1)
                print(f"[render] {sid} take {n} attempt {attempt+1} failed: {e}; "
                      f"sleeping {wait}s and retrying.")
                time.sleep(wait)
        if last_err and not takes:
            raise RuntimeError(f"{sid} take {n} failed after {max_tries} attempts: {last_err}")
        if last_err:
            print(f"[render] {sid} take {n} gave up after {max_tries} attempts; "
                  f"using whatever was produced.")
    best = pick_best(takes, flicker_weight=flicker_weight)
    _do_postprocess(root, best, os.path.join(outdir, "final.mp4"),
                    cfg["upscale"], cfg["interpolate_fps"],
                    cfg["codeformer_fidelity"],
                    os.path.join(root, "refs", "grade_target.json"))
    thermal_monitor.snapshot(gpu_idx, f"after_{sid}")
    print(f"[ok] {sid} -> {outdir}/final.mp4")

def _take_metrics(path):
    """Raw per-take fidelity metrics. Sharpness is measured on a median-blurred frame
    so random generation NOISE is not mistaken for real detail. Flicker is the mean
    |2nd-order temporal difference| of luminance, so steady/legit motion scores ~0
    while flicker/jitter/instability spikes it. Deterministic + testable."""
    import cv2, numpy as np
    cap = cv2.VideoCapture(path)
    grays, sharp = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gd = cv2.medianBlur(g, 3)  # suppress per-pixel noise before measuring detail
        sharp.append(cv2.Laplacian(gd, cv2.CV_64F).var())
        grays.append(g.astype("float32"))
    cap.release()
    if not grays:
        return None
    jerks = [float(np.mean(np.abs(grays[i] - 2 * grays[i - 1] + grays[i - 2])))
             for i in range(2, len(grays))]
    mean_lum = float(np.mean([g.mean() for g in grays]))
    return {
        "sharpness": float(np.mean(sharp)),
        "flicker": float(np.mean(jerks)) if jerks else 0.0,
        "exposure_ok": 25.0 <= mean_lum <= 230.0,
    }


def pick_best(takes, flicker_weight=1.5):
    """Pick the best take: rank candidates by NORMALIZED detail vs. temporal stability.
    Temporal stability is weighted higher than raw sharpness because faithful, stutter-
    free motion reads as higher quality than a sharp-but-flickery clip. Poorly exposed
    takes are penalized. Falls back to the first take if OpenCV is unavailable."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return takes[0]
    scored = []
    for t in takes:
        try:
            m = _take_metrics(t)
        except Exception:
            m = None
        if m:
            scored.append((t, m))
    if not scored:
        return takes[0]
    sh = [m["sharpness"] for _, m in scored]
    fl = [m["flicker"] for _, m in scored]

    def _norm(x, xs):
        lo, hi = min(xs), max(xs)
        return 0.0 if hi <= lo else (x - lo) / (hi - lo)

    best, best_score = scored[0][0], float("-inf")
    for t, m in scored:
        score = (_norm(m["sharpness"], sh)
                 - flicker_weight * _norm(m["flicker"], fl)
                 - (0.0 if m["exposure_ok"] else 1.0))
        if score > best_score:
            best, best_score = t, score
    return best


def _do_postprocess(root, inp, out, upscale, fps, fidelity, grade):
    """Run postprocess.sh on Linux, ffmpeg-only Python fallback on Windows."""
    import shutil
    if sys.platform != "win32" and shutil.which("bash"):
        subprocess.run(["bash", os.path.join(root, "src", "postprocess.sh"),
                        inp, out, str(upscale), str(fps), str(fidelity), grade], check=True)
    else:
        # Cross-platform ffmpeg fallback (no bash needed)
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            den = os.path.join(tmp, "d.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inp,
                "-vf", "hqdn3d=4.0:3.0:6.0:3.0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "10",
                "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                den], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", den,
                "-vf", f"scale=iw*{upscale}:ih*{upscale}:flags=lanczos,"
                       f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                "-c:v", "libx264", "-preset", "slow", "-crf", "12",
                "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                out], check=True)
            print(f"[postprocess(ffmpeg)] -> {out}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _auto_shutdown():
    """Stop the meter the instant work ends -- the #1 way people burn pockets is a
    FINISHED render idling on a live GPU. Set KIT_SHUTDOWN_CMD to your provider's
    stop/terminate call (e.g. an API curl for spot); else best-effort OS poweroff."""
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
    ap.add_argument("--smoke", action="store_true",
                    help="cheap paid-env validation: first shot only, 1 take, low steps, no upscale")
    ap.add_argument("--shutdown", action="store_true",
                    help="auto-stop the instance the moment the render loop ends")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="hard wall-clock budget; will not START a new shot past this (resumable)")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    cfg = load_cfg(root); work = cfg["paths"]["work"]; os.makedirs(work, exist_ok=True)
    if a.smoke:
        cfg["rejection_samples"] = 1
        cfg["steps"] = min(int(cfg.get("steps", 30)), 12)
        cfg["upscale"] = 1.0
        print("[smoke] validation pass: 1 shot, 1 take, low steps, no upscale (cheap paid-env check).")
    shots = load_shots(root, cfg)["shots"]
    if a.smoke:
        shots = shots[:1]
    sched = gpu_scheduler.Scheduler()
    print(f"[render] GPU scheduler active: {sched.gpu_count} GPU(s).")
    t0 = time.time()
    try:
        for shot in shots:
            if a.only and shot["id"] != a.only:
                continue
            if a.max_hours and (time.time() - t0) / 3600.0 >= a.max_hours:
                print(f"[budget] reached --max-hours={a.max_hours}; stopping before {shot['id']}. "
                      "Re-run to resume -- finished shots are auto-skipped.")
                break
            gpu_idx = sched.pick()
            render_shot(shot, cfg, root, work, gpu_idx=gpu_idx)
            sched.finish(gpu_idx)
        print("[ok] render loop finished.")
        print("[render] GPU scheduler summary:\n" + sched.summary())
    finally:
        if a.shutdown:
            _auto_shutdown()

if __name__ == "__main__":
    main()
