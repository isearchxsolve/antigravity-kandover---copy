#!/usr/bin/env python3
"""selftest.py -- shrink CPU debug cycles by catching the MAXIMUM number of errors per run.

Instead of failing at the first error (one fix per cycle), this collects EVERY static
problem in a single pass, then runs the whole non-GPU pipeline against synthetic media
(COMFY_MOCK=1 + POST_MODE=cpu) and finishes with a qc.py quality gate.

  python3 src/selftest.py --quick   # ~1s, static only: compile + graph/_bind + config/shots + deps
  python3 src/selftest.py --fast    # full mock pipeline, KIT_FAST_POST=1 skips slow interpolation
  python3 src/selftest.py           # full mock pipeline + cpu post + clips + stitch + qc

Exit 0 = green. Non-zero = a list of bugs you can fix for FREE right here, all at once,
before spending a rupee on a GPU.
"""
import os, sys, json, glob, shutil, subprocess, argparse, py_compile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


# ---------------------------------------------------------------- static checks
def quick_checks():
    """Return a list of ALL static failures (never stops at the first)."""
    fails = []

    # 1) every source file compiles
    for fn in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        try:
            py_compile.compile(fn, doraise=True)
        except Exception as e:
            fails.append(f"compile {os.path.basename(fn)}: {e}")

    # 2) python deps present
    for mod in ("cv2", "numpy", "yaml", "PIL"):
        try:
            __import__(mod)
        except Exception as e:
            fails.append(f"missing dependency {mod}: {e}")

    # 3) comfy_client imports and points at the real ComfyUI address
    try:
        sys.path.insert(0, SRC)
        import comfy_client
        c = comfy_client.ComfyClient()
        base = str(getattr(c, "base", getattr(c, "base_url", "")))
        if "127.0.0.1:8188" not in base and "localhost:8188" not in base:
            fails.append(f"comfy_client base URL unexpected: {base!r}")
    except Exception as e:
        fails.append(f"comfy_client import/init: {e}")

    # 4) every workflow graph parses and every _bind target resolves to a real node input
    wfdir = os.path.join(ROOT, "workflows")
    for fn in sorted(glob.glob(os.path.join(wfdir, "*.json"))):
        name = os.path.basename(fn)
        try:
            g = json.load(open(fn))
        except Exception as e:
            fails.append(f"workflow {name}: JSON parse error: {e}"); continue
        for key, tgt in (g.get("_bind", {}) or {}).items():
            if not (isinstance(tgt, (list, tuple)) and len(tgt) == 2):
                fails.append(f"workflow {name}: _bind[{key}] malformed: {tgt!r}"); continue
            nid, inp = str(tgt[0]), tgt[1]
            node = g.get(nid)
            if not isinstance(node, dict):
                fails.append(f"workflow {name}: _bind[{key}] -> node {nid} missing"); continue
            if inp not in (node.get("inputs", {}) or {}):
                fails.append(f"workflow {name}: _bind[{key}] -> {nid}.{inp} is not an input of that node")

    # 5) every config + shot list parses
    import yaml
    for fn in sorted(glob.glob(os.path.join(ROOT, "config*.yaml"))):
        try:
            yaml.safe_load(open(fn))
        except Exception as e:
            fails.append(f"config {os.path.basename(fn)}: YAML error: {e}")
    for fn in sorted(glob.glob(os.path.join(ROOT, "shots_*.json"))):
        try:
            d = json.load(open(fn))
            # a valid shot list has either a flat shots[] array or a scenes[]/beats[] tree
            if not (isinstance(d.get("shots"), list) or isinstance(d.get("scenes"), list)):
                fails.append(f"shots {os.path.basename(fn)}: no 'shots' or 'scenes' array")
        except Exception as e:
            fails.append(f"shots {os.path.basename(fn)}: JSON error: {e}")

    return fails


# ---------------------------------------------------------------- pipeline run
def run(cmd, env, label):
    """Run a stage; print tail; return True on success (never raises)."""
    print(f"\n=== {label} ===", flush=True)
    p = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write((p.stdout or "")[-3000:])
    if p.returncode != 0:
        sys.stdout.write("\n--- STDERR ---\n" + (p.stderr or "")[-3000:])
        print(f"\n[stage FAILED] {label} (exit {p.returncode})")
        return False
    return True


def pipeline(fast):
    work = "/tmp/kit_selftest_work"
    outname = "output_selftest"
    out = os.path.join(ROOT, outname)
    models = "/tmp/kit_selftest_models"
    audio_bg = "/tmp/kit_selftest_audio"
    for d in (work, out, models, audio_bg):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
    open(os.path.join(models, "DUMMY.txt"), "w").write("stub so preflight passes\n")

    shots = {"title": "SELFTEST", "globals": {"tone": "test tone", "negative_prompt": "blurry, watermark"},
             "shots": [
                 {"id": "t01", "dur": 2, "chars": [], "talking": False, "prompt": "a test shot", "palette": "grey"},
                 {"id": "t02", "dur": 2, "chars": [], "talking": False, "prompt": "another test shot", "palette": "blue"}]}
    shots_path = os.path.join(ROOT, "shots_selftest.json")
    json.dump(shots, open(shots_path, "w"))

    cfg = {
        "film_title": "SELFTEST", "shots_file": "shots_selftest.json",
        "fps": 24, "interpolate_fps": 48, "width": 256, "height": 144,
        "steps": 4, "cfg": 5.0, "upscale": 1.0, "rejection_samples": 2,
        "codeformer_fidelity": 0.7, "max_clip_seconds": 5,
        "paths": {"refs": "refs", "audio_bg": audio_bg, "work": work, "out": outname, "models": models},
        "score": {"duck_db": 18, "crossfade_ms": 400, "movements": {
            "m1": {"file": "none1.mp3", "shots": ["t01"]},
            "m2": {"file": "none2.mp3", "shots": ["t02"]}}},
        "clips": {"horizontal": {"w": 640, "h": 360}, "vertical": {"w": 360, "h": 640},
                  "thumbnail_at": 0.5, "music_under_clip": True, "ai_label": True, "creator": "Selftest"},
        "rate_per_gpu_hr": 0.0,
    }
    import yaml
    cfg_path = os.path.join(ROOT, "config_selftest.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    env = dict(os.environ)
    env.update({"COMFY_MOCK": "1", "POST_MODE": "cpu", "KIT_CONFIG": "config_selftest.yaml"})
    if fast:
        env["KIT_FAST_POST"] = "1"

    fails = []
    try:
        if not run([sys.executable, "src/preflight.py"], env, "preflight (cpu/mock)"):
            fails.append("preflight")
        if run([sys.executable, "src/render_film.py"], env, "render_film (mock generate + cpu post)"):
            for sid in ("t01", "t02"):
                if not os.path.exists(os.path.join(work, sid, "final.mp4")):
                    fails.append(f"render: missing {sid}/final.mp4")
        else:
            fails.append("render_film")

        if not fails and run([sys.executable, "src/clips.py"], env, "clips.py (master + vertical + thumb + manifest)"):
            for sid in ("t01", "t02"):
                base = os.path.join(out, "clips", sid)
                for f in (f"{sid}_master.mp4", f"{sid}_vertical.mp4", f"{sid}_thumb.jpg", f"{sid}_manifest.json"):
                    if not os.path.exists(os.path.join(base, f)):
                        fails.append(f"clips: missing {sid}/{f}")
        elif not fails:
            fails.append("clips.py")

        if not fails and run([sys.executable, "src/stitch.py", "--root", ROOT], env, "stitch.py (concat + score + mux + master)"):
            if not os.path.exists(os.path.join(out, "SELFTEST.mp4")):
                fails.append("stitch: missing SELFTEST.mp4")
        elif not fails:
            fails.append("stitch.py")

        # quality gate on the mock output (should PASS video; no-audio -> WARN, not FAIL)
        if not fails:
            run([sys.executable, "src/qc.py"], env, "qc.py (quality gate)")
    finally:
        for p in (shots_path, cfg_path):
            try:
                os.remove(p)
            except OSError:
                pass
        for d in (out, work, models, audio_bg):
            shutil.rmtree(d, ignore_errors=True)
    return fails


# ---------------------------------------------------------------- motion-path run
# This exercises the *production* render path that the Kaggle notebook invokes
# (render_motion.py), as opposed to pipeline()/render_film.py (the legacy path
# covered by --fast and the default mode). It catches the failure mode the
# original Defect #1 in GODSHAND_FULL_AUDIT_AND_FIXES_REPORT.md described: the
# indentation catastrophe that orphaned render_motion()'s render loop and made
# every shot exit silently with no final.mp4 produced. selftest --fast stayed
# green throughout because it uses render_film.py, not render_motion.py.
def motion_pipeline(fast):
    work = "/tmp/kit_selftest_work_motion"
    outname = "output_selftest_motion"
    out = os.path.join(ROOT, outname)
    models = "/tmp/kit_selftest_models_motion"
    audio_bg = "/tmp/kit_selftest_audio_motion"
    for d in (work, out, models, audio_bg):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
    open(os.path.join(models, "DUMMY.txt"), "w").write("stub so preflight passes\n")

    # A single minimal shot; render_motion.py --smoke reduces rejection_samples=1
    # and caps steps at 12. The shot id "m01" is intentionally distinct from the
    # legacy t01/t02 used by pipeline() so the two paths never share work dirs.
    shots = {"title": "SELFTEST_MOTION", "globals": {"tone": "test", "negative_prompt": "blurry"},
             "shots": [
                 {"id": "m01", "dur": 2, "chars": [], "talking": False, "prompt": "a motion test shot", "palette": "grey"}]}
    shots_path = os.path.join(ROOT, "shots_selftest_motion.json")
    json.dump(shots, open(shots_path, "w"))

    cfg = {
        "film_title": "SELFTEST_MOTION", "shots_file": "shots_selftest_motion.json",
        "fps": 24, "interpolate_fps": 16, "width": 256, "height": 144,
        "steps": 4, "cfg": 5.0, "upscale": 1.0, "rejection_samples": 2,
        "codeformer_fidelity": 0.7, "max_clip_seconds": 5,
        "paths": {"refs": "refs", "audio_bg": audio_bg, "work": work, "out": outname, "models": models},
        "rate_per_gpu_hr": 0.0,
    }
    import yaml
    cfg_path = os.path.join(ROOT, "config_selftest_motion.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    env = dict(os.environ)
    env.update({"COMFY_MOCK": "1", "POST_MODE": "cpu", "KIT_CONFIG": "config_selftest_motion.yaml"})
    if fast:
        env["KIT_FAST_POST"] = "1"

    fails = []
    try:
        # render_motion.py --smoke already drops rejection_samples to 1 and caps
        # steps at 12; --only m01 is belt-and-suspenders to never render more than
        # our single fixture shot even if KIT_SHARD shape shifts.
        cmd = [sys.executable, "src/render_motion.py", "--smoke", "--only", "m01"]
        if not run(cmd, env, "render_motion (production path, mock + cpu post, shot m01)"):
            fails.append("render_motion")
        else:
            expected = os.path.join(work, "m01", "final.mp4")
            if not os.path.exists(expected):
                fails.append(f"render_motion: missing {expected} "
                             f"(silently returned without producing final.mp4 -- this is the "
                             f"Defect-#1 signature; verify the render_motion() body is contiguous)")
    finally:
        for p in (shots_path, cfg_path):
            try:
                os.remove(p)
            except OSError:
                pass
        for d in (out, work, models, audio_bg):
            shutil.rmtree(d, ignore_errors=True)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="static checks only (~1s, no ffmpeg)")
    ap.add_argument("--fast", action="store_true", help="full pipeline but skip slow interpolation")
    ap.add_argument("--motion", action="store_true",
                    help="exercise the production render path (render_motion.py --smoke) "
                         "instead of the legacy render_film.py path. Use this to catch "
                         "regressions in notebook-path rendering; this is the path the "
                         "Kaggle notebook actually executes.")
    ap.add_argument("--logs", action="append", default=[], metavar="PATH",
                    help="scan Kaggle/ComfyUI runtime log files for defect "
                         "signatures (Invalid PNG / Killed / CUDA OOM / VHS "
                         "warnings / runaway repeated messages / per-prompt "
                         "timing regressions). Repeatable. Exits 1 if any "
                         "FATAL signature is found. Does NOT run static or "
                         "pipeline checks.")
    a = ap.parse_args()

    if a.logs:
        print("=== selftest --logs (Kaggle/ComfyUI log-severity scan) ===")
        import kaggle_log_validator as klv
        any_fatal = False
        for path in a.logs:
            fnd = klv.scan_path(path)
            print(f"\n=== {path} ===")
            for f in fnd:
                marker = "FATAL" if f["severity"] == "FATAL" else "WARN "
                mult = f"({f['count']}x)" if f.get("count", 1) > 1 else ""
                print(f"  [{marker}] L{f['line']:>5} {mult} {f['id']}")
            if any(f["severity"] == "FATAL" for f in fnd):
                any_fatal = True
        if any_fatal:
            raise SystemExit("\n[selftest --logs] FAIL -- FATAL findings above.")
        print("\n[selftest --logs] OK -- no FATAL findings.")
        return

    print("=== static checks (compile + deps + comfy_client + workflows + configs/shots) ===")
    static = quick_checks()
    if static:
        print(f"[static] {len(static)} issue(s):")
        for f in static:
            print(f"   x {f}")
    else:
        print("[static] OK -- all sources compile, deps present, graphs/_bind valid, configs/shots parse.")

    if a.quick:
        if static:
            raise SystemExit(f"\n[selftest --quick] {len(static)} static issue(s) above.")
        print("\n[selftest --quick] ALL GREEN (static).")
        return

    # Static errors block the pipeline anyway -- report them all in this one cycle.
    if static:
        raise SystemExit(f"\n[selftest] {len(static)} static issue(s) above -- fix these first (pipeline skipped).")

    # --motion: exercise the production render path (render_motion.py) instead
    # of the legacy render_film.py path. This catches the failure mode the
    # original audit's Defect #1 introduced (truncated render_motion() body ->
    # every shot returns silently with no final.mp4) which --fast/full could not
    # detect because they use render_film.py.
    if a.motion:
        fails = motion_pipeline(fast=a.fast)
        if fails:
            print(f"\n[selftest --motion] {len(fails)} pipeline issue(s):")
            for f in fails:
                print(f"   x {f}")
            raise SystemExit(1)
        mode = "fast" if a.fast else "full"
        print(f"\n[selftest --motion] ALL GREEN ({mode}) "
              f"-- render_motion.py (the production/notebook path) runs end to end.")
        return

    fails = pipeline(fast=a.fast)
    if fails:
        print(f"\n[selftest] {len(fails)} pipeline issue(s):")
        for f in fails:
            print(f"   x {f}")
        raise SystemExit(1)
    mode = "fast" if a.fast else "full"
    print(f"\n[selftest] ALL GREEN ({mode}) -- the full non-GPU pipeline runs end to end.")


if __name__ == "__main__":
    main()
