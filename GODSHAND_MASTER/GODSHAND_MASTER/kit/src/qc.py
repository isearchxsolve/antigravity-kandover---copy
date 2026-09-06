#!/usr/bin/env python3
"""qc.py -- objective QUALITY CHECK on rendered output before you publish/monetize.

Per shot it reports resolution, fps, duration, audio loudness (LUFS), black-frame and
freeze detection, and cv2 sharpness/flicker, then a PASS / WARN / FAIL verdict.
Exit code is non-zero if ANY shot FAILs, so you can gate a publish step on it.

Thresholds come from the `qc:` section of the active config (calibration = edit config,
no code change). RELIABLE checks: resolution / duration / fps / black / freeze / loudness.
ROUGH TRIAGE ONLY: sharpness and flicker (uncalibrated until tuned on real frames).

    python3 src/qc.py            # checks work/<id>/final.mp4 for every shot in config
    python3 src/qc.py --clips    # checks output/clips/<id>/<id>_master.mp4 instead
"""
import os, sys, json, re, subprocess, argparse

DEFAULT_TH = {"min_w": 640, "min_h": 360, "dur_tol": 0.75, "fps_tol": 1.5,
               "sharp_min": 15, "flicker_max": 12, "lufs_hi": -9, "lufs_lo": -30,
               "black_pct_max": 2.0, "freeze_dur_max": 0.6, "color_var_min": 5.0,
               "silence_dur_max": 2.0}

def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", path], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}

def parse_fps(s):
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0

def loudness(path):
    r = subprocess.run(["ffmpeg", "-i", path, "-af", "loudnorm=print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr)
    if m:
        try:
            return float(json.loads(m.group(0))["input_i"])
        except Exception:
            return None
    return None

def black_freeze(path):
    r = subprocess.run(["ffmpeg", "-i", path, "-vf",
                        "blackdetect=d=0.3:pic_th=0.98,freezedetect=n=0.003:d=0.5",
                        "-an", "-f", "null", "-"], capture_output=True, text=True)
    return len(re.findall(r"black_start", r.stderr)), len(re.findall(r"freeze_start", r.stderr))

def silence_detect(path):
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                         "silencedetect=n=-50dB:d=2.0", "-f", "null", "-"],
                        capture_output=True, text=True)
    return len(re.findall(r"silence_start", r.stderr))

def color_variance(path):
    try:
        import cv2, numpy as np
    except Exception:
        return None
    cap = cv2.VideoCapture(path)
    variances = []
    prev = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype("float32")
        if prev is not None:
            variances.append(float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    return float(np.mean(variances)) if variances else None

def encode_check(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
                          capture_output=True, text=True)
    return [l for l in r.stderr.splitlines() if any(k in l for k in
              ["error", "failed", "corrupt", "broken", "Invalid"])]


def black_lead_tail(path, edge_frames=3, luma_thresh=8.0):
    """FAIL if the FIRST or LAST few frames are near-solid black. A real film
    should fade in/out -- pure-black leads/tails indicate a failed encoder or
    an empty shot. Returns (lead_black, tail_black) booleans."""
    try:
        import cv2, numpy as np
    except Exception:
        return False, False
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return True, True
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total < edge_frames * 2:
        cap.release()
        return True, True
    lead_blacks, tail_blacks = 0, 0
    for i in range(edge_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            lum = float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean())
            if lum <= luma_thresh:
                lead_blacks += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1 - i)
        ok, fr = cap.read()
        if ok:
            lum = float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean())
            if lum <= luma_thresh:
                tail_blacks += 1
    cap.release()
    return lead_blacks > 0, tail_blacks > 0


def histogram_legal(path, luma_lo=16, luma_hi=235, chroma_lo=16, chroma_hi=240):
    """Rec.709 broadcast-legal limited-range check. Samples 8 evenly-spaced
    frames and computes the % of pixels that fall outside legal levels. A
    properly-mastered Rec.709 delivery has <1% out-of-gamut pixels in static
    frames; renders with crushed blacks / blown highlights exceed this.
    Returns pct_out_of_legal (float) or None if cv2 unavailable."""
    try:
        import cv2, numpy as np
    except Exception:
        return None
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total < 8:
        cap.release()
        return None
    samples = [int(total * f / 8.0) for f in range(8)]
    bad = 0; tot = 0
    for fi in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        ycrcb = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[..., 0].astype("int32")
        cr = ycrcb[..., 1].astype("int32")
        cb = ycrcb[..., 2].astype("int32")
        luma_bad = ((y < luma_lo) | (y > luma_hi))
        chroma_bad = ((cr < chroma_lo) | (cr > chroma_hi)) | ((cb < chroma_lo) | (cb > chroma_hi))
        bad += int(luma_bad.sum() + chroma_bad.sum())
        tot += int(y.size + cr.size + cb.size)
    cap.release()
    return (bad / tot) if tot else 0.0


def frame_count_check(path, expect_dur, fps):
    """Returns (actual_frames, expected_frames, drift_pct). Drift > 0.5% is FAIL."""
    try:
        import cv2
    except Exception:
        return None, None, None
    cap = cv2.VideoCapture(path)
    actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if not (expect_dur and fps) or actual == 0:
        return actual, None, None
    expected = int(round(expect_dur * fps))
    drift = abs(actual - expected) / max(expected, 1) * 100.0
    return actual, expected, drift


def codec_check(path, expect_pix_fmt=None, expect_codec=None):
    """Returns dict {pix_fmt, codec_name, color_space, color_primaries,
    color_transfer} from ffprobe; flags mismatches with the expected values
    when provided."""
    info = ffprobe(path)
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        return {}
    v = vs[0]
    out = {
        "pix_fmt": v.get("pix_fmt", ""),
        "codec_name": v.get("codec_name", ""),
        "color_space": v.get("color_space", ""),
        "color_primaries": v.get("color_primaries", ""),
        "color_transfer": v.get("color_transfer", ""),
    }
    if expect_pix_fmt and out["pix_fmt"] != expect_pix_fmt:
        out["pix_fmt_mismatch"] = (out["pix_fmt"], expect_pix_fmt)
    if expect_codec and out["codec_name"] != expect_codec:
        out["codec_mismatch"] = (out["codec_name"], expect_codec)
    return out


def audio_sync_check(path, expect_dur, drift_sec=0.25):
    """Compare audio stream duration vs container duration. Drift > drift_sec
    means voice-over is desynced from the picture -- a clear amateur tell.
    Returns dict {a_dur, v_dur, drift} or None if no audio."""
    info = ffprobe(path)
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    aud = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not (vs and aud):
        return None
    v_dur = float(info.get("format", {}).get("duration", 0) or 0)
    a_dur = float(aud[0].get("duration", 0) or 0)
    if a_dur <= 0:
        return None
    drift = abs(v_dur - a_dur)
    return {"a_dur": round(a_dur, 3), "v_dur": round(v_dur, 3),
            "drift": round(drift, 3), "in_sync": drift <= drift_sec,
            "expect_dur": round(expect_dur, 3) if expect_dur else None,
            "dur_drift": round(abs(v_dur - expect_dur), 3) if expect_dur else None}

def cv_metrics(path):
    try:
        import cv2, numpy as np
    except Exception:
        return None
    cap = cv2.VideoCapture(path)
    grays, sharp = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(cv2.medianBlur(g, 3), cv2.CV_64F).var())
        grays.append(g.astype("float32"))
    cap.release()
    if not grays:
        return None
    jerks = [float(np.mean(np.abs(grays[i] - 2 * grays[i - 1] + grays[i - 2])))
             for i in range(2, len(grays))]
    return {"sharpness": float(np.mean(sharp)), "flicker": float(np.mean(jerks)) if jerks else 0.0}

def _worse(cur, new):
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return new if order[new] > order[cur] else cur

def check(path, expect_dur, fps_target, th, cfg=None):
    if not os.path.exists(path):
        return "FAIL", ["missing file"], {}
    info = ffprobe(path)
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    aud = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not vs:
        return "FAIL", ["no video stream"], {}
    v = vs[0]
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    fps = parse_fps(v.get("r_frame_rate", "0/1"))
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    stats = {"res": f"{w}x{h}", "fps": round(fps, 1), "dur": round(dur, 2), "audio": bool(aud)}
    verdict, issues = "PASS", []
    if w < th["min_w"] or h < th["min_h"]:
        issues.append(f"low res {w}x{h}"); verdict = _worse(verdict, "WARN")
    if expect_dur and abs(dur - expect_dur) > th["dur_tol"]:
        issues.append(f"dur {dur:.1f}s vs ~{expect_dur}s"); verdict = _worse(verdict, "WARN")
    if fps_target and abs(fps - fps_target) > th["fps_tol"]:
        issues.append(f"fps {fps:.1f} vs {fps_target}"); verdict = _worse(verdict, "WARN")
    b, fz = black_freeze(path)
    if b:
        issues.append(f"{b} black segment(s)"); verdict = _worse(verdict, "FAIL")
    if fz:
        issues.append(f"{fz} freeze(s)"); verdict = _worse(verdict, "WARN")
    m = cv_metrics(path)
    if m:
        stats["sharp"] = round(m["sharpness"], 1); stats["flicker"] = round(m["flicker"], 2)
        if m["sharpness"] < th["sharp_min"]:
            issues.append("very soft/blurry"); verdict = _worse(verdict, "WARN")
        if m["flicker"] > th["flicker_max"]:
            issues.append("high flicker"); verdict = _worse(verdict, "WARN")
    if aud:
        lu = loudness(path)
        if lu is not None:
            stats["LUFS"] = round(lu, 1)
            if lu > th["lufs_hi"] or lu < th["lufs_lo"]:
                # Surface the configured bus target (not a hard-coded -16 literal);
                # falls back to config audio_master or -16 only when not set.
                _am = (cfg or {}).get("audio_master", {}) or {}
                _tgt = _am.get("loudness_target", -16)
                issues.append(f"loudness {lu:.1f} LUFS off {_tgt:g} target"); verdict = _worse(verdict, "WARN")
        sil = silence_detect(path)
        if sil:
            issues.append(f"{sil} silence event(s)"); verdict = _worse(verdict, "WARN")
    else:
        issues.append("no audio"); verdict = _worse(verdict, "WARN")
    cv = color_variance(path)
    if cv is not None:
        stats["color_var"] = round(cv, 2)
        if cv < th.get("color_var_min", 5.0):
            issues.append(f"low color variance ({cv:.1f}) possible color cast"); verdict = _worse(verdict, "WARN")
    enc_errors = encode_check(path)
    if enc_errors:
        issues.append(f"encode errors: {len(enc_errors)}"); verdict = _worse(verdict, "FAIL")

    lead_black, tail_black = black_lead_tail(path)
    if lead_black or tail_black:
        issues.append(f"black {'lead' if lead_black else ''}"
                      f"{'/' if lead_black and tail_black else ''}"
                      f"{'tail' if tail_black else ''} frame (failed encode?)")
        verdict = _worse(verdict, "FAIL")

    actual_fc, expected_fc, fc_drift = frame_count_check(path, expect_dur, fps_target)
    if actual_fc is not None and expected_fc is not None and fc_drift is not None:
        stats["frames"] = f"{actual_fc}/{expected_fc}"
        if fc_drift > 0.5:
            issues.append(f"frame count drift {fc_drift:.2f}% ({actual_fc}/{expected_fc})")
            verdict = _worse(verdict, "FAIL")

    legal = histogram_legal(path)
    if legal is not None:
        stats["legal_pct"] = round(100.0 - legal * 100.0, 2)
        if legal > 0.05:
            issues.append(f"broadcast-legal {100*(1-legal):.2f}% "
                          f"({legal*100:.2f}% pixels out-of-gamut)")
            verdict = _worse(verdict, "WARN")

    delivery = (cfg or {}).get("delivery", {}) or {}
    expect_pix = delivery.get("pix_fmt") if (delivery.get("enabled") or
                                              os.environ.get("KIT_4K_DELIVERY") == "1") else None
    codec = codec_check(path, expect_pix_fmt=expect_pix)
    if codec:
        stats["pix"] = codec.get("pix_fmt", "?")
        if "pix_fmt_mismatch" in codec:
            issues.append(f"pix_fmt {codec['pix_fmt_mismatch'][0]} != "
                          f"expected {codec['pix_fmt_mismatch'][1]}")
            verdict = _worse(verdict, "FAIL")

    sync = audio_sync_check(path, expect_dur)
    if sync is not None:
        stats["a_sync"] = "ok" if sync["in_sync"] else "DRIFT"
        if not sync["in_sync"]:
            issues.append(f"audio drift {sync['drift']:.3f}s vs video")
            verdict = _worse(verdict, "FAIL")

    return verdict, issues, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", action="store_true", help="check output/clips masters instead of work clips")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    th = dict(DEFAULT_TH); th.update(cfg.get("qc", {}) or {})
    # 4K delivery mode lifts QC minimums so a HD-render + 4K-master chain
    # can be QC-gated at the delivered target resolution. KIT_4K_DELIVERY env
    # or config 'delivery.enabled' triggers the lift; the `qc:` block still
    # wins if it explicitly sets a higher/lower threshold.
    d = cfg.get("delivery", {}) or {}
    if d.get("enabled", False) or os.environ.get("KIT_4K_DELIVERY") == "1":
        # Only raise the floor if qc overrides did NOT pin it lower/higher.
        qc_override = cfg.get("qc", {}) or {}
        if "min_w" not in qc_override:
            th["min_w"] = max(th["min_w"], 3840)
        if "min_h" not in qc_override:
            th["min_h"] = max(th["min_h"], 2160)
        # At 4K24 cinema, fps tolerance is tight: 24.0 +/- 0.5 (no 23.976->24 mismatch drift).
        if "fps_tol" not in qc_override:
            th["fps_tol"] = min(th["fps_tol"], 0.5)
        print(f"[qc] KIT_4K_DELIVERY=1 active -> min_w>={th['min_w']}, "
              f"min_h>={th['min_h']}, fps_tol<={th['fps_tol']}")
    work = cfg["paths"]["work"]
    out = os.path.join(root, cfg["paths"]["out"])
    fps_target = cfg.get("interpolate_fps", cfg.get("fps", 24))
    shots = json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]
    fails = warns = passes = 0
    for s in shots:
        sid = s["id"]
        path = (os.path.join(out, "clips", sid, f"{sid}_master.mp4") if a.clips
                else os.path.join(work, sid, "final.mp4"))
        verdict, issues, stats = check(path, s.get("dur", 0), fps_target, th, cfg)
        fails += verdict == "FAIL"; warns += verdict == "WARN"; passes += verdict == "PASS"
        print(f"[{verdict}] {sid}  {stats}  {'; '.join(issues) if issues else 'ok'}")
    print(f"\n[qc] {len(shots)} checked: {passes} PASS, {warns} WARN, {fails} FAIL")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
