#!/usr/bin/env python3
"""clips.py -- package individual shots as MONETIZATION-READY standalone clips.

Runs on CPU (pure ffmpeg) -- do it FREE on a Lightning CPU Studio / laptop.
For each shot's finished work/<id>/final.mp4 it writes, under output/clips/<id>/:

  <id>_master.mp4     16:9 mastered clip (per-clip audio: music bed ducked under voice)
  <id>_vertical.mp4   9:16 center-crop for Shorts / Reels / TikTok
  <id>_thumb.jpg      thumbnail frame
  <id>_manifest.json  C2PA-style content-credentials manifest (AI-generated disclosure)

Both MP4s carry AI-generated + creator metadata. If `c2patool` is installed the
clips are cryptographically signed with the manifest; otherwise you get the
embedded metadata + sidecar manifest (install c2patool to sign).

Usage:
  python3 src/clips.py                # every shot that has a rendered final.mp4
  python3 src/clips.py --only b05_doctor
  python3 src/clips.py --no-music     # voice/silent only, skip music bed

REMINDER: metadata/C2PA is best practice but you STILL must toggle each platform's
AI-disclosure (e.g. YouTube 'altered or synthetic content') at upload time.
"""
import os, sys, json, subprocess, argparse, shutil

def sh(*a): subprocess.run(list(a), check=True)
def has(cmd): return shutil.which(cmd) is not None


def delivery_params(cfg):
    """Resolve the active delivery specification. Honor KIT_4K_DELIVERY env
    (wins over config['delivery']['enabled']). Returns a dict with keys
    pix_fmt, colorspace, color_primaries, color_trc, video_bitrate, audio_bitrate,
    master_crf, width, height, enabled, force_4k_scale.

    When not in 4K delivery mode, this returns the existing 8-bit CRF-based
    behaviour so render/stitch behaviour is unchanged for HD runs.
    """
    d = (cfg or {}).get("delivery", {}) or {}
    enabled = bool(d.get("enabled", False)) or os.environ.get("KIT_4K_DELIVERY") == "1"
    if not enabled:
        return {
            "enabled": False,
            "pix_fmt": "yuv420p",
            "colorspace": "bt709", "color_primaries": "bt709", "color_trc": "bt709",
            "video_bitrate": None, "audio_bitrate": "192k",
            "master_crf": 16, "vertical_crf": 16,   # vertical was 18 -> 16 (fix)
            "width": None, "height": None,          # use clips.horizontal/.vertical
        }
    # 4K delivery: 10-bit + bitrate floor
    return {
        "enabled": True,
        "pix_fmt": d.get("pix_fmt", "yuv420p10le"),
        "colorspace": d.get("colorspace", "bt709"),
        "color_primaries": d.get("color_primaries", "bt709"),
        "color_trc": d.get("color_trc", "bt709"),
        "video_bitrate": d.get("video_bitrate", "120M"),
        "audio_bitrate": d.get("audio_bitrate", "320k"),
        "master_crf": d.get("master_crf"),         # None -> bitrate mode
        "vertical_crf": d.get("master_crf"),        # vertical uses same bitrate mode in 4K
        "width": int(d.get("width", 3840)),
        "height": int(d.get("height", 2160)),
    }


def _color_flags(d):
    return ["-colorspace", d["colorspace"],
            "-color_primaries", d["color_primaries"],
            "-color_trc", d["color_trc"]]


def _master_vcodec(d):
    """Build the libx264 vcodec arg list honoring 4K bitrate/CRF mode."""
    args = ["-c:v", "libx264", "-preset", "slow"]
    if d["enabled"] and d["video_bitrate"]:
        args += ["-b:v", d["video_bitrate"], "-pix_fmt", d["pix_fmt"]]
    else:
        args += ["-crf", str(d.get("master_crf") or 16), "-pix_fmt", d["pix_fmt"]]
    args += _color_flags(d)
    return args

def ffprobe_dur(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", p], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except Exception:
        return 0.0

def movement_music(cfg, root, sid):
    mv = (cfg.get("score", {}) or {}).get("movements", {}) or {}
    bg = os.path.join(root, cfg["paths"]["audio_bg"])
    for _, spec in mv.items():
        if sid in spec.get("shots", []):
            p = os.path.join(bg, spec["file"])
            return p if os.path.exists(p) else None
    return None

def build_audio(cfg, root, sid, work, dur, use_music):
    """Per-clip audio: instrumental music bed ducked under the shot's voice. Cascading
    fallbacks so a filter failure never blocks the clip -- returns a path or None."""
    voice = os.path.join(work, "voice", f"{sid}.wav")
    have_voice = os.path.exists(voice)
    music = movement_music(cfg, root, sid) if use_music else None
    outp = os.path.join(work, sid, "_clip_audio.m4a")
    # Loudness target: read from config audio_master (master bus). Default -16 only
    # when the config key is missing -- never override a configured target.
    am = (cfg.get("audio_master", {}) or {})
    lufs_i = float(am.get("loudness_target", -16))
    lufs_tp = float(am.get("true_peak", -1.5))
    lufs_lra = float(am.get("loudness_range", 11))
    audio_bitrate = "192k"
    if have_voice and music:
        try:
            sh("ffmpeg", "-y", "-loglevel", "error", "-i", music, "-i", voice, "-filter_complex",
               f"[0:a]atrim=0:{dur:.2f},volume=-6dB[m];[1:a]atrim=0:{dur:.2f},asplit=2[v][vk];"
               f"[m][vk]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=400[md];"
               f"[md][v]amix=inputs=2:normalize=0,"
               f"loudnorm=I={lufs_i:g}:TP={lufs_tp:g}:LRA={lufs_lra:g}[a]",
               "-map", "[a]", "-c:a", "aac", "-b:a", audio_bitrate, outp)
            return outp
        except subprocess.CalledProcessError:
            pass
    if have_voice:
        try:
            sh("ffmpeg", "-y", "-loglevel", "error", "-i", voice, "-t", f"{dur:.2f}",
               "-c:a", "aac", "-b:a", audio_bitrate, outp)
            return outp
        except subprocess.CalledProcessError:
            pass
    if music:
        try:
            sh("ffmpeg", "-y", "-loglevel", "error", "-i", music, "-t", f"{dur:.2f}",
               "-af", "volume=-6dB", "-c:a", "aac", "-b:a", audio_bitrate, outp)
            return outp
        except subprocess.CalledProcessError:
            pass
    return None

# Legacy COLOR constant retained for older callers/tests reading this module attr.
COLOR = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]

def export_clip(cfg, root, sid, work, out_root, use_music):
    src = os.path.join(work, sid, "final.mp4")
    if not os.path.exists(src):
        print(f"[skip] {sid}: no rendered final.mp4")
        return False
    outdir = os.path.join(out_root, "clips", sid)
    os.makedirs(outdir, exist_ok=True)
    dur = ffprobe_dur(src)
    cl = cfg.get("clips", {}) or {}
    H = cl.get("horizontal", {"w": 1920, "h": 1080})
    V = cl.get("vertical", {"w": 1080, "h": 1920})
    creator = cl.get("creator", "")
    meta = ["-metadata", "comment=AI-generated video (Wan2.2). Altered/synthetic content.",
            "-metadata", f"artist={creator}",
            "-metadata", "description=AI-generated / altered synthetic content. Disclose at upload."]
    try:
        aud = build_audio(cfg, root, sid, work, dur, use_music)
    except Exception as e:
        print(f"[warn] {sid}: audio build failed ({e}); exporting silent.")
        aud = None

    d = delivery_params(cfg)
    # 4K delivery overrides clip dimensions (the master is upscaled to 4K).
    if d["enabled"] and d["width"] and d["height"]:
        H = {"w": d["width"], "h": d["height"]}

    audio_bitrate = d["audio_bitrate"]

    # 1) 16:9 master (letterbox-safe scale+pad, high quality)
    master = os.path.join(outdir, f"{sid}_master.mp4")
    vf_h = (f"scale={H['w']}:{H['h']}:force_original_aspect_ratio=decrease,"
            f"pad={H['w']}:{H['h']}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    if aud:
        base += ["-i", aud]
    base += ["-vf", vf_h, "-map", "0:v"]
    if aud:
        base += ["-map", "1:a", "-c:a", "aac", "-b:a", audio_bitrate, "-shortest"]
    base += _master_vcodec(d) + meta + [master]
    sh(*base)

    # 2) 9:16 vertical center-crop (Shorts/Reels/TikTok) from the master
    vert = os.path.join(outdir, f"{sid}_vertical.mp4")
    vf_v = f"crop=trunc(ih*{V['w']}/{V['h']}/2)*2:ih,scale={V['w']}:{V['h']},setsar=1"
    vcmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", master, "-vf", vf_v,
            "-c:v", "libx264", "-preset", "slow"]
    # Vertical: copy master's bitrate/CRF mode (was hardcoded CRF 18; now matches master mode).
    if d["enabled"] and d["video_bitrate"]:
        # 4K delivery: vertical cut uses bitrate mode with a slightly lower floor
        # (vertical is half the pixel count of the 16:9 master at any target res).
        vcmd += ["-b:v", d["video_bitrate"], "-pix_fmt", d["pix_fmt"]]
    else:
        vcmd += ["-crf", str(d.get("vertical_crf") or 16), "-pix_fmt", d["pix_fmt"]]
    vcmd += _color_flags(d)
    vcmd += (["-c:a", "copy"] if aud else ["-an"]) + meta + [vert]
    sh(*vcmd)

    # 3) thumbnail
    thumb = os.path.join(outdir, f"{sid}_thumb.jpg")
    t = max(0.0, dur * float(cl.get("thumbnail_at", 0.5)))
    sh("ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", master,
       "-frames:v", "1", "-q:v", "2", thumb)

    # 4) C2PA-style content-credentials manifest (+ sign if c2patool present)
    manifest = {
        "claim_generator": "film_kit_v3/clips.py",
        "title": sid,
        "assertions": [
            {"label": "c2pa.actions",
             "data": {"actions": [{"action": "c2pa.created",
                                   "softwareAgent": "Wan2.2 + ComfyUI",
                                   "digitalSourceType":
                                   "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}]}},
            {"label": "org.aiLabel",
             "data": {"aiGenerated": True,
                      "disclosure": "AI-generated / altered synthetic content"}},
        ],
        "creator": creator,
    }
    mpath = os.path.join(outdir, f"{sid}_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2)
    if has("c2patool"):
        for f in (master, vert):
            signed = f[:-4] + "_signed.mp4"
            try:
                sh("c2patool", f, "-m", mpath, "-o", signed, "-f")
                os.replace(signed, f)
            except Exception as e:
                print(f"[warn] c2patool sign failed for {os.path.basename(f)}: {e}")
    else:
        print(f"[note] {sid}: c2patool not installed -> embedded metadata + sidecar manifest only "
              "(install c2patool to cryptographically sign content credentials).")

    print(f"[ok] {sid} -> {outdir}  (master + vertical + thumb + manifest)")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--no-music", action="store_true", help="skip the music bed (voice/silent only)")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    if not has("ffmpeg"):
        print("[FAIL] ffmpeg not found (apt/brew install ffmpeg)"); sys.exit(1)
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    work = cfg["paths"]["work"]
    out_root = os.path.join(root, cfg["paths"]["out"])
    shots = json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]
    n = 0
    for s in shots:
        if a.only and s["id"] != a.only:
            continue
        if export_clip(cfg, root, s["id"], work, out_root, use_music=not a.no_music):
            n += 1
    print(f"[ok] exported {n} clip package(s) -> {os.path.join(out_root, 'clips')}")

if __name__ == "__main__":
    main()
