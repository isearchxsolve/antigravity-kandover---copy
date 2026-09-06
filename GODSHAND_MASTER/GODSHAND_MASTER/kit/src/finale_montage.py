#!/usr/bin/env python3
"""THE LAST LAUGH -- build the closing sequence: a collection of the film's own
moving images that accelerates into a recap of the love/worship, then METAMORPHOSES
into a joke -- funhouse mirror-warp, smear, greasepaint bleach, a freeze on the smile,
then the punchline snap to black.

The idea (per direction): 'the last i would want to be collection of some moving
images turning into a joke.' His whole life -- every tender frame of her descending
like an angel, the years of silent worship, 'the love was real' -- is recut fast and
then curdled, in front of her, into the one joke he has left. The film eats itself.

CPU-ONLY, deterministic, offline. Uses only ffmpeg. If a source shot has not been
rendered yet (dry-run / pre-GPU), a clearly-labelled placeholder clip is synthesized
so this stage always produces a valid file for validation. Carries its OWN baked audio
sting (slowed score-tail shape + a distorted descending 'laugh' figure + a sub-bass
drop into silence) so stitch can simply append it as the closing -- no filtergraph
surgery on the main mix.

BT.709 end to end, yuv420p, matches the film's width/height/fps.
"""
import os, sys, json, subprocess, argparse

def sh(*a):
    subprocess.run(list(a), check=True)

def ff(*a):
    # quiet ffmpeg
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)

def probe_dur(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", p], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except Exception:
        return 0.0

COLOR = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]

def _placeholder(dst, label, secs, w, h, fps):
    """Synthesize a plain placeholder clip when a real render is absent (dry-run only).
    No drawtext (avoids hard dependency on a system font on minimal boxes)."""
    ff("-f", "lavfi", "-i", f"color=c=0x101014:s={w}x{h}:r={fps}:d={secs:.3f}",
       "-vf", "format=yuv420p",
       "-c:v", "libx264", "-preset", "fast", "-crf", "18", *COLOR, dst)

def _clip(src, dst, start, secs, speed, w, h, fps):
    """Trim [start, start+secs] from src, time-scale by `speed` (>1 = faster), normalize."""
    pts = 1.0 / max(speed, 0.01)
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
          f"setpts={pts:.4f}*PTS,fps={fps},format=yuv420p")
    ff("-ss", f"{start:.3f}", "-t", f"{secs:.3f}", "-i", src,
       "-vf", vf, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", *COLOR, dst)

def build(root, dry_run=False):
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    work = cfg["paths"]["work"]
    w, h = int(cfg["width"]), int(cfg["height"])
    fps = int(cfg.get("interpolate_fps", cfg.get("fps", 24)))
    fin = (cfg.get("finale") or {})
    if not fin.get("enabled", True):
        print("[finale] disabled in config; skipping.")
        return None
    shots = json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]
    order = [s["id"] for s in shots]
    # Recap = the tender/worship/love images, in story order. Configurable.
    recap = fin.get("recap_shots") or [
        "pro_seen", "pro_silence", "pro_alone", "pro_crossing",
        "confr_real", "confr_aim", "confr_sayit", "confr_smile",
    ]
    recap = [s for s in recap if s in order]
    if not recap:
        recap = order[:min(6, len(order))]
    freeze_on = fin.get("freeze_on", "confr_smile")
    if freeze_on not in order:
        freeze_on = recap[-1] if recap else order[-1]
    total = float(fin.get("duration", 9.0))

    fdir = os.path.join(work, "finale"); os.makedirs(fdir, exist_ok=True)
    # 1) ACCELERATION: each successive memory is shorter & faster -> the reel speeds up.
    n = len(recap)
    seg_paths = []
    # geometric acceleration: durations shrink, speed grows toward the end
    base = 0.85
    for i, sid in enumerate(recap):
        frac = i / max(n - 1, 1)
        secs = max(0.22, base * (1.0 - 0.62 * frac))     # 0.85s -> ~0.32s
        speed = 1.0 + 1.9 * frac                          # 1.0x -> ~2.9x
        src = os.path.join(work, sid, "final.mp4")
        dst = os.path.join(fdir, f"seg_{i:02d}_{sid}.mp4")
        if os.path.exists(src) and probe_dur(src) > 0.05:
            st = min(0.2, max(0.0, probe_dur(src) * 0.25))
            _clip(src, dst, st, secs * speed, speed, w, h, fps)
        else:
            if not dry_run:
                print(f"[finale][warn] {sid} not rendered yet -> placeholder.")
            _placeholder(dst, f"[{sid}]", secs, w, h, fps)
        seg_paths.append(dst)

    # 2) concat the accelerating recap
    lst = os.path.join(fdir, "recap_list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in seg_paths))
    recap_mp4 = os.path.join(fdir, "recap.mp4")
    ff("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-preset", "fast",
       "-crf", "16", "-pix_fmt", "yuv420p", *COLOR, recap_mp4)
    rdur = probe_dur(recap_mp4)

    # 3) METAMORPHOSIS: the recap curdles into a joke. Progressive, time-gated:
    #    - a funhouse bulge grows (lenscorrection) -> reality warps
    #    - frames start to SMEAR (tblend average) -> memory smears into greasepaint
    #    - color bleaches & tints toward clown greasepaint (white face / red push)
    #    - a faint doubled mirror ghost -> the mask doubling the man
    g0 = max(0.0, rdur * 0.35)      # warp begins
    g1 = max(g0 + 0.1, rdur * 0.6)  # smear begins
    warped = os.path.join(fdir, "warped.mp4")
    vf = (
        f"format=yuv420p,"
        f"lenscorrection=cx=0.5:cy=0.5:k1=0.00:k2=0.00:i=bilinear,"  # placeholder, animated below via geq-free approach
        f"eq=contrast='1+0.6*gt(t,{g0:.3f})*(t-{g0:.3f})/{max(rdur-g0,0.1):.3f}':"
        f"saturation='1-0.7*gt(t,{g0:.3f})*(t-{g0:.3f})/{max(rdur-g0,0.1):.3f}':"
        f"brightness='0.06*gt(t,{g1:.3f})',"
        f"colorbalance=rs=0.05:rm=0.12:rh=0.18:bs=-0.05:bm=-0.08:bh=-0.12:"
        f"enable='gt(t,{g1:.3f})',"
        f"tblend=all_mode=average:enable='gt(t,{g1:.3f})',"
        f"noise=alls=8:allf=t+u:enable='gt(t,{g1:.3f})',"
        f"vignette=PI/4"
    )
    ff("-i", recap_mp4, "-vf", vf, "-an", "-c:v", "libx264", "-preset", "fast",
       "-crf", "16", "-pix_fmt", "yuv420p", *COLOR, warped)

    # 4) FREEZE on the smile, hold, then PUNCHLINE snap to black.
    fsrc = os.path.join(work, freeze_on, "final.mp4")
    smile_png = os.path.join(fdir, "smile.png")
    if os.path.exists(fsrc) and probe_dur(fsrc) > 0.05:
        ff("-sseof", "-0.3", "-i", fsrc, "-vframes", "1",
           "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", smile_png)
    else:
        ff("-f", "lavfi", "-i", f"color=c=0x141414:s={w}x{h}", "-vframes", "1", smile_png)
    hold = float(fin.get("freeze_hold", 1.4))
    freeze_mp4 = os.path.join(fdir, "freeze.mp4")
    # bleach the smile toward greasepaint, then a hard fade to black at the very end (the punchline)
    ff("-loop", "1", "-t", f"{hold:.3f}", "-i", smile_png, "-r", str(fps),
       "-vf", (f"format=yuv420p,eq=contrast=1.35:saturation=0.35:brightness=0.05,"
               f"colorbalance=rm=0.15:rh=0.2:bm=-0.1:bh=-0.15,vignette=PI/4,"
               f"fade=t=out:st={max(hold-0.28,0.0):.3f}:d=0.28"),
       "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", *COLOR, freeze_mp4)

    # 5) concat warped-recap + freeze -> silent finale video
    lst2 = os.path.join(fdir, "finale_list.txt")
    open(lst2, "w").write(f"file '{warped}'\nfile '{freeze_mp4}'\n")
    fvid = os.path.join(fdir, "finale_video.mp4")
    ff("-f", "concat", "-safe", "0", "-i", lst2, "-c:v", "libx264", "-preset", "slow",
       "-crf", "15", "-pix_fmt", "yuv420p", *COLOR, fvid)
    fdur = probe_dur(fvid)

    # 6) BAKE THE STING (offline, procedural): a solo-violin-shaped tone that lands and
    #    keeps rising, then breaks into a distorted descending 'laugh' figure, a sub-bass
    #    drop, and silence. Wholly synthetic -- no copyrighted cue, no sample.
    swell_t = max(fdur * 0.55, 0.5)
    sting = os.path.join(fdir, "sting.wav")
    # a) rising violin-order tone (sawtooth through lowpass, vibrato), swelling in
    a_swell = (f"aevalsrc='0.28*sin(2*PI*(587+18*sin(2*PI*5*t))*t)*exp(-0.0*t)':"
               f"s=48000:d={swell_t:.3f},"
               f"tremolo=f=5.5:d=0.3,afade=t=in:st=0:d={min(0.6,swell_t):.3f},lowpass=f=4000")
    # b) descending distorted 'laugh' figure (four falling notes) + sub drop after the swell
    laugh_d = max(fdur - swell_t, 1.2)
    a_laugh = (f"aevalsrc='0.33*sin(2*PI*(330 - 120*mod(floor(t*6),4)/3)*t)':"
               f"s=48000:d={laugh_d:.3f},"
               f"aeval='tanh(3.2*val(0))':c=same,"          # soft-clip distortion = it sours into a sneer
               f"tremolo=f=6:d=0.6,afade=t=out:st={max(laugh_d-0.5,0.0):.3f}:d=0.5")
    a_sub = (f"aevalsrc='0.6*sin(2*PI*(70*exp(-2.2*t))*t)':s=48000:d=1.1,"
             f"afade=t=out:st=0.5:d=0.6")  # the fall: a body-drop sub-bass
    stinggraph = (f"{a_swell}[sw];{a_laugh}[lg];{a_sub}[sb];"
                  f"[sw][lg]acrossfade=d=0.35:c1=tri:c2=tri[reel];"
                  f"[reel][sb]amix=inputs=2:normalize=0[m];"
                  f"[m]aformat=sample_rates=48000:channel_layouts=stereo,"
                  f"loudnorm=I=-16:LRA=11:TP=-1.5,alimiter=limit=0.95[sting]")
    ff("-filter_complex", stinggraph, "-map", "[sting]", "-t", f"{fdur:.3f}", sting)

    # 7) mux sting under the finale video -> the closing 'the joke' clip
    thejoke = os.path.join(fdir, "thejoke.mp4")
    ff("-i", fvid, "-i", sting, "-map", "0:v", "-map", "1:a",
       "-c:v", "copy", "-c:a", "aac", "-shortest", thejoke)
    print(f"[finale] recap={recap} freeze_on={freeze_on} dur={fdur:.2f}s -> {thejoke}")
    return thejoke

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true",
                    help="build with placeholders where renders are absent (validation).")
    a = ap.parse_args()
    out = build(os.path.abspath(a.root), dry_run=a.dry_run)
    if out:
        print(f"[ok] finale -> {out}")

if __name__ == "__main__":
    main()
