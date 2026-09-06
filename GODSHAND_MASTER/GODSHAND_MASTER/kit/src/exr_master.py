#!/usr/bin/env python3
"""EXPERIMENTAL: color-managed 4:4:4 10-bit ProRes master from scene-linear EXR.

This is the ARCHITECTURE change for true >8-bit output. It only helps if the
frames are captured as 16-bit-float EXR straight from the Wan VAE decode (before
any 8-bit PNG / H.264 step). See EXPERIMENTAL_EXR_MASTER.md for the ComfyUI
Save-EXR node change that produces those frames.

VALIDATED here: the color-managed encode itself (scene-linear EXR -> Rec.709
OETF via zscale -> ProRes 4444 10-bit 4:4:4, NCLC 1-1-1 tagged).
NOT tested: the upstream EXR capture (needs the GPU/ComfyUI session).

Key facts baked in:
- EXR is scene-LINEAR float; it MUST get a linear->Rec.709 transfer or the master
  comes out dark/flat. The zscale stage does exactly that.
- prores_ks accepts only 10-bit input (yuv444p10le); the .mov probes as
  yuv444p12le (12-bit container) but real precision is 10-bit. True 12-bit needs
  a proprietary encoder (Resolve Studio / Compressor).

Usage: exr_master.py EXR_DIR OUT.mov [FPS=24] [AUDIO.wav]
Non-fatal: returns non-zero on failure but never raises.
"""
import sys, os, re, glob, subprocess


def _has_filter(name):
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                           capture_output=True, text=True)
        return any(len(p) > 1 and p[1] == name
                   for p in (l.split() for l in r.stdout.splitlines()))
    except Exception:
        return False


def main():
    if len(sys.argv) < 3:
        print("usage: exr_master.py EXR_DIR OUT.mov [FPS] [AUDIO.wav]")
        return 1
    exr_dir, out = sys.argv[1], sys.argv[2]
    fps = sys.argv[3] if len(sys.argv) > 3 else "24"
    audio = sys.argv[4] if len(sys.argv) > 4 else None

    frames = sorted(glob.glob(os.path.join(exr_dir, "*.exr")))
    if not frames:
        print(f"[exr] no EXR frames in {exr_dir}"); return 1
    base = os.path.basename(frames[0])
    m = re.match(r"(.*?)(\d+)(\.exr)$", base)
    if not m:
        print(f"[exr] unexpected EXR naming: {base}"); return 1
    pattern = os.path.join(exr_dir, f"{m.group(1)}%0{len(m.group(2))}d{m.group(3)}")

    if _has_filter("zscale"):
        # EXR carries no color tags, so declare the INPUT as scene-linear
        # (tin=linear) with bt709 primaries, then convert to bt709 transfer.
        vf = ("zscale=tin=linear:pin=bt709:min=bt709:rin=full:"
              "t=bt709:p=bt709:m=bt709:r=tv,format=yuv444p10le")
    else:
        print("[exr] WARNING: zscale/zimg unavailable; no linear->709 transfer "
              "applied. Master will look dark/flat until zimg is installed.")
        vf = "format=yuv444p10le"

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern]
    if audio:
        cmd += ["-i", audio]
    cmd += ["-vf", vf,
            "-c:v", "prores_ks", "-profile:v", "4", "-vendor", "apl0",
            "-pix_fmt", "yuv444p10le",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-r", str(fps)]
    if audio:
        cmd += ["-c:a", "pcm_s24le", "-ar", "48000", "-shortest"]
    cmd += [out]

    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(out):
        print("[exr] master encode failed"); return 1
    print(f"[exr] color-managed 4:4:4 10-bit master -> {out} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
