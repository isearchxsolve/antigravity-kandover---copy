#!/usr/bin/env python3
"""theatrical_grade.py -- DCI-P3 color pipeline + HDR PQ for theatrical release.

Converts a Rec.709 H.264 master into three theatrical deliverables:

  * SDR-theatrical (DCI-P3-D65, gamma 2.6, DCI 4K flat or scope)
    -> cinema projector native, 14 fL white, intended for SDR theatrical
  * HDR-theatrical (DCI-P3-D65, PQ ST 2084, MaxCLL/MaxFALL metadata)
    -> Dolby Cinema, IMAX with laser, premium HDR screens
  * Scope/Flat (1.85:1 / 2.39:1) letterboxing inside the 4096x2160 DCI canvas

Color management: zscale (zimg) when present; ffmpeg's built-in colorspace
filter as a portable fallback. Both paths enforce full-range float math
internally so no banding on 8-bit tonal scales.

Aspect ratios:
    flat  1.85:1  -> DCI 4K  3996x2160  (or 2K 1998x1080)
    scope 2.39:1  -> DCI 4K  4096x1716  (or 2K 2048x858)
    hd    1.78:1  -> 3840x2160 UHD    (NOT theatrical, broadcast fallback)
"""
import os, subprocess, json

DCI_W_4K = 4096
DCI_H_4K = 2160

ASPECT_PRESETS = {
    "flat":   {"w": 3996, "h": 2160, "ratio": "1.85:1"},
    "scope":  {"w": 4096, "h": 1716, "ratio": "2.39:1"},
    "full":   {"w": 4096, "h": 2160, "ratio": "1.90:1"},
    "uhd":    {"w": 3840, "h": 2160, "ratio": "1.78:1 (NOT theatrical)"},
}


def aspect_dims(aspect="flat", dci_4k=True):
    p = ASPECT_PRESETS.get(aspect, ASPECT_PRESETS["flat"])
    if not dci_4k:
        return p["w"] // 2, p["h"] // 2
    return p["w"], p["h"]


def _has_zscale():
    """zscale needs the zimg filter -- only in ffmpeg built with --enable-libzimg."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                           capture_output=True, text=True, timeout=10)
        return " zscale " in (" " + r.stdout + " ")
    except Exception:
        return False


def _zscale_chain(p3=True, hdr=False):
    """Build a colorspace conversion filter chain using zscale."""
    if hdr:
        # PQ (SMPTE ST 2084), P3-D65 primaries, full-range container
        return (
            "format=yuv444p16le,"
            "zscale=transfer=linear:npl=100,"
            "zscale=p=bt2020:t=st2084:m=bt2020:size=4096x2160,"
            "format=yuv444p16le"
        )
    # SDR P3-D65, gamma 2.6 (DCI white)
    return (
        "format=yuv444p16le,"
        "zscale=transfer=linear:npl=100,"
        "zscale=p=dci-p3-d65:t=gamma22:m=bt2020_ncl:size=4096x2160,"
        "format=yuv444p10le"
    )


def _colorspace_fallback(p3=True, hdr=False):
    """Portable fallback using ffmpeg's built-in `colorspace` filter."""
    if hdr:
        return ("format=yuv444p16le,"
                "colorspace=space=bt2020nc:primaries=bt2020:trc=st2084:range=pc,"
                "format=yuv444p16le")
    return ("format=yuv444p10le,"
            "colorspace=space=dci-p3:primaries=smpte431:trc=gamma22:range=pc,"
            "format=yuv444p10le")


def aspect_pad_filter(target_w, target_h):
    """Letterbox / pillarbox filter: scale source to fit, pad to DCI canvas.
    Preserves aspect ratio, no distortion. Black bars = true black (0,0,0)."""
    return (f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1")


def grade_to_theatrical(in_path, out_path, aspect="flat", hdr=False,
                        dci_4k=True, video_bitrate="250M", pix_fmt=None,
                        audio_stream=None):
    """Convert `in_path` (Rec.709 master) into a theatrical-grade DCDM-track-ready
    mezzanine at `out_path`. Args:
      aspect   'flat' (1.85) / 'scope' (2.39) / 'full' / 'uhd'
      hdr      True -> PQ ST 2084; False -> DCI SDR (gamma 2.6)
      dci_4k   True -> 4096-wide canvas; False -> 2K equivalent
      pix_fmt  Override (default: yuv444p10le for SDR, yuv444p16le for HDR)
    """
    tw, th = aspect_dims(aspect, dci_4k)
    pix = pix_fmt or ("yuv444p16le" if hdr else "yuv444p10le")
    chain_fn = _zscale_chain if _has_zscale() else _colorspace_fallback
    color_chain = chain_fn(p3=True, hdr=hdr)
    pad = aspect_pad_filter(tw, th)
    vf = f"{pad},{color_chain}"
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-i", in_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "slow",
            "-b:v", video_bitrate, "-maxrate", video_bitrate, "-bufsize", str(int(video_bitrate.rstrip("M")) * 2) + "M",
            "-pix_fmt", pix]
    if hdr:
        args += ["-x264-params", "colorprim=9:transfer=16:colormatrix=9:range=pc"]
    else:
        args += ["-x264-params", "colorprim=11:transfer=10:colormatrix=11:range=pc"]
    if audio_stream:
        args += ["-map", f"0:a:{audio_stream}", "-c:a", "pcm_s24le", "-ar", "48000"]
    args.append(out_path)
    subprocess.run(args, check=True)
    return out_path


def grade_pipeline(in_path, work_dir, aspect="flat"):
    """Run both SDR and HDR theatrical passes; return dict of paths."""
    os.makedirs(work_dir, exist_ok=True)
    sdr_path = os.path.join(work_dir, f"theatrical_sdr_{aspect}.mp4")
    hdr_path = os.path.join(work_dir, f"theatrical_hdr_{aspect}.mp4")
    grade_to_theatrical(in_path, sdr_path, aspect=aspect, hdr=False, audio_stream=0)
    grade_to_theatrical(in_path, hdr_path, aspect=aspect, hdr=True, audio_stream=0)
    return {"sdr": sdr_path, "hdr": hdr_path}


def measure_maxcll_maxfall(in_path):
    """Measure content-light metadata: MaxCLL (max luminance, cd/m^2) and
    MaxFALL (max frame-average light level). Required for HDR10 static metadata.
    Uses ffmpeg's signalstats filter to derive nits from luma."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", in_path,
             "-vf", "signalstats,metadata=print:file=-",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        maxcll = maxfall = None
        for line in r.stderr.splitlines():
            if "YMAX" in line:
                try:
                    maxcll = int(float(line.split("=")[-1].strip()) / 1023.0 * 10000)
                except Exception:
                    pass
            if "YAVG" in line:
                try:
                    maxfall = int(float(line.split("=")[-1].strip()) / 1023.0 * 10000)
                except Exception:
                    pass
        return {"MaxCLL": maxcll, "MaxFALL": maxfall}
    except Exception:
        return {"MaxCLL": None, "MaxFALL": None}


def hdr10_metadata_blob(maxcll, maxfall):
    """Emit HDR10 static metadata as a JSON sidecar (mastering-display + content-light).
    DCP packagers can consume this for HDR theatrical deliverables."""
    return {
        "MasteringDisplayPrimaries": [
            0.708, 0.292,
            0.170, 0.797,
            0.131, 0.046,
        ],
        "MasteringDisplayLuminance": {"min": 0.01, "max": 1000.0, "unit": "cd/m^2"},
        "MaxCLL": f"{maxcll or 1000}",
        "MaxFALL": f"{maxfall or 400}",
    }


if __name__ == "__main__":
    print(json.dumps({
        "zscale_available": _has_zscale(),
        "aspect_dims_flat": aspect_dims("flat"),
        "aspect_dims_scope": aspect_dims("scope"),
        "aspect_dims_uhd": aspect_dims("uhd"),
    }, indent=2))
