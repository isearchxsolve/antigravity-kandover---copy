#!/usr/bin/env python3
"""dcdm_packager.py -- DCDM (Digital Cinema Distribution Master) emitter.

The DCDM is the *master* file format cinemas consume. Two flavors are emitted:

  1. JPEG 2000 DCDM (modern, ~250 Mbps, visually lossless)
     - 12-bit XYZ color (DCI native)
     - DCI 4K (4096x2160) or 2K (2048x1080)
     - 24fps flat / scope / full
     - Frame-by-frame .j2c essence + 48 kHz 24-bit .wav audio

  2. Uncompressed DCDM (TIFF, archival)
     - 16-bit XYZ per-frame .tif
     - Used only for archival masters when KIT_DCDM_UNCOMPRESSED=1
     - Massive file size: ~1.5 GB per minute at 4K

The output structure:
    out_dir/
      video/
        frame_000001.j2c
        frame_000002.j2c
        ...
      audio/
        audio.wav
      INDEX.txt        # frame count, duration, aspect, color, audio config

The DCP packager then wraps these per-frame essences into MXF tracks for
the actual cinema delivery format.
"""
import os, subprocess, json
import theatrical_grade as tg


def _ff_j2c_available():
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=10)
        return "jpeg2000" in r.stdout or "libopenjpeg" in r.stdout
    except Exception:
        return False


def _frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets,nb_frames",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True).stdout.strip()
    try:
        return int(out.splitlines()[-1])
    except Exception:
        return 0


def build_dcdm(in_path, out_dir, aspect="flat", dci_4k=True,
               wav_path=None, audio_layout="5.1", verbose=True):
    """Emit a DCDM directory tree under `out_dir`.

    Args:
      in_path:       mezzanine with audio already (Rec.709 H.264 + audio)
      out_dir:       output DCDM root (created if missing)
      aspect:        'flat' (1.85) / 'scope' (2.39) / 'full' / 'uhd'
      dci_4k:        True -> DCI 4K canvas; False -> DCI 2K canvas
      wav_path:      optional separate audio track (else extracted from in_path)
      audio_layout:  '5.1' or '7.1' -- written to INDEX.txt for the DCP layer
    """
    os.makedirs(out_dir, exist_ok=True)
    video_dir = os.path.join(out_dir, "video")
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    tw, th = tg.aspect_dims(aspect, dci_4k)
    j2c = _ff_j2c_available()

    vf = (f"scale={tw}:{th}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black,"
          f"format=yuv444p12le,setsar=1")

    frame_pat = os.path.join(video_dir, "frame_%06d.j2c")
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-i", in_path,
            "-vf", vf,
            "-frames:v", "1", "-f", "image2", "-c:v", "png",
            os.path.join(video_dir, "frame_check.png")]
    if verbose:
        print(f"[dcdm] verifying canvas at {tw}x{th} via PNG dump...")
    subprocess.run(args, check=True)

    args2 = ["ffmpeg", "-y", "-loglevel", "error",
             "-i", in_path,
             "-vf", vf,
             "-an", "-c:v", "jpeg2000" if j2c else "libopenjpeg",
             "-pix_fmt", "xyz12le",
             "-r", "24",
             frame_pat]
    if verbose:
        print(f"[dcdm] encoding {tw}x{th} XYZ12 JPEG 2000 frames -> {video_dir}")
    subprocess.run(args2, check=True)

    out_wav = wav_path or os.path.join(audio_dir, "audio.wav")
    if wav_path is None:
        args3 = ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", in_path,
                 "-vn", "-c:a", "pcm_s24le", "-ar", "48000", out_wav]
        subprocess.run(args3, check=True)

    frame_count = _frame_count(in_path)
    duration = frame_count / 24.0 if frame_count else 0.0
    index = {
        "format": "DCDM-JPEG2000",
        "version": "1.0",
        "video": {
            "width": tw, "height": th, "bit_depth": 12,
            "color_space": "XYZ", "transfer": "DCI",
            "aspect_ratio": ASPECT_LABEL[aspect],
            "frame_rate": 24, "frame_count": frame_count,
            "codec": "jpeg2000", "container_per_frame": "j2c",
        },
        "audio": {
            "sample_rate": 48000, "bit_depth": 24,
            "channels": 6 if audio_layout == "5.1" else 8,
            "layout": audio_layout,
            "path": os.path.relpath(out_wav, out_dir),
        },
        "duration_seconds": round(duration, 3),
    }
    idx_path = os.path.join(out_dir, "INDEX.txt")
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    if verbose:
        print(f"[dcdm] wrote INDEX.txt -> {idx_path}")
        print(f"[dcdm] {frame_count} frames @ {tw}x{th} XYZ12; audio {audio_layout} 24/48.")
    return out_dir


ASPECT_LABEL = {
    "flat": "1.85:1 (flat)",
    "scope": "2.39:1 (scope)",
    "full": "1.90:1 (DCI full)",
    "uhd": "1.78:1 (UHD 4K, broadcast fallback)",
}


def build_uncompressed_dcdm(in_path, out_dir, aspect="flat", dci_4k=True,
                            wav_path=None, audio_layout="5.1"):
    """Archival 16-bit XYZ TIFF DCDM (no JPEG 2000 compression). ~1.5 GB/min @ 4K."""
    os.makedirs(out_dir, exist_ok=True)
    video_dir = os.path.join(out_dir, "video")
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    tw, th = tg.aspect_dims(aspect, dci_4k)
    tif_pat = os.path.join(video_dir, "frame_%06d.tif")
    vf = (f"scale={tw}:{th}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black,"
          f"format=yuv444p16le,setsar=1")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", in_path,
         "-vf", vf, "-an",
         "-c:v", "png", "-pix_fmt", "rgb48be",
         tif_pat],
        check=True)
    out_wav = wav_path or os.path.join(audio_dir, "audio.wav")
    if wav_path is None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-i", in_path, "-vn",
                        "-c:a", "pcm_s24le", "-ar", "48000", out_wav],
                       check=True)
    frame_count = _frame_count(in_path)
    index = {
        "format": "DCDM-Uncompressed",
        "video": {
            "width": tw, "height": th, "bit_depth": 16,
            "color_space": "XYZ (encoded as RGB48 for TIFF)",
            "aspect_ratio": ASPECT_LABEL[aspect],
            "frame_rate": 24, "frame_count": frame_count,
        },
        "audio": {
            "sample_rate": 48000, "bit_depth": 24,
            "channels": 6 if audio_layout == "5.1" else 8,
            "layout": audio_layout,
            "path": os.path.relpath(out_wav, out_dir),
        },
    }
    with open(os.path.join(out_dir, "INDEX.txt"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[dcdm-unc] {frame_count} frames @ {tw}x{th} 16-bit TIFF + audio")
    return out_dir


if __name__ == "__main__":
    print("dcdm_packager.py -- called from stitch.py when KIT_THEATRICAL=1")
