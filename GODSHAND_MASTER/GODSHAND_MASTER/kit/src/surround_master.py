#!/usr/bin/env python3
"""surround_master.py -- Stereo → 5.1 / 7.1 theatrical surround upmix.

Naive stereo→5.1 (just duplicating channels) sounds flat and amateur.
This module does a *proper* upmix:

  Channel | Source                | Why
  --------|-----------------------|-------------------------------------
  L       | L                     | original
  R       | R                     | original
  C       | 0.7 * (L+R)           | dialogue reinforcement (phantom-center)
  LFE     | lowpass(L+R, 120 Hz)   | bass extraction, +10 dB shelf
  Ls      | 0.5 * (L-R), lowpass  | ambient surround, room tone
  Rs      | 0.5 * (R-L), lowpass  | ambient surround, room tone

For 7.1 we split Ls/Rs into two pairs at different delays so the rear wall
has dimensional depth instead of a single plane.

Output: 48 kHz / 24-bit / 6 or 8 channels, channel layout tag written so
the DCP packager maps the right essence to the right DCP channel.
"""
import os, subprocess

CHANNEL_LAYOUT_51 = "5.1"
CHANNEL_LAYOUT_71 = "7.1"


def _stereo_to_51_filter():
    """Filter chain for stereo → 5.1 upmix using ffmpeg's filter graph."""
    return (
        "[0:a]asplit=2[orig][diff];"
        "[orig]pan=5.1|"
        "c0=c0|c1=c1|"                          # L, R
        "c2=0.7*0.5*(c0+c1)|"                  # C (phantom-center)
        "c3=0.7*(c0+c1)|"                      # LFE input (will lowpass below)
        "c4=0.5*(c0-c1)|"                      # Ls
        "c5=0.5*(c1-c0)"                       # Rs
        "[pre];"
        "[pre]highpass=f=28,lowpass=f=18000[pre2];"
        "[pre2]asplit=6[L][R][C][LFEraw][Ls][Rs];"
        "[LFEraw]lowpass=f=120,volume=+10dB[LFE];"
        "[Ls]lowpass=f=8000,highpass=f=200[Ls2];"
        "[Rs]lowpass=f=8000,highpass=f=200[Rs2];"
        "[L][R][C][LFE][Ls2][Rs2]amerge=inputs=6[out]"
    )


def _stereo_to_71_filter():
    """Stereo → 7.1 with split surrounds for dimensional rear."""
    return (
        "[0:a]asplit=2[orig][diff];"
        "[orig]pan=7.1|"
        "c0=c0|c1=c1|"
        "c2=0.7*0.5*(c0+c1)|"
        "c3=0.7*(c0+c1)|"
        "c4=0.5*(c0-c1)|"
        "c5=0.5*(c1-c0)|"
        "c6=0.4*(c0-c1)|"
        "c7=0.4*(c1-c0)"
        "[pre];"
        "[pre]highpass=f=28,lowpass=f=18000[pre2];"
        "[pre2]asplit=8[L][R][C][LFEraw][Ls][Rs][Lsr][Rsr];"
        "[LFEraw]lowpass=f=120,volume=+10dB[LFE];"
        "[Ls]lowpass=f=8000,highpass=f=200,adelay=0|0[Ls2];"
        "[Rs]lowpass=f=8000,highpass=f=200,adelay=0|0[Rs2];"
        "[Lsr]lowpass=f=8000,highpass=f=200,adelay=15|15[Lsr2];"
        "[Rsr]lowpass=f=8000,highpass=f=200,adelay=15|15[Rsr2];"
        "[L][R][C][LFE][Ls2][Rs2][Lsr2][Rsr2]amerge=inputs=8[out]"
    )


def _safe_filter_chain(layout):
    """Use ffmpeg's native `surround` filter when available, else our manual chain."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                           capture_output=True, text=True, timeout=10)
        if " surround " in (" " + r.stdout + " "):
            return f"surround=layout={layout}:lfe=1[out]"
    except Exception:
        pass
    if layout == "5.1":
        return _stereo_to_51_filter()
    return _stereo_to_71_filter()


def upmix(in_path, out_path, layout="5.1"):
    """Upmix a stereo audio file to 5.1 or 7.1, write 24-bit WAV."""
    if layout not in (CHANNEL_LAYOUT_51, CHANNEL_LAYOUT_71):
        raise ValueError(f"layout must be 5.1 or 7.1, got {layout!r}")
    fc = _safe_filter_chain(layout)
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-i", in_path,
            "-filter_complex", fc,
            "-map", "[out]",
            "-ar", "48000", "-sample_fmt", "s32",
            "-ac", "6" if layout == "5.1" else "8",
            "-c:a", "pcm_s24le",
            out_path]
    subprocess.run(args, check=True)
    return out_path


def channel_layout_tag(layout="5.1"):
    """Return the ffmpeg channel-layout name to embed in the WAV header."""
    return "5.1" if layout == "5.1" else "7.1"


def theatrical_bed_mix(stem_dir, out_path, layout="5.1"):
    """Build a theatrical 5.1/7.1 bed from the existing stems (music, ambience,
    voice, chorus, reverb_return) using broadcast-standard routing:

        L/R  <- music + ambience (stereo)
        C    <- voice (mono phantom-center, dialogue anchor)
        LFE  <- music bass-redirect (lowpass 120 Hz, +10 dB)
        Ls/Rs<- ambience + reverb_return (room tone)
    """
    music = os.path.join(stem_dir, "music.wav")
    amb = os.path.join(stem_dir, "ambience.wav")
    voice = os.path.join(stem_dir, "voice.wav")
    rev = os.path.join(stem_dir, "reverb_return.wav")
    if not all(os.path.exists(p) for p in (music, amb, voice, rev)):
        missing = [p for p in (music, amb, voice, rev) if not os.path.exists(p)]
        raise FileNotFoundError(f"missing stem(s) for theatrical mix: {missing}")
    n = 6 if layout == "5.1" else 8
    fc = (
        f"[0:a]volume=-4dB,aresample=48000[m];"
        f"[1:a]volume=-10dB,aresample=48000[a];"
        f"[2:a]volume=+2dB,aresample=48000[v];"
        f"[3:a]volume=-12dB,aresample=48000[r];"
        f"[m]pan={layout}|c0=c0|c1=c1[m51];"
        f"[m]lowpass=f=120,volume=+10dB[m_lfe];"
        f"[v]pan={layout}|c2=c0[v51];"
        f"[a][r]amix=inputs=2:normalize=0,pan={layout}|c4=c0|c5=c1[amb51];"
        f"[m51][v51][m_lfe][amb51]amerge=inputs={n}[merged]"
    )
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-i", music, "-i", amb, "-i", voice, "-i", rev,
            "-filter_complex", fc,
            "-map", "[merged]",
            "-ar", "48000", "-ac", str(n),
            "-c:a", "pcm_s24le",
            out_path]
    subprocess.run(args, check=True)
    return out_path


if __name__ == "__main__":
    print("surround_master.py — filter graph only; run stitch.py to exercise.")
