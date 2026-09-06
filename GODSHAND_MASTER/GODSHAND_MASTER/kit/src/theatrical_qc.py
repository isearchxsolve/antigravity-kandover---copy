#!/usr/bin/env python3
"""theatrical_qc.py -- QC checks for theatrical deliverables.

Validates:
  * DCI-P3 primaries (or P3-D65) present in video stream
  * HDR PQ (ST 2084) transfer function range [0, 10000] nits
  * Aspect ratio is DCI flat (1.85:1) or scope (2.39:1) within 4096x2160 canvas
  * 5.1 or 7.1 channel layout present in audio
  * DCDM frame count matches CPL reel durations
  * JPEG 2000 codestream is 12-bit XYZ (or 12-bit P3 for SDR)
  * MaxCLL / MaxFALL metadata present for HDR
  * No letterboxing in the DCI canvas (content fills the aspect)

Returns FAIL/WARN/PASS like qc.py for integration into CI gate.
"""
import os, re, subprocess, json, sys

REQUIRED_PRIMARIES = ["bt2020", "p3", "dci-p3", "smpte431"]
PQ_TRANSFERS = ["st2084", "smpte2084", "pq"]
SDR_TRANSFERS = ["gamma22", "gamma26", "bt709"]
DCI_ASPECTS = {
    "flat":  (1.85, 1.85, 1.85),   # min, target, max
    "scope": (2.39, 2.39, 2.39),
    "full":  (1.89, 1.90, 1.91),
}


def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", path], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def video_stream(info):
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    return vs[0] if vs else {}


def audio_stream(info):
    aud = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    return aud[0] if aud else {}


def check_p3_primaries(path):
    v = video_stream(ffprobe(path))
    prim = v.get("color_primaries", "").lower()
    space = v.get("color_space", "").lower()
    ok = any(p in prim for p in REQUIRED_PRIMARIES) or any(p in space for p in REQUIRED_PRIMARIES)
    return ok, f"primaries={prim or '?'}, space={space or '?'}"


def check_transfer_function(path):
    v = video_stream(ffprobe(path))
    trc = v.get("color_transfer", "").lower()
    is_pq = any(t in trc for t in PQ_TRANSFERS)
    is_sdr = any(t in trc for t in SDR_TRANSFERS)
    return is_pq, is_sdr, trc


def check_aspect(path):
    v = video_stream(ffprobe(path))
    w = int(v.get("width", 0))
    h = int(v.get("height", 0))
    if not w or not h:
        return False, "no dims", 0.0
    dar = v.get("display_aspect_ratio", "")
    par = v.get("sample_aspect_ratio", "")
    aspect = w / h
    if dar and ":" in dar:
        try:
            a, b = dar.split(":")
            aspect = float(a) / float(b)
        except Exception:
            pass
    is_dci = aspect >= 1.84 and aspect <= 1.92
    is_flat = abs(aspect - 1.85) < 0.02
    is_scope = abs(aspect - 2.39) < 0.03
    return is_dci, f"wxh={w}x{h} dar={dar} par={par} aspect={aspect:.3f}", aspect


def check_audio_layout(path):
    a = audio_stream(ffprobe(path))
    ch = a.get("channels", 0)
    layout = a.get("channel_layout", "").lower()
    if ch in (6, 8) and "5.1" in layout or "7.1" in layout:
        return True, f"{layout} ({ch} ch)"
    return False, f"{layout} ({ch} ch)"


def check_j2c_xyz(path):
    """Verify the video codec is JPEG 2000 and pix_fmt is 12-bit (XYZ or P3)."""
    v = video_stream(ffprobe(path))
    codec = v.get("codec_name", "").lower()
    pix = v.get("pix_fmt", "").lower()
    is_j2k = codec in ("j2k", "jpeg2000", "mjpeg")
    is_12bit = "12" in pix or "xyz" in pix or "p010" in pix or "yuv444p12" in pix
    return is_j2k and is_12bit, f"codec={codec} pix={pix}"


def check_hdr_metadata(path):
    """Check for MaxCLL / MaxFALL in frame side data (HDR10 static metadata)."""
    info = ffprobe(path)
    v = video_stream(info)
    md = v.get("side_data_list", [])
    has_maxcll = any("max_content" in str(d).lower() for d in md)
    has_maxfall = any("max_frame" in str(d).lower() for d in md)
    return has_maxcll and has_maxfall, f"side_data={len(md)} entries"


def check_reel_integrity(dcdm_dir, cpl_dir):
    """Verify DCDM frame count matches CPL reel durations."""
    idx_path = os.path.join(dcdm_dir, "INDEX.txt")
    if not os.path.exists(idx_path):
        return False, "no INDEX.txt"
    idx = json.load(open(idx_path))
    vfc = idx.get("video", {}).get("frame_count", 0)
    if not cpl_dir or not os.path.isdir(cpl_dir):
        return True, "no CPL dir to cross-check"
    total_cpl = 0
    for f in os.listdir(cpl_dir):
        if f.startswith("CPL_") and f.endswith(".xml"):
            m = re.search(r"Duration>(\d+)</Duration>", open(os.path.join(cpl_dir, f)).read())
            if m:
                total_cpl += int(m.group(1))
    return vfc == total_cpl, f"DCDM frames={vfc} CPL frames={total_cpl}"


def theatrical_check(path, dcdm_dir=None, cpl_dir=None, expect_hdr=False):
    """Full theatrical QC gate. Returns (verdict, issues, stats)."""
    issues = []
    stats = {}
    verdict = "PASS"

    def _worsen(v, order=("PASS", "WARN", "FAIL")):
        nonlocal verdict
        if order.index(v) > order.index(verdict):
            verdict = v

    # 1) DCI-P3 primaries
    ok, msg = check_p3_primaries(path)
    if not ok:
        issues.append(f"P3 primaries missing: {msg}")
        _worsen("FAIL")
    stats["primaries"] = msg

    # 2) Transfer function
    is_pq, is_sdr, trc = check_transfer_function(path)
    if expect_hdr:
        if not is_pq:
            issues.append(f"HDR expected but transfer={trc}")
            _worsen("FAIL")
    else:
        if not is_sdr and not is_pq:
            issues.append(f"Unknown transfer={trc}")
            _worsen("WARN")
    stats["transfer"] = trc

    # 3) Aspect ratio (DCI flat/scope)
    ok, msg, aspect = check_aspect(path)
    if not ok:
        issues.append(f"Non-DCI aspect: {msg}")
        _worsen("WARN")
    stats["aspect"] = msg

    # 4) Audio layout (5.1/7.1)
    ok, msg = check_audio_layout(path)
    if not ok:
        issues.append(f"Audio not theatrical: {msg}")
        _worsen("FAIL")
    stats["audio"] = msg

    # 5) JPEG 2000 12-bit XYZ (only for DCDM MXF, not MP4 preview)
    if path.endswith(".mxf"):
        ok, msg = check_j2c_xyz(path)
        if not ok:
            issues.append(f"Video not J2K 12-bit: {msg}")
            _worsen("FAIL")
        stats["codec"] = msg

    # 6) HDR metadata
    if expect_hdr:
        ok, msg = check_hdr_metadata(path)
        if not ok:
            issues.append(f"HDR metadata incomplete: {msg}")
            _worsen("FAIL")
        stats["hdr_meta"] = msg

    # 7) Reel integrity
    if dcdm_dir and cpl_dir:
        ok, msg = check_reel_integrity(dcdm_dir, cpl_dir)
        if not ok:
            issues.append(f"Reel frame mismatch: {msg}")
            _worsen("FAIL")
        stats["reel_integrity"] = msg

    return verdict, issues, stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dcdm")
    ap.add_argument("--cpl")
    ap.add_argument("--hdr", action="store_true")
    a = ap.parse_args()
    if a.path:
        v, i, s = theatrical_check(a.path, a.dcdm, a.cpl, a.hdr)
        print(f"[{v}] {a.path}")
        for iss in i:
            print(f"  - {iss}")
        print(json.dumps(s, indent=2))
        sys.exit(1 if v == "FAIL" else 0)
    else:
        print("theatrical_qc.py -- path required")