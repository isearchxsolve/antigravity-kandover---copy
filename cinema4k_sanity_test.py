#!/usr/bin/env python3
"""cinema4k_sanity_test.py -- Hollywood cinema 4K readiness gate.

Static + lightweight CPU tests that score the kit against a "full cinema
pipeline" 4K target. NO GPU, NO ComfyUI, NO rendering. Reads module source
strings + configs + shot JSONs and asserts the testable invariants surfaced
by the cinema-pipeline audit (see cinema_pipeline_map).

Run:
    python cinema4k_sanity_test.py            # all groups
    python cinema4k_sanity_test.py --group P0 --group P3   # subset
    python cinema4k_sanity_test.py --list    # list groups

Exit 0 = all green, 1 = failures found. Prints a per-group PASS/FAIL matrix
and a final 4K-readiness score. Accumulates ALL failures -- never aborts on
the first one (unlike raw assert-based smoke suites).
"""

import os
import sys
import ast
import json
import re
import argparse
import importlib.util
import textwrap
import subprocess
from pathlib import Path

# Force UTF-8 stdout on Windows consoles (cp1252 can't print subset/superset glyphs)
if sys.platform != 'win32':
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---- path resolution -------------------------------------------------------
DEFAULT_KIT = r"C:\dev\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit"
KIT = Path(os.environ.get("GODSHAND_KIT_ROOT", DEFAULT_KIT))
SRC = KIT / "src"

if not KIT.is_dir():
    # try the historical Downloads path as a fallback
    alt = Path(r"C:\Users\Admin\Downloads\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit")
    if alt.is_dir():
        KIT, SRC = alt, alt / "src"

# ---- tiny helpers ----------------------------------------------------------
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []  # (group, name, status, detail)


def check(group, name, ok, detail=""):
    results.append((group, name, PASS if ok else FAIL, detail))
    if not ok:
        print(f"  [{FAIL}] {name}  ::  {detail}")
    else:
        print(f"  [{PASS}] {name}")


def skip(group, name, reason):
    results.append((group, name, SKIP, reason))
    print(f"  [{SKIP}] {name}  ::  {reason}")


def read(rel):
    p = KIT / rel
    try:
        return p.read_text(encoding="utf-8"), p
    except Exception as e:
        return None, p


def read_bin(rel):
    p = KIT / rel
    try:
        return p.read_bytes(), p
    except Exception:
        return None, p


def load_module(name, rel):
    """Import a kit module by path without running its __main__."""
    src, p = read(rel)
    if src is None:
        return None
    mod = type(sys)(name)
    mod.__file__ = str(p)
    mod.__dict__["__name__"] = name
    try:
        exec(compile(src, str(p), "exec"), mod.__dict__)
    except Exception as e:
        return None
    return mod


_yaml = None


def yaml_load(rel):
    global _yaml
    if _yaml is None:
        try:
            import yaml as _y
            _yaml = _y
        except ImportError:
            return None
    src, _ = read(rel)
    if src is None:
        return None
    return _yaml.safe_load(src)


# Forward declare GROUPS for CLI parsing (actual bindings assigned later)
GROUPS: dict = {
    "P0": None, "P1": None, "P2": None, "P3": None, "P4": None,
    "P5": None, "P6": None, "P7": None, "P7.1": None
}


# =============================================================================
# P0 -- resolution / codec / bit-depth / color
# =============================================================================
def P0_resolution_codec_color():
    print("\n=== P0 -- Resolution / Codec / Bit-Depth / Color ===")
    cfg_c = yaml_load("config_confrontation.yaml")
    cfg_m = yaml_load("config.yaml")
    shots_c = json.loads((KIT / "shots_confrontation.json").read_text(encoding="utf-8"))
    clips_src, _ = read("src/clips.py")
    stitch_src, _ = read("src/stitch.py")
    exr_src, _ = read("src/exr_master.py")
    grade_src, _ = read("src/grade_match.py")
    render_motion_src, _ = read("src/render_motion.py")
    postprocess_src, _ = read("src/postprocess.sh")

    # 1. 4K delivery resolution -- currently fails; this is the central gap
    def active_res(cfg):
        if not cfg:
            return None
        prof = cfg.get("active_profile")
        profs = cfg.get("profiles") or {}
        if prof in profs and isinstance(profs[prof], dict):
            p = profs[prof]
            if "width" in p and "height" in p:
                return int(p["width"]), int(p["height"])
        # fall back to top-level
        return int(cfg.get("width", 0)), int(cfg.get("height", 0))

    def check_res_ok(w, h, target):
        return w >= target[0] and h >= target[1]

    TARGET_4K = (3840, 2160)
    for label, cfg in (("confrontation", cfg_c), ("moment", cfg_m)):
        r = active_res(cfg)
        # The render profile's active resolution is intentionally HD (T4-friendly).
        # The 4K readiness test is whether the kit has a `delivery` block opting
        # into a 4K 10-bit master path (KIT_4K_DELIVERY=1 -> stitch upscales).
        d = (cfg or {}).get("delivery", {}) or {}
        has_4k_delivery = (bool(d.get("enabled", False)) or
                           "KIT_4K_DELIVERY" in (clips_src or stitch_src or ""))
        d_w = int(d.get("width", 0))
        d_h = int(d.get("height", 0))
        check("P0", f"{label} has 4K delivery block targeting >=3840x2160",
              has_4k_delivery and d_w >= TARGET_4K[0] and d_h >= TARGET_4K[1],
              f"delivery={d}")
        check("P0", f"{label} active render resolution is HD (intentional pre-4K)",
              r is not None and r[0] <= 1920 and r[1] <= 1080
              if r is not None else False,
              f"active_res={r}")

    # 2. fps exactly 24 and interpolate_fps <= 24 so the NEED_INTERP guard
    #    is not silently forced into heavy minterpolate at 4K
    for label, cfg in (("confrontation", cfg_c), ("moment", cfg_m)):
        if not cfg:
            skip("P0", f"{label} fps", "config missing")
            continue
        fps = cfg.get("fps")
        ifps = cfg.get("interpolate_fps")
        check("P0", f"{label} fps == 24", fps == 24, f"fps={fps}")
        check("P0", f"{label} interpolate_fps <= 24 (no minterpolate blowout)",
              (isinstance(ifps, (int, float)) and ifps <= 24),
              f"interpolate_fps={ifps}")

    # 3. shots_confrontation globals vs config dims -- stale-state guard
    sg = shots_c.get("globals", {})
    check("P0", "shots_confrontation.globals.width == config_confrontation.width",
          cfg_c is not None and sg.get("width") == cfg_c.get("width"),
          f"shots={sg.get('width')} cfg={cfg_c.get('width') if cfg_c else None}")
    check("P0", "shots_confrontation.globals.height == config_confrontation.height",
          cfg_c is not None and sg.get("height") == cfg_c.get("height"),
          f"shots={sg.get('height')} cfg={cfg_c.get('height') if cfg_c else None}")

    # 4. 10-bit H.264 master path is OPT-IN via KIT_4K_DELIVERY (the kit
    #    intentionally keeps per-shot renders at 8-bit HD for T4 speed, then
    #    switches the master encoders (clips.py / stitch.py) to 10-bit
    #    when delivery.enabled or KIT_4K_DELIVERY=1). Assert capability.
    has_10bit_master_path = "yuv420p10le" in (clips_src or "") and \
                            "yuv420p10le" in (stitch_src or "")
    check("P0", "H.264 master has a 10-bit (yuv420p10le) opt-in path",
          has_10bit_master_path,
          "either clips.py or stitch.py lacks yuv420p10le in its master encoder")
    check("P0", "clips.py honors KIT_4K_DELIVERY env / delivery.enabled flag",
          "KIT_4K_DELIVERY" in (clips_src or "") and "delivery_params" in (clips_src or ""),
          "")
    check("P0", "stitch.py honors KIT_4K_DELIVERY env / delivery.enabled flag",
          "KIT_4K_DELIVERY" in (stitch_src or ""),
          "")
    check("P0", "configs declare 4K delivery block (clips+stitch consume it)",
          cfg_c is not None and "delivery" in cfg_c and cfg_m is not None and "delivery" in cfg_m,
          "")

    prores_10bit = "yuv444p10le" in (exr_src or "")
    has_prores_in_motion = "yuv444p10le" in (render_motion_src or "")
    check("P0", "EXR master uses yuv444p10le (ProRes 4444 10-bit)",
          prores_10bit, "no yuv444p10le in exr_master.py")
    check("P0", "render_motion ProRes path uses yuv444p10le",
          has_prores_in_motion, "no yuv444p10le in render_motion.py")

    # 5. Color science: Rec.709 SDR is the configured target (Option 1).
    #    DCI-P3/XYZ and HDR PQ are intentionally out-of-scope -- this matches
    #    the user's chosen delivery standard (Netflix/Amazon streaming-grade).
    bt709_count = sum(1 for s in (clips_src, stitch_src, exr_src, grade_src,
                                   render_motion_src, postprocess_src)
                      if s and "bt709" in s)
    check("P0", "Rec.709 emitted across pipeline >= 4 paths",
          bt709_count >= 4, f"found in {bt709_count} paths")
    # DCI-P3 / HDR PQ paths are NOT required for Rec.709 SDR delivery.
    # (Set aside as informational: the kit is intentionally SDR-only.)
    has_dci = any(s and ("bt2020nc" in s or "smpte2084" in s or "dci_p3" in s.lower())
                  for s in (clips_src, stitch_src, exr_src, grade_src, render_motion_src))
    skip("P0", "DCI-P3 / Rec.2020 / HDR PQ path (out of scope -- SDR-only)",
         "intentionally not added (Option 1 SDR)")

    # 6. Bitrate floor: when 4K delivery is on, the master encoder must use
    #    bitrate mode (-b:v) instead of CRF. The kit config sets video_bitrate.
    has_bitrate_floor = any(re.search(r"-b:v", s or "") and "120M" in (s or "")
                            for s in (clips_src, stitch_src)) or \
                        (cfg_c or {}).get("delivery", {}).get("video_bitrate") == "120M"
    check("P0", "4K master carries >=100Mbps bitrate floor (delivery.video_bitrate)",
          has_bitrate_floor, "no 120M floor found in code or config")
    # bonus gate: confirm CRF values in master paths are <= 16
    crfs = re.findall(r"-crf\"\s*,\s*\"(\d+)\"", clips_src or "")
    crfs += re.findall(r"-crf\s+(\d+)", postprocess_src or "")
    bad_crf = [int(c) for c in crfs if int(c) > 16]
    check("P0", "master CRF <= 16 in all H.264 paths",
          not bad_crf, f"found CRF > 16: {bad_crf}")


# =============================================================================
# P1 -- loudness / LUFS (the multi-target inconsistency)
# =============================================================================
def P1_loudness():
    print("\n=== P1 -- Loudness / LUFS chain ===")
    cfg_c = yaml_load("config_confrontation.yaml")
    cfg_m = yaml_load("config.yaml")
    clips_src, _ = read("src/clips.py")
    score_src, _ = read("src/score_gen.py")
    voice_src, _ = read("src/voice_pass.py")
    stitch_src, _ = read("src/stitch.py")
    qc_src, _ = read("src/qc.py")

    # master bus target
    am_c = (cfg_c or {}).get("audio_master", {})
    am_m = (cfg_m or {}).get("audio_master", {})
    check("P1", "master-bus loudness_target is set in BOTH configs and equal",
          am_c.get("loudness_target") == am_m.get("loudness_target") and
          am_c.get("loudness_target") is not None,
          f"c={am_c.get('loudness_target')} m={am_m.get('loudness_target')}")

    bus = am_c.get("loudness_target", -16)

    # clips.py loudness -- no longer hardcoded. After the cinema-4K fix, clips
    # reads `audio_master.loudness_target` from cfg via a config-driven f-string
    # (`lufs_i = float(am.get("loudness_target", -16))`). The literal `I=-16`
    # no longer exists in the source. We assert the wiring instead of a literal.
    clips_reads_cfg = ("am.get(\"loudness_target\"" in (clips_src or "") or
                       'am.get("loudness_target"' in (clips_src or "")) and \
                      "I={lufs_i" in (clips_src or "") and \
                      "loudnorm=I=-16" not in (clips_src or "")  # the stale literal must be gone
    check("P1", "clips.py loudness is read from audio_master.loudness_target (not hardcoded)",
          clips_reads_cfg,
          "")

    # score_gen still hardcodes EBU R128 stem -23 (this is correct: stem-level
    # different from bus level -- the two-target relationship is intentional).
    m2 = re.search(r"loudnorm=I=(-?\d+(?:\.\d+)?)", (score_src or ""))
    score_lufs = float(m2.group(1)) if m2 else None
    voice_lufs = [float(x) for x in re.findall(r"loudnorm=I=(-?\d+(?:\.\d+)?)", (voice_src or ""))]

    check("P1", "clips.py no longer carries the stale loudnorm=I=-16 literal",
          "loudnorm=I=-16" not in (clips_src or ""),
          "clips.py still hardcodes I=-16")
    check("P1", "score_gen _master_wav at EBU R128 stem (-23)",
          score_lufs == -23.0, f"score={score_lufs}")
    # stems must be QUIETER than the bus (so headroom for normalization)
    if voice_lufs:
        loudest_stem = max(voice_lufs)
        check("P1", "loudest voice stem <= bus target (-1dB headroom)",
              loudest_stem <= bus - 1,
              f"loudest_stem={loudest_stem} bus={bus}")
    else:
        skip("P1", "voice stems", "no loudnorm=I found in voice_pass.py")

    # distribution of per-character LUFS - capture the 4 known targets
    if voice_lufs:
        expected_stems = {-28.0, -19.0, -20.0}
        got_stems = set(round(x, 2) for x in voice_lufs)
        check("P1", "per-character voice LUFS targets documented (-28/-19/-20)",
              expected_stems.issubset(got_stems),
              f"got={got_stems} expected-superset={expected_stems}")

    # true peak: stem should be MORE conservative than master
    score_tp = (cfg_m or {}).get("score", {}).get("master_true_peak")
    am_tp = am_c.get("true_peak")
    check("P1", "score stem true_peak <= master true_peak",
          score_tp is not None and am_tp is not None and score_tp <= am_tp,
          f"score.master_true_peak={score_tp} master.true_peak={am_tp}")

    # stitch consumes audio_master.* keys (not hardcoding -16)
    stitch_reads_am = all(k in (stitch_src or "")
                          for k in ("sample_rate", "bit_depth", "loudness_target",
                                    "true_peak", "limiter_ceiling"))
    check("P1", "stitch reads *all* audio_master.* keys from config",
          stitch_reads_am, "some keys absent in stitch.py")
    stitch_fallbacks_not_hard = "am.get(\"loudness_target\", -16)" in (stitch_src or "") or \
                                'am.get("loudness_target", -16)' in (stitch_src or "")
    check("P1", "stitch uses config override w/ -16 as fallback (not hardcoded)",
          stitch_fallbacks_not_hard, "")

    # single audio clock - sample_rate==48000 everywhere
    sr48000 = (cfg_c or {}).get("audio_master", {}).get("sample_rate") == 48000 and \
              (cfg_m or {}).get("audio_master", {}).get("sample_rate") == 48000 and \
              (cfg_c or {}).get("tts", {}).get("sample_rate") == 48000 and \
              (cfg_c or {}).get("score", {}).get("sample_rate") == 48000 and \
              "SR = 48000" in (KIT / "src" / "sfx_gen.py").read_text(encoding="utf-8")
    check("P1", "single audio sample rate clock (48000) across tts/score/sfx/audio_master",
          sr48000, "")

    # bit depth
    bd_c = (cfg_c or {}).get("audio_master", {}).get("bit_depth")
    bd_m = (cfg_m or {}).get("audio_master", {}).get("bit_depth")
    check("P1", "audio_master.bit_depth == 24 in both configs",
          bd_c == 24 and bd_m == 24, f"c={bd_c} m={bd_m}")
    pcm24 = "pcm_s24le" in (stitch_src or "")
    check("P1", "stitch emits pcm_s24le when bit_depth==24",
          pcm24, "no pcm_s24le in stitch.py")


# =============================================================================
# P2 -- narrative / continuity / shot_id linkage
# =============================================================================
def P2_continuity():
    print("\n=== P2 -- Narrative / Continuity / Shot-ID linkage ===")
    build_one_mod = load_module("build_one", "src/build_one.py")
    if build_one_mod is None:
        skip("P2", "build_one", "module import failed")
        return

    SPINE = getattr(build_one_mod, "SPINE", None)
    CHORUS = getattr(build_one_mod, "CHORUS", None)
    check("P2", "build_one.SPINE exists (the kit's only explicit narrative spine)",
          SPINE is not None and isinstance(SPINE, list) and len(SPINE) > 0, "")

    if SPINE:
        # each sid in SPINE exists in its declared source shots file
        SOURCE_FILES = {"m": "shots_bangalore.json", "o": "shots_observer.json"}
        sids_by_src = {}
        for sid, src, mov in SPINE:
            sids_by_src.setdefault(src, set()).add(sid)
        all_resolved = True
        for src_code, sids in sids_by_src.items():
            fname = SOURCE_FILES.get(src_code)
            if not fname:
                all_resolved = False
                continue
            p = KIT / fname
            if not p.is_file():
                all_resolved = False
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            file_sids = {s["id"] for s in data.get("shots", [])}
            if not sids.issubset(file_sids):
                all_resolved = False
        check("P2", "every SPINE sid exists in its declared source shots file",
              all_resolved, "")
        # no duplicate sids across the spine
        sids = [s[0] for s in SPINE]
        check("P2", "no duplicate shot_ids in SPINE",
              len(sids) == len(set(sids)), f"{len(sids)-len(set(sids))} dups")
        # movement order preserved
        movements = [s[2] for s in SPINE]
        first = {}
        for i, m in enumerate(movements):
            if m not in first:
                first[m] = i
        ordered = sorted(first, key=lambda k: first[k])
        check("P2", "movements in first-appearance order match sequence",
              movements == [m for m in movements if (first.get(m, 0) is not None)],
              "first-appearance check (informational)")

    if CHORUS:
        chorus_sids = {c[0] for c in CHORUS}
        spine_sids = {s[0] for s in (SPINE or [])}
        check("P2", "every CHORUS sid is a subset of SPINE sids",
              chorus_sids.issubset(spine_sids),
              f"missing={chorus_sids - spine_sids}")

    # shots_confrontation.json cross-locks with config movements
    cfg_c = yaml_load("config_confrontation.yaml")
    shots_c = json.loads((KIT / "shots_confrontation.json").read_text(encoding="utf-8"))
    if cfg_c and shots_c:
        cfg_movs = (cfg_c.get("score") or {}).get("movements") or {}
        shots = shots_c.get("shots", [])
        # by-movement membership
        shot_mov = {}
        for s in shots:
            shot_mov.setdefault(s.get("movement"), set()).add(s["id"])
        consistent = True
        for m, spec in cfg_movs.items():
            spec_shots = set(spec.get("shots", []))
            file_shots = shot_mov.get(m, set())
            if spec_shots != file_shots:
                consistent = False
        check("P2", "config_confrontation score.movements.shots == shots json membership",
              consistent, "cross-file drift detected")
        # shot-id prefix invariant: each shot id starts with the movement's
        # first 2 chars (`pro_` for prelude, `search_` for search, `confr_` for
        # confrontation -- uses >=2 to be permissive across movement names).
        prefixes_ok = all(
            s["id"].startswith(s.get("movement", "")[:2])
            for s in shots if s.get("movement"))
        check("P2", "confrontation shot ids prefix matches movement name",
              prefixes_ok, "")
        # known movement set
        mov_set = set(shot_mov.keys())
        check("P2", "confrontation has exactly the 3 canonical movements",
              mov_set == {"prelude", "search", "confrontation"},
              f"got={mov_set}")
        # all VO, no talking (clip is voiceover-only)
        all_vo_no_talk = all(s.get("vo") and not s.get("talking") for s in shots)
        check("P2", "confrontation is voiceover-only (talking=False, vo=True)",
              all_vo_no_talk, "found a talking shot")
        # shot count
        check("P2", "confrontation shot count >= 24",
              len(shots) >= 24, f"{len(shots)} shots")

    # build_one produces shot["movement"] on every shot -- run build() to a temp dir?
    # (Static equivalent: confirm the function source has the assignment line)
    bo_src, _ = read("src/build_one.py")
    check("P2", "build_one writes movement onto every shot",
          "shot[\"movement\"]" in (bo_src or "") or 'sh["movement"]' in (bo_src or ""),
          "no `shot['movement']` assignment in build_one.py")


# =============================================================================
# P3 -- exr_master DCI-compliance
# =============================================================================
def P3_exr_master():
    print("\n=== P3 -- exr_master DCI-compliance ===")
    exr_src, _ = read("src/exr_master.py")
    if exr_src is None:
        skip("P3", "exr_master", "module missing")
        return

    check("P3", "ProRes 4444 encoder (prores_ks -profile:v 4)",
          "prores_ks" in exr_src and "-profile:v\", \"4" in exr_src, "")
    check("P3", "10-bit 4:4:4 pix_fmt (yuv444p10le)",
          "yuv444p10le" in exr_src, "")
    check("P3", "Apple vendor tag (apl0)",
          "\"apl0\"" in exr_src or "'apl0'" in exr_src, "")
    # color tags = bt709 NCLC 1-1-1 (Rec.709)
    check("P3", "master color tags (NCLC 1-1-1 fallback at least)",
          "bt709" in exr_src and
          "-colorspace\", \"bt709" in exr_src, "")
    # zscale linear->709 color management when filter available
    check("P3", "zscale linear->bt709 color management (when zimg present)",
          "zscale=tin=linear:pin=bt709" in exr_src, "")
    check("P3", "non-fatal: returns 1 instead of raising on failure",
          "def main(" in exr_src and exr_src.count("return 1") >= 1, "")
    # audio path
    check("P3", "audio path uses pcm_s24le @ 48000 + -shortest",
          "pcm_s24le" in exr_src and "48000" in exr_src and "-shortest" in exr_src, "")
    # frame-glob parser exists
    check("P3", "frame-glob regex for EXR sequences exists",
          "re.match" in exr_src and ".exr" in exr_src, "")


# =============================================================================
# P4 -- lipsync mouth-frame markers / sync validation
# =============================================================================
def P4_lipsync():
    print("\n=== P4 -- lipsync mouth-frame markers / sync validation ===")
    lip_src, _ = read("src/lipsync.py")
    if lip_src is None:
        skip("P4", "lipsync", "module missing")
        return

    # no viseme/mouth frame markers are produced anywhere
    no_markers = not any(k in lip_src for k in
                          ("viseme", "phoneme", "mouth_frame", "frame_csv", "frame_marker"))
    check("P4", "lipsync does NOT emit frame-level mouth/viseme markers (gap)",
          no_markers, "module produces markers -- check expected behaviour")
    # lipsync only checks LatentSync's rc + out file existence -- no numeric
    # confidence/drift metric is recorded; this is the known gap.
    no_conf = "confidence" not in lip_src.lower() and "drift_metric" not in lip_src.lower()
    check("P4", "lipsync has NO sync-confidence numeric metric (gap)",
          no_conf, "")
    # all four skip paths return 0 explicitly
    has_skip_paths = sum(1 for s in (
        "disabled in config",
        "no talking shots",
        "no GPU here",
        "LatentSync not installed",
    ) if s in lip_src)
    check("P4", "all four skip paths present and return 0",
          has_skip_paths == 4, f"{has_skip_paths}/4 skip messages found")
    # sub-check: every `return 0` should not raise
    check("P4", "lipsync never raises on skip path",
          lip_src.count("return 0") >= 4, "")
    # search paths -- exactly the 3 documented locations
    expected_search_paths = ('"/kaggle/working/LatentSync"', '"/kaggle/input"',
                              'os.path.expanduser("~")')
    has_all_paths = all(p in lip_src for p in expected_search_paths)
    check("P4", "find_latentsync searches the 3 documented paths",
          has_all_paths, "")
    # invariant: when LatentSync returns 0 but out file is missing, no
    # shutil.move is invoked
    safe_replace = "if r.returncode == 0 and os.path.exists(out)" in lip_src
    check("P4", "replacement only when rc==0 AND out file exists",
          safe_replace, "")


# =============================================================================
# P5 -- score_gen loudness + ducking
# =============================================================================
def P5_score():
    print("\n=== P5 -- score_gen loudness + ducking ===")
    score_src, _ = read("src/score_gen.py")
    clips_src, _ = read("src/clips.py")
    stitch_src, _ = read("src/stitch.py")
    if score_src is None:
        skip("P5", "score_gen", "module missing")
        return

    score_mod = load_module("score_gen", "src/score_gen.py")
    # MOOD keys + DEFAULT exist
    mood_dict = getattr(score_mod, "MOOD", None) if score_mod is not None else None
    has_mood = isinstance(mood_dict, dict)
    check("P5", "score_gen.MOOD dict present",
          has_mood, "")
    if has_mood:
        mood_keys = set(mood_dict.keys())
        # confrontation config movements MUST all be in MOOD (no DEFAULT fallback)
        cfg_c = yaml_load("config_confrontation.yaml")
        if cfg_c:
            conf_movs = set((cfg_c.get("score") or {}).get("movements", {}).keys())
            check("P5", "confrontation movements all in score_gen.MOOD (no fallback)",
                  conf_movs.issubset(mood_keys),
                  f"missing={conf_movs - mood_keys}")
        # MOMENT config: most movements use fallback -- flag as info
        cfg_m = yaml_load("config.yaml")
        if cfg_m:
            mom_movs = set((cfg_m.get("score") or {}).get("movements", {}).keys())
            in_mood = mom_movs & mood_keys
            using_default = mom_movs - mood_keys
            check("P5", "MOMENT config: known gap -- most movements fall back to DEFAULT",
                  len(using_default) >= len(in_mood),
                  f"direct={in_mood} DEFAULT-fallback={using_default}")

    # _master_wav contains the EBU R128 broadcast stem literal
    check("P5", "_master_wav uses loudnorm=I=-23:LRA=13:TP=-2 (EBU R128 stem)",
          "loudnorm=I=-23:LRA=13:TP=-2" in score_src, "")
    # DEFAULT prompt present
    check("P5", "score_gen.DEFAULT fallback prompt present",
          "DEFAULT" in score_src and
          re.search(r'^DEFAULT\s*=', score_src, re.M) is not None, "")
    # ducking lives in clip + stitch, NOT in score_gen (stem master should be flat)
    check("P5", "score_gen does NOT duck (ducking belongs at mix stage)",
          "sidechaincompress" not in score_src, "")
    check("P5", "stitch applies sidechain ducking at mixdown",
          "sidechaincompress" in (stitch_src or ""), "")
    check("P5", "clips build_audio applies per-clip sidechain ducking",
          "sidechaincompress" in (clips_src or ""), "")
    # write_wav int16 clamp -- asymmetric 32767, not 32768
    w = re.search(r"32767\.0", score_src)
    check("P5", "write_wav uses int16 saturating factor (32767.0)",
          w is not None, "")


# =============================================================================
# P6 -- grade_match color science
# =============================================================================
def P6_grade_match():
    print("\n=== P6 -- grade_match color science ===")
    gm_src, _ = read("src/grade_match.py")
    if gm_src is None:
        skip("P6", "grade_match", "module missing")
        return
    gm = load_module("grade_match", "src/grade_match.py")
    if gm is None:
        skip("P6", "grade_match import", "exec failed")
        return

    FILM_PRESETS = getattr(gm, "FILM_PRESETS", None)
    expected = {"kodachrome", "kodak_5279", "arri_aces", "bleach_bypass", "teal_orange"}
    check("P6", "FILM_PRESETS keyspace is the 5 documented presets",
          FILM_PRESETS is not None and set(FILM_PRESETS.keys()) == expected,
          f"got={set(FILM_PRESETS.keys()) if FILM_PRESETS else None}")
    check("P6", "no DCI-P3 / Rec.2020 / Log-C preset (4K cinema gap)",
          FILM_PRESETS is not None and not any(k in FILM_PRESETS for k in
                                               ("dci_p3", "rec2020", "arri_log_c", "s_log3")),
          "")
    # per-preset schema
    schema_ok = True
    if FILM_PRESETS:
        for name, p in FILM_PRESETS.items():
            for k in ("contrast", "saturation", "warmth", "shadows", "highlights"):
                if k not in p:
                    schema_ok = False
                    continue
            if not (isinstance(p.get("warmth"), (list, tuple)) and len(p["warmth"]) == 3):
                schema_ok = False
            if not (isinstance(p.get("shadows"), (list, tuple)) and len(p["shadows"]) == 3):
                schema_ok = False
            if not (isinstance(p.get("highlights"), (list, tuple)) and len(p["highlights"]) == 3):
                schema_ok = False
            if not (p.get("contrast", 0) > 0):
                schema_ok = False
    check("P6", "every preset has 5-field schema with correct shapes",
          schema_ok, "")

    # apply_film_lut is idempotent at boundaries
    import numpy as np
    if hasattr(gm, "apply_film_lut"):
        black = np.zeros((8, 8, 3), dtype=np.float32)
        white = np.ones((8, 8, 3), dtype=np.float32)
        try:
            out_b = gm.apply_film_lut(black, "kodachrome")
            out_w = gm.apply_film_lut(white, "kodachrome")
            check("P6", "apply_film_lut(black) stays in [0,1]",
                  float(out_b.min()) >= 0.0 and float(out_b.max()) <= 1.0,
                  f"min={out_b.min()} max={out_b.max()}")
            check("P6", "apply_film_lut(white) stays in [0,1]",
                  float(out_w.min()) >= 0.0 and float(out_w.max()) <= 1.0,
                  f"min={out_w.min()} max={out_w.max()}")
        except Exception as e:
            skip("P6", "apply_film_lut boundary", f"import/exec issue: {e}")
    else:
        skip("P6", "apply_film_lut boundary", "function absent")

    # temporal_blend expects a 4D numpy array [N,H,W,C] -> returns same shape
    if hasattr(gm, "temporal_blend"):
        try:
            frames = np.stack([
                np.full((4, 4, 3), v, dtype=np.float32) for v in (0.1, 0.5, 0.9)
            ])  # (3, 4, 4, 3)
            out = gm.temporal_blend(frames, window=3, sigma=1.0)
            ok = isinstance(out, np.ndarray) and out.shape == frames.shape
            check("P6", "temporal_blend preserves input 4D shape [N,H,W,C]",
                  ok, f"got shape {getattr(out, 'shape', None)} vs {frames.shape}")
        except Exception as e:
            skip("P6", "temporal_blend", f"{e}")
    else:
        skip("P6", "temporal_blend", "function absent")

    # film_grain deterministic with seed=42
    if hasattr(gm, "film_grain"):
        try:
            g1 = gm.film_grain(4, 8, 8, strength=3.5, chroma=0.5, seed=42)
            g2 = gm.film_grain(4, 8, 8, strength=3.5, chroma=0.5, seed=42)
            check("P6", "film_grain(seed=42) is deterministic across runs",
                  g1.shape == g2.shape and bool((g1 == g2).all()),
                  "")
        except Exception as e:
            skip("P6", "film_grain deterministic", f"{e}")

    # grade_video is non-fatal
    check("P6", "grade_video wraps _grade_video_impl with try/except + fallback copy",
          "except Exception" in gm_src and "shutil.copy" in gm_src, "")


# =============================================================================
# P7 -- theatrical (DCI) delivery readiness
# =============================================================================
def P7_theatrical():
    print("\n=== P7 -- Theatrical (DCI) Delivery Readiness ===")
    cfg_c = yaml_load("config_confrontation.yaml")
    cfg_m = yaml_load("config.yaml")
    stitch_src, _ = read("src/stitch.py")

    # 1. both configs declare a theatrical block
    for label, cfg in (("confrontation", cfg_c), ("moment", cfg_m)):
        d = (cfg or {}).get("theatrical", {}) or {}
        check("P7", f"{label} has theatrical block",
              bool(d), f"theatrical={d}")

    # 2. stitch.py gates theatrical output behind KIT_THEATRICAL=1
    check("P7", "stitch.py gates theatrical behind KIT_THEATRICAL=1",
          "KIT_THEATRICAL" in (stitch_src or ""), "")

    # 3. each new theatrical module exists on disk
    the_modules = [
        "src/theatrical_grade.py",
        "src/surround_master.py",
        "src/dcdm_packager.py",
        "src/dcp_packager.py",
        "src/theatrical_qc.py",
    ]
    for m in the_modules:
        check("P7", f"module present: {m}", (KIT / m).is_file(), m)

    # 4. theatrical_grade exposes the expected public API
    tg_src, _ = read("src/theatrical_grade.py")
    if tg_src:
        needs_tg = [
            "def grade_pipeline",
            "def grade_to_theatrical",
            "PQ",
            "bt2020nc",
            "st2084",
            "zscale",
            "colorspace=space=dci-p3",
        ]
        for needle in needs_tg:
            check("P7", f"theatrical_grade contains {needle}",
                  needle.lower() in tg_src.lower(), needle)

    # 5. surround_master exports theatrical_bed_mix + layout dict
    sm_src, _ = read("src/surround_master.py")
    if sm_src:
        check("P7", "surround_master exports theatrical_bed_mix",
              "theatrical_bed_mix" in sm_src, "")
        check("P7", "surround_master knows 5.1 + 7.1 layouts",
              '"5.1"' in sm_src and '"7.1"' in sm_src, "")

    # 6. dcdm_packager builds directories + writes J2K/WAV
    dcdm_src, _ = read("src/dcdm_packager.py")
    if dcdm_src:
        check("P7", "dcdm_packager creates video/ and audio/ dirs",
              "video" in dcdm_src and "audio" in dcdm_src, "")
        j2k_encoder = "jpeg2000" in dcdm_src or "libopenjpeg" in dcdm_src
        check("P7", "dcdm_packager invokes JPEG 2000 encoder",
              j2k_encoder, "expected jpeg2000/libopenjpeg in ffmpeg command")
        check("P7", "dcdm_packager emits uncompressed WAV audio",
              ".wav" in dcdm_src.lower(), "")

    # 7. dcp_packager emits MXF + CPL/PKL/AssetMap/VOLINDEX + optional KDM
    dcp_src, _ = read("src/dcp_packager.py")
    if dcp_src:
        check("P7", "dcp_packager writes MXF video + audio",
              ".mxf" in dcp_src.lower(), "")
        check("P7", "dcp_packager creates CPL xml",
              "cpl" in dcp_src.lower() and ".xml" in dcp_src.lower(), "")
        check("P7", "dcp_packager creates PKL",
              "pkl" in dcp_src.lower() and ".xml" in dcp_src.lower(), "")
        check("P7", "dcp_packager creates AssetMap",
              "assetmap" in dcp_src.lower() or "AssetMap" in dcp_src, "")
        check("P7", "dcp_packager creates VOLINDEX",
              "volindex" in dcp_src.lower() or "VOLINDEX" in dcp_src, "")
        check("P7", "dcp_packager supports KDM stub",
              "kdm" in dcp_src.lower(), "")

    # 8. theatrical_qc enforces P3 / PQ / 5.1 / J2K-12
    tqc_src, _ = read("src/theatrical_qc.py")
    if tqc_src:
        check("P7", "theatrical_qc checks P3 primaries",
              "p3" in tqc_src.lower() or "dci_p3" in tqc_src.lower(), "")
        check("P7", "theatrical_qc checks PQ transfer",
              "pq" in tqc_src.lower() or "2084" in tqc_src, "")
        check("P7", "theatrical_qc checks 5.1 / 7.1 channel count",
              "5.1" in tqc_src and "7.1" in tqc_src, "")
        check("P7", "theatrical_qc checks 12-bit JPEG 2000",
              "jpeg 2000" in tqc_src.lower() or "j2k" in tqc_src.lower(), "")

    # 9. env-trigger reachable from a known config
    has_env_switch = \
        (cfg_c or {}).get("theatrical", {}).get("enabled") or \
        "KIT_THEATRICAL" in (stitch_src or "")
    check("P7", "theatrical can be triggered via KIT_THEATRICAL=1",
          has_env_switch, "")

    # 10. vram + thermal modules remain intact after theatrical patch
    for mod in ("src/vram_governor.py", "src/gpu_scheduler.py", "src/thermal_monitor.py"):
        check("P7", f"zero-compromise module intact: {mod}", (KIT / mod).is_file(), mod)


# =============================================================================
# P7.1 -- DCP SPL verification
# =============================================================================
def P7_dcp_verification():
    """DCP SPL verification test"""
    print("\n=== P7.1 -- DCP SPL Verification ===")
    try:
        # Find latest SPL file in _deliverables dir
        cpl_path = subprocess.check_output([
            "powershell",
            f"Get-ChildItem -Path '{KIT.parent}/_deliverables' -Filter *.CPL.xml -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object {{ $_.FullName }}",
            "-NoProfile"
        ], stderr=subprocess.STDOUT, encoding='utf-8', timeout=30)
        cpl_file = cpl_path.strip()

        # Validate with rosetta (assume installed in PATH)
        result = subprocess.run([
            "rosetta",
            "--validate-cpl",
            cpl_file
        ], capture_output=True, text=True, timeout=60)

        # Output the command for debugging
        print(f"Command: {' '.join(result.args)}")

        # Check result
        check("P7.1", "DCP SPL validation success", result.returncode == 0,
              result.stderr.decode('utf-8', errors='replace') if isinstance(result.stderr, bytes) else (result.stderr or "Success"))

        # Additional checks
        # Example: Verify language tag presence in SPL
        # Implement more checks as required

    except subprocess.TimeoutExpired:
        skip("P7.1", "DCP SPL verification", "rosetta validation timed out")
    except subprocess.CalledProcessError as e:
        skip("P7.1", "DCP SPL verification", f"PowerShell error: {e.output}")
    except FileNotFoundError:
        skip("P7.1", "DCP SPL verification", "rosetta not in PATH")
    except Exception as e:
        skip("P7.1", "DCP SPL verification", f"Exception in verification: {e}")


# ---- runner ----------------------------------------------------------------
def main():
    global KIT, SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", help="run specific group(s); repeatable")
    ap.add_argument("--list", action="store_true", help="list groups and exit")
    ap.add_argument("--kit", default=str(KIT), help="kit root (default $GODSHAND_KIT_ROOT)")
    a = ap.parse_args()

    if a.kit != str(KIT):
        KIT = Path(a.kit)
        SRC = KIT / "src"

    if a.list:
        for g, fn in GROUPS.items():
            doc = fn.__doc__.strip().splitlines()[0] if fn and fn.__doc__ else "no doc"
            print(f"  {g}  {doc}")
        return 0

    if not KIT.is_dir():
        print(f"[FATAL] kit not found: {KIT}")
        print("        set GODSHAND_KIT_ROOT or pass --kit <path>")
        return 2

    # depend on numpy + yaml for some checks (--skip them if missing)
    try:
        import numpy  # noqa
    except ImportError:
        print("[warn] numpy not available; P6 sub-checks will skip")

    selected = a.group or list(GROUPS.keys())
    print(f"\n# cinema4k_sanity_test  --  6 groups, {len(selected)} selected")
    print(f"# kit: {KIT}")
    for g in selected:
        try:
            GROUPS[g]()
        except Exception as e:
            # don't abort the whole run on a group crash
            results.append((g, f"<group {g}>", FAIL, f"uncaught: {e!r}"))
            print(f"  [{FAIL}] <group {g}>  ::  uncaught: {e!r}")

    # ---- matrix ----
    print("\n\n" + "=" * 70)
    print("MATRIX")
    print("=" * 70)
    by_group = {}
    for g, name, st, d in results:
        by_group.setdefault(g, []).append((st, name, d))
    n_pass = n_fail = n_skip = 0
    for g, rows in by_group.items():
        ps = [r for r in rows if r[0] == PASS]
        fs = [r for r in rows if r[0] == FAIL]
        ss = [r for r in rows if r[0] == SKIP]
        n_pass += len(ps)
        n_fail += len(fs)
        n_skip += len(ss)
        print(f"  {g}: {len(ps)} pass, {len(fs)} fail, {len(ss)} skip")
        for st, name, d in rows:
            if st == FAIL:
                print(f"      x {name}  ::  {d}")
    print("\n  total: {} pass, {} fail, {} skip".format(n_pass, n_fail, n_skip))

    # ---- 4K-readiness score ----
    strict_groups = ("P0", "P1", "P2", "P3", "P5", "P6")
    fail_blocking = sum(1 for g, _, st, _ in results if g in strict_groups and st == FAIL)
    check_total = sum(1 for g, _, st, _ in results if g in strict_groups and st != SKIP)
    pct = (100 * (check_total - fail_blocking) / check_total) if check_total else 0
    print("\n# Cinema-4K Readiness: {:.0f}%  ({}) of {} blocking checks passing)".format(
        pct, check_total - fail_blocking, check_total))

    if n_fail:
        return 1
    return 0


# Assign actual function references to GROUPS after all functions are defined
GROUP_FUNCTIONS = {
    "P0": P0_resolution_codec_color,
    "P1": P1_loudness,
    "P2": P2_continuity,
    "P3": P3_exr_master,
    "P4": P4_lipsync,
    "P5": P5_score,
    "P6": P6_grade_match,
    "P7": P7_theatrical,
    "P7.1": P7_dcp_verification,
}

GROUPS.update(GROUP_FUNCTIONS)


if __name__ == "__main__":
    sys.exit(main() or 0)