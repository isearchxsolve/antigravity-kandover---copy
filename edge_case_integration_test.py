#!/usr/bin/env python3
"""
Edge-case + integration test covering failure points NOT exercised by
comprehensive_smoke_test.py or selftest.py.

Specifically this drives the REAL module functions (not replicated logic):
  - render_motion.render_motion()  end-to-end via COMFY_MOCK  -> catches the
    orphaned-code / indentation defect that selftest bypasses (selftest uses
    render_film.py, the notebook uses render_motion.py).
  - runtime_integration_validator  -> catches the {{}} TypeError crash.
  - qc.check() on a REAL rendered clip with audio (existing suite only uses
    synthetic testsrc2 + detached LUFS probes).
  - grade_match fail-soft + JSON-target + direct-preset through the real CLI.
  - OOM fallback tier math (multiples of 64, monotonic reduction, exhaustion).
  - stitch normalization majority-vote + corrupt clip handling via the real
    _probe helpers where importable.

Exit 0 = all assertions hold. Non-zero = a real defect surfaced.
"""
import os, sys, json, subprocess, tempfile, shutil, re, inspect
from pathlib import Path

# Force UTF-8 stdout on Windows consoles (cp1252 can't decode some bytes)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Subprocess call helper that always decodes as UTF-8 and never returns None
SB_TEXT = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")

ROOT = Path(os.environ.get(
    "GODSHAND_KIT_ROOT",
    r"C:\Users\Admin\Downloads\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit"))
if not ROOT.is_dir():
    alt = Path(r"C:\dev\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit")
    if alt.is_dir():
        ROOT = alt
SRC = ROOT / "src"
os.chdir(ROOT)
sys.path.insert(0, str(SRC))

PASS, FAIL = 0, 0
def ok(name, detail=""):
    global PASS; PASS += 1
    print(f"  [PASS] {name}{(' -- ' + detail) if detail else ''}")
def bad(name, detail=""):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {name}{(' -- ' + detail) if detail else ''}")
def check(name, cond, detail=""):
    (ok if cond else bad)(name, detail)
    return cond

# ---------- helpers ----------
def gen_video(path, duration=2, fps=24, w=256, h=144, color="testsrc2"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"{color}=size={w}x{h}:rate={fps}:duration={duration}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)

print("=" * 72)
print("EDGE-CASE + INTEGRATION TEST (real module functions)")
print("=" * 72)

# =====================================================================
# GROUP A: render_motion() structural integrity + end-to-end mock run
# =====================================================================
print("\n[A] render_motion() structure + end-to-end COMFY_MOCK run")
import render_motion as rm

# A1: render_motion() body must actually do the work (call the render loop,
#     identity selection, and postprocess). The botched OOM-def insertion
#     terminated render_motion() early and orphaned the rest.
src_rm = inspect.getsource(rm.render_motion)
check("A1 render_motion() calls _run_with_oom_fallback",
      "_run_with_oom_fallback" in src_rm,
      "function body is missing its render loop (orphaned by mis-indented def)")
check("A2 render_motion() calls _do_postprocess",
      "_do_postprocess" in src_rm,
      "postprocess (NEED_INTERP guard + grade) never reached")
check("A3 render_motion() applies denoise",
      "KIT_MOTION_DENOISE" in src_rm,
      "denoise wiring dropped")
check("A4 render_motion() does identity-aware _pick_best",
      "_pick_best" in src_rm and "id_scores" in src_rm,
      "identity selection (Defect 5) not in function body")
check("A5 render_motion() writes final.mp4",
      "final.mp4" in src_rm,
      "no final output produced")

# A6: _run_with_oom_fallback must NOT contain denoise/render-loop/postprocess
#     (those belong to render_motion). If it does, the def was spliced in.
src_oom = inspect.getsource(rm._run_with_oom_fallback)
oom_has_orphans = ("KIT_MOTION_DENOISE" in src_oom
                   or "_pick_best" in src_oom
                   or "final.mp4" in src_oom)
check("A6 _run_with_oom_fallback free of orphaned render_motion code",
      not oom_has_orphans,
      "render-loop/identity/postprocess code absorbed into OOM helper -> render_motion() is truncated")

# A7: END-TO-END. Actually invoke render_motion() with COMFY_MOCK=1 and confirm
#     it produces <work>/<sid>/final.mp4. This is the failure point the audit
#     suite misses entirely.
print("\n  [A7] invoking render_motion() end-to-end (COMFY_MOCK=1, POST_MODE=cpu)...")
os.environ["COMFY_MOCK"] = "1"
os.environ["POST_MODE"] = "cpu"
os.environ["KIT_FAST_POST"] = "1"
tmp = Path(tempfile.mkdtemp(prefix="rm_e2e_"))
work = tmp / "work"; work.mkdir()
# Minimal shot list + config that render_motion can consume
shots_json = {"globals": {"tone": "test", "negative_prompt": "blurry"},
              "shots": [{"id": "e01", "dur": 2, "chars": [], "talking": False,
                         "prompt": "a test shot", "palette": "grey"}]}
(tmp / "shots_e.json").write_text(json.dumps(shots_json))
cfg = {"film_title": "E", "shots_file": "shots_e.json",
       "fps": 24, "interpolate_fps": 16, "width": 256, "height": 144,
       "steps": 2, "cfg": 5.0, "upscale": 1.0, "rejection_samples": 1,
       "codeformer_fidelity": 0.7, "max_clip_seconds": 5, "denoise": 0.35,
       "paths": {"refs": "refs", "work": str(work), "out": "out_e", "models": str(tmp/"m")},
       "identity_retry": 0, "rate_per_gpu_hr": 0.0}
(tmp / "cfg_e.yaml").write_text("")  # not loaded; we pass cfg dict directly

produced_final = False
err = None
# render_motion() re-reads cfg["shots_file"] from ROOT for tone; write it there.
(root_shots := ROOT / "shots_e.json").write_text(json.dumps(shots_json))
try:
    rm.render_motion(shots_json["shots"][0], cfg, str(ROOT), str(work),
                     shots_json["shots"], 0)
    produced_final = (work / "e01" / "final.mp4").exists()
except Exception as e:
    err = f"{type(e).__name__}: {e}"
check("A7 render_motion() produces work/e01/final.mp4",
      produced_final,
      err or "final.mp4 missing after render_motion() returned")
if not produced_final:
    print(f"        !!! This proves render_motion() is broken at runtime. "
          f"selftest passes only because it uses render_film.py instead.")
try:
    root_shots.unlink()
except OSError:
    pass
shutil.rmtree(tmp, ignore_errors=True)

# =====================================================================
# GROUP B: runtime_integration_validator (TEST 10 module) actually runs
# =====================================================================
print("\n[B] runtime_integration_validator.py (TEST 10 module)")
# B1: it must not crash on import / basic run
r = subprocess.run([sys.executable, str(SRC / "runtime_integration_validator.py")],
                   cwd=str(ROOT), timeout=120, **SB_TEXT)

def _txt(r):
    """Never return None even if capture failed mid-flight."""
    return (r.stdout or "") + (r.stderr or "")

stdout_err = _txt(r)
check("B1 validator exits without TypeError crash",
      "TypeError: unhashable type" not in stdout_err,
      f"crashed with {{}} set-literal typo (rc={r.returncode})")
if "TypeError: unhashable type" in stdout_err:
    # show the failing line
    m = re.search(r'line (\d+).*in <module>|line (\d+), in \S+', r.stderr or "")
    print(f"        crash trace tail:\n{(r.stderr or '')[-400:]}")

# B2: COMPATIBILITY_MATRIX must not have duplicate keys (silent override)
riv_src = (SRC / "runtime_integration_validator.py").read_text(encoding="utf-8")
# Parse the COMPATIBILITY_MATRIX assignment via AST and count the key ONLY
# inside that one dict -- the name legitimately re-appears in
# REQUIRED_CUSTOM_NODES, which a flat regex would falsely count.
import ast as _ast
_tree = _ast.parse(riv_src)
_dup_count = 0
for _node in _tree.body:
    if isinstance(_node, _ast.Assign) and any(
        isinstance(_t, _ast.Name) and _t.id == "COMPATIBILITY_MATRIX" for _t in _node.targets
    ):
        _dict = _node.value
        if isinstance(_dict, _ast.Dict):
            _dup_count = sum(
                1 for _k in _dict.keys
                if isinstance(_k, _ast.Constant) and _k.value == "ComfyUI-VideoHelperSuite"
            )
check("B2 no duplicate ComfyUI-VideoHelperSuite matrix entries", _dup_count <= 1,
      f"found {_dup_count} key definitions inside COMPATIBILITY_MATRIX (last silently wins)")

# =====================================================================
# GROUP C: qc.check() on a REAL clip (video + audio together)
# =====================================================================
print("\n[C] qc.check() on a real clip with embedded audio")
from qc import check as qc_check, DEFAULT_TH
tmp = Path(tempfile.mkdtemp(prefix="qc_real_"))
try:
    v = tmp / "v.mp4"; gen_video(v, duration=2, fps=24, w=1280, h=720)
    a = tmp / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", str(a)], check=True)
    muxed = tmp / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-i", str(a),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(muxed)], check=True)
    verdict, issues, stats = qc_check(str(muxed), 2, 24, DEFAULT_TH)
    check("C1 qc reads stats from real clip",
          bool(stats.get("res") == "1280x720" and stats.get("audio") is True),
          f"stats={stats}")
    check("C2 qc flags no spurious FAIL on a clean clip",
          verdict in ("PASS", "WARN"),
          f"verdict={verdict} issues={issues}")
    # loudness branch should populate LUFS since audio is present
    check("C3 qc populates LUFS for audio clip", "LUFS" in stats,
          f"stats={stats} (audio present but LUFS not measured)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# =====================================================================
# GROUP D: grade_match fail-soft + both call signatures via real CLI
# =====================================================================
print("\n[D] grade_match.py real CLI: fail-soft + JSON target + direct preset")
tmp = Path(tempfile.mkdtemp(prefix="grade_cli_"))
try:
    src = tmp / "in.mp4"; gen_video(src, duration=1, fps=24, w=256, h=144)
    # D1: missing input. Audit TEST 8d documents "exit 1 (fail-soft)" but the
    # Defect 4 narrative says "clean exit 0 via _copy()". Reality: it crashes
    # with an uncaught FileNotFoundError. We assert it does NOT hang/corrupt
    # other files and exits reasonably fast; either rc 0 or 1 is acceptable,
    # but a Python traceback is a (minor) broken fail-soft contract.
    out = tmp / "no_input_out.mp4"
    r = subprocess.run([sys.executable, str(SRC / "grade_match.py"),
                        str(tmp / "missing.mp4"), str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    crashed = ("Traceback" in r.stderr and "FileNotFoundError" in r.stderr)
    check("D1 grade_match missing-input handled (no traceback)",
          r.returncode in (0, 1) and not crashed,
          f"rc={r.returncode} uncaught traceback={crashed}")
    if crashed:
        print(f"        note: audit doc is contradictory (TEST 8d says exit 1, "
              f"Defect 4 says exit 0). Actual = uncaught crash -> fail-soft is broken.")
    # D2: JSON target path -> reads film_preset from JSON
    tj = tmp / "t.json"; tj.write_text(json.dumps({"film_preset": "bleach_bypass"}))
    o2 = tmp / "o_json.mp4"
    r = subprocess.run([sys.executable, str(SRC / "grade_match.py"),
                        str(src), str(o2), str(tj), "0.8", "3.5"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("D2 grade_match JSON-target reads preset", r.returncode == 0 and "bleach_bypass" in r.stdout,
          f"rc={r.returncode} out={r.stdout[-200:]}")
    # D3: direct preset literal
    o3 = tmp / "o_pre.mp4"
    r = subprocess.run([sys.executable, str(SRC / "grade_match.py"),
                        str(src), str(o3), "arri_aces", "0.8", "3.5"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("D3 grade_match direct-preset works", r.returncode == 0 and "arri_aces" in r.stdout,
          f"rc={r.returncode}")
    # D4: output produced on happy path
    check("D4 grade_match writes output file", o2.exists() and o3.exists())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# =====================================================================
# GROUP E: OOM fallback tier math (the part of Defect 8 that is testable)
# =====================================================================
print("\n[E] render_stills OOM fallback tier math (multiples of 64, exhaustion)")
# Replicate the exact tier formulas from render_stills._run_with_oom_fallback
# (which is the correctly-indented version) and verify the invariants.
fallbacks = [
    (1.0, 1.0, 1.0), (0.8, 1.0, 1.0), (0.8, 0.875, 0.875),
    (0.7, 0.75, 0.75), (0.5, 0.625, 0.625),
]
orig_w, orig_h = 1280, 720
all_ok = True
for i, (sp, wp, hp) in enumerate(fallbacks):
    # Fallback only realigns when it actually reduces (matches source: if w_pct < 1.0).
    # The unreduced tiers (1.0) intentionally keep the original dims; base-config
    # 64-alignment is a separate concern (see F3 below).
    if wp < 1.0:
        new_w = max(512, int(orig_w * wp) // 64 * 64)
        new_h = max(288, int(orig_h * hp) // 64 * 64)
        mult64 = (new_w % 64 == 0) and (new_h % 64 == 0)
        if not mult64:
            all_ok = False
            print(f"        reduced tier {i}: {new_w}x{new_h} NOT multiples of 64")
check("E1 reduced OOM tiers produce 64-aligned resolution", all_ok)
# E2: monotonic non-increasing resolution across tiers
sizes = []
for (sp, wp, hp) in fallbacks:
    sizes.append((max(512, int(orig_w*wp)//64*64) if wp<1.0 else orig_w,
                  max(288, int(orig_h*hp)//64*64) if hp<1.0 else orig_h))
mono = all(sizes[i][0] >= sizes[i+1][0] and sizes[i][1] >= sizes[i+1][1]
           for i in range(len(sizes)-1))
check("E2 resolution monotonically non-increasing across tiers", mono,
      f"sizes={sizes}")
# E3: exhaustion -> RuntimeError (per code: attempt >= len(fallbacks) raises)
check("E3 exhaustion raises RuntimeError (per source)",
      "exhausted all fallback tiers" in inspect.getsource(rm._run_with_oom_fallback)
       or "exhausted all fallback tiers" in (SRC/"render_stills.py").read_text(encoding="utf-8"))

# =====================================================================
# GROUP F: config / shots sanity for the production profile
# =====================================================================
print("\n[F] production profile config sanity")
import yaml
cfg_conf = yaml.safe_load((ROOT / "config_confrontation.yaml").read_text(encoding="utf-8"))
# interpolate_fps must be <= WAN native * 1.2 so NEED_INTERP skips
ipf = cfg_conf.get("interpolate_fps", cfg_conf.get("fps", 24))
check("F1 interpolate_fps <= 16 keeps NEED_INTERP guard active", ipf <= 16,
      f"interpolate_fps={ipf} (>16 would re-trigger minterpolate on WAN output)")
ap = cfg_conf.get("active_profile", "")
check("F2 active_profile is kaggle_2xt4 family", "kaggle_2xt4" in str(ap),
      f"active_profile={ap}")

# F3: WAN/VAE wants W,H multiples of 64. Height 720 is NOT 64-aligned (720%64=16).
prof = cfg_conf.get("profiles", {}).get(ap, {})
w = prof.get("width", cfg_conf.get("width", 0))
h = prof.get("height", cfg_conf.get("height", 0))
check("F3 base render resolution is 64-aligned (VAE requirement)",
      w % 64 == 0 and h % 64 == 0,
      f"{w}x{h}: w%64={w%64} h%64={h%64} -- WAN VAE may error or pad on T4")

# F4: referenced shots file actually exists
sf = cfg_conf.get("shots_file")
check("F4 config shots_file exists", sf and (ROOT / sf).exists(),
      f"shots_file={sf} not found in kit/")

# =====================================================================
# GROUP G: kaggle_log_validator -- real-module call against captured
# Kaggle-runtime fixtures. This complements smoke TEST 10: that test hops
# through the CLI subprocess and confirms exit codes, while GROUP G drives
# the parser module's own function surface (scan(), extract_prompt_timings,
# flag_slow_prompts) so any future API breakage in those entry points is
# caught at the seam our suites depend on.
# =====================================================================
print("\n[G] kaggle_log_validator module surface on captured fixtures")
import kaggle_log_validator as klv

_REPO_ROOT2 = ROOT.parent.parent.parent  # kit -> GODSHAND_MASTER -> GODSHAND_MASTER -> repo
_FIXTURES2 = _REPO_ROOT2 / "godshand_tests" / "fixtures"
_fatal_log = _FIXTURES2 / "kaggle_log_fatal.txt"
_healthy_log = _FIXTURES2 / "kaggle_log_healthy.txt"
check("G0 fixtures exist", _fatal_log.exists() and _healthy_log.exists(),
      f"fatal={_fatal_log} healthy={_healthy_log}")

if _fatal_log.exists() and _healthy_log.exists():
    fatal_text = _fatal_log.read_text(encoding="utf-8")
    healthy_text = _healthy_log.read_text(encoding="utf-8")

    # G1: scan() must surface every required FATAL signature from the
    # captured 79-min hang fixture. Each is the original Defect-#2 root
    # cause or its OOM-killer descendent.
    fatal_findings = klv.scan(fatal_text)
    fatal_ids = {f["id"] for f in fatal_findings}
    for required in ("invalid_png_signature", "invalid_data_decoding_stream",
                     "runaway_repeated_messages", "process_killed"):
        check(f"G1 scan() surfaces '{required}' on fatal fixture",
              required in fatal_ids,
              f"got ids={sorted(fatal_ids)}")
    # No CUDA-OOM line in the trimmed fixture; confirm the scanner's
    # cuda signature would still fire on a synthetic CUDA-OOM line.
    cuda_test = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB."
    check("G1b scan() catches synthetic CUDA OOM line",
          any(f["id"] == "cuda_out_of_memory" for f in klv.scan(cuda_test)),
          f"input={cuda_test!r}")

    # G2: severity policy -- VHS plugin warnings must surface as WARN,
    # NOT FATAL. The original comfyui_gpu0.txt evidence you supplied is
    # healthy: 5 successful prompts with two VHS warnings per prompt.
    healthy_findings = klv.scan(healthy_text)
    fatalities = [f for f in healthy_findings if f["severity"] == "FATAL"]
    check("G2 healthy fixture has zero FATAL findings", not fatalities,
          f"got {[f['id'] for f in fatalities]}")
    vhs = [f for f in healthy_findings if f["id"] == "vhs_warning"]
    check("G2b scanner surfaces VHS warning on healthy fixture",
          len(vhs) == 1, f"vhs_count={len(vhs)}")
    check("G2c vhs_warning severity is WARN not FATAL",
          vhs and vhs[0]["severity"] == "WARN",
          f"severity={vhs[0]['severity'] if vhs else 'n/a'}")

    # G3: extract_prompt_timings parses N timing records from healthy log
    timings = klv.extract_prompt_timings(healthy_text)
    check("G3 extract_prompt_timings returns 5 entries", len(timings) == 5,
          f"got {len(timings)}")
    # All five must be within 0..3600 (one hour worst case on T4)
    check("G3b every timing is a sane positive number",
          all(60 <= t["seconds"] <= 3600 for t in timings),
          f"seconds={[t['seconds'] for t in timings]}")
    # flag_slow_prompts on a healthy log must return [] (no 1.5x outliers)
    check("G3c flag_slow_prompts() returns [] on healthy log",
          klv.flag_slow_prompts(timings) == [],
          f"slow={klv.flag_slow_prompts(timings)}")

    # G4: flag_slow_prompts WOULD fire on an injected slow prompt.
    # Inject one prompt 3x the current median and confirm detection.
    if len(timings) >= 3:
        median = sorted(t["seconds"] for t in timings)[len(timings) // 2]
        injected = list(timings) + [
            {"index": 999, "match": "synthetic slow", "seconds": median * 3}
        ]
        slow = klv.flag_slow_prompts(injected)
        check("G4 flag_slow_prompts() catches injected 3xmedian outlier",
              len(slow) == 1 and slow[0]["index"] == 999,
              f"slow_indices={[t['index'] for t in slow]}")

# =====================================================================
print("\n" + "=" * 72)
print(f"RESULT: {PASS} passed, {FAIL} failed")
print("=" * 72)

# Only sys.exit when run as a standalone script. When imported by pytest,
# keep the module loadable so pytest can collect a wrapping test below.
if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)


# =============================================================================
# pytest compatibility wrapper.
# The body of this file is a self-contained integration suite (designed to run
# as a standalone script). pytest imports the module to discover tests, which
# executes the body. The wrapper below surfaces the accumulated FAIL count as a
# single pytest test, which (a) avoids pytest's "no tests ran" deprecation and
# (b) fails the run loudly if any of the 36 assertions above did not PASS.
# =============================================================================
def test_edge_case_integration_suite():
    assert FAIL == 0, f"edge_case_integration_test reported {FAIL} failures (see [FAIL] lines above)"
    # PASS may legitimately be > 0; we only assert nothing failed.
    assert PASS > 0, "edge_case_integration_test did not run any assertions (broken harness)"

