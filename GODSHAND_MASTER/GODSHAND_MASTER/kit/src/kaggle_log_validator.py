#!/usr/bin/env python3
"""kaggle_log_validator.py -- assert on Kaggle / ComfyUI runtime logs.

The cleanest summary of why this file exists:

  The GODSHAND audit threads (selftest, smoke, edge_case) all consume
  CODE -- they probe imports, AST, mock pipelines, dry-runs. None of them
  parse Kaggle's actual stdout/stderr (comfyui.log, download.txt,
  comfyui_gpu0.txt) before declaring the engine green. Two real Kaggle
  logs proved that gap:

    download.txt   -- 1000+ `Invalid PNG signature 0x2066747970` followed
                       by `Killed` (the 79-min hang root cause).

    comfyui_gpu0.txt -- 5 successful prompts but 2 `[VideoHelperSuite] -
                        WARNING - Missing input for ...` on every prompt.

  Both would have slipped through every existing suite. They are also both
  richly parseable: each carriesМы clear signatures -- `Invalid PNG signature`,
  `Killed`, `CUDA out of memory`, `[VideoHelperSuite] - WARNING`, and
  `Prompt executed in HH:MM:SS`. This module turns those signatures into
  structured findings.

CLI:
    python kaggle_log_validator.py --scan PATH [--scan PATH2 ...]
        Exit 0 = no fatal findings (warnings may still be reported).
        Exit 1 = at least one fatal finding (Killed / OOM / Invalid PNG / etc).
    python kaggle_log_validator.py --extract-prompts PATH
        Print JSON list of {index, prompt, seconds} for `Prompt executed in ...`.

Findings are returned by `scan(...)` as a list of dicts, so all three suites
(selftest --logs, comprehensive_smoke_test, edge_case_integration_test) can
introspect what was (and was not) detected in a captured log without needing
to parse the text themselves.

Severity policy (deliberate):
    FATAL  -- Killed, CUDA out of memory, Invalid PNG signature, Invalid data
              found when processing input, repeated `Last message repeated`
              seen ≥ 50 times (the original hang fingerprint).
    WARN   -- [VideoHelperSuite] - WARNING. By policy these are warnings,
              not failures: VHS ships the same warning from its INPUT_TYPES
              validator on prompts that contain zero VHS nodes, because VHS
              is installed. A clean WAN-only run with VHS installed still
              produces them. Failing on them would block every healthy run
              that keeps VHS around for unrelated reasons. The notebook-level
              "no VHS_VideoCombine node in the inline graph" check is owned
              by the static suite (Notebook integrity test), NOT here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Optional, Iterable

# =============================================================================
# SIGNATURE PATTERNS
# =============================================================================
# - All FATAL signatures match patterns observed in the bug-side real log
#   (download.txt). - All WARN signatures observable in the healthy real log
#   (comfyui_gpu0.txt).
# =============================================================================

_FATAL_PATTERNS: List[Dict] = [
    {
        "id": "invalid_png_signature",
        "regex": re.compile(r"Invalid PNG signature\b"),
        "severity": "FATAL",
        "summary": "ffmpeg was asked to decode PNG frames but received a "
                   "different container (typically MP4 -- '0x2066747970' is "
                   "the ASCII ' ftyp' MP4 file signature). The notebook graph "
                   "is emitting the wrong frame format for the postprocess stage.",
    },
    {
        "id": "invalid_data_decoding_stream",
        "regex": re.compile(
            r"Error while decoding stream #\d+:\d+: Invalid data found"
        ),
        "severity": "FATAL",
        "summary": "ffmpeg demuxer could not decode the input stream -- "
                   "usually paired with Invalid PNG signature when the input "
                   "container doesn't match the demuxer expectation.",
    },
    {
        "id": "process_killed",
        # Real Kaggle logs prefix every line with a wall-clock stamp, e.g.
        #   `4759.0s 1889 Killed`      <- OOM-killer on bash
        #   `1889 Killed`              <- plain bash
        #   `[12345] Killed`           <- bracketed PID format
        # Accept a leading `[<digits>]` OR `<digits>` OR any leading prefix
        # text followed by a trailing `\d+\s+Killed` or a bare `Killed` line.
        # Equivalent in plain English: any line that ends with "Killed" and
        # either starts with optional noise + a PID, or is exactly "Killed".
        "regex": re.compile(
            r"^\s*(?:\[\d+\]|\d+(?:\.\d+)?s?)\s+\d*\s*Killed\b"
            r"|^\s*\d+\s+Killed\b"
            r"|^\s*Killed\s*$",
            re.MULTILINE,
        ),
        "severity": "FATAL",
        "summary": "Linux OOM-killer (or manual SIGKILL) terminated the run. "
                   "On a 2xT4 Kaggle session this is the death certificate of "
                   "the 79-min hang: VRAM/swap exhausted, kernel reclaimed.",
    },
    {
        "id": "cuda_out_of_memory",
        "regex": re.compile(
            r"CUDA out of memory|OutOfMemoryError|out of memory.*PYTORCH",
            re.IGNORECASE,
        ),
        "severity": "FATAL",
        "summary": "Explicit CUDA OOM from PyTorch. Even if our internal OOM "
                   "fallback ladder catches and retries, a leaked OOM in the "
                   "log means the fallback was either not wired or exhausted.",
    },
    {
        "id": "comfyui_runtime_error",
        "regex": re.compile(
            r"ComfyUI.* (?:RuntimeError|ERROR) DETECTED|prompt execution "
            r"failed|Traceback \(most recent call last\)"
        ),
        "severity": "FATAL",
        "summary": "ComfyUI itself or the prompt handler reported a non-OOM "
                   "runtime error / Python traceback.",
    },
]

_WARN_PATTERNS: List[Dict] = [
    {
        "id": "vhs_warning",
        "regex": re.compile(
            r"\[VideoHelperSuite\][^\n]*WARNING[^\n]*Missing input for "
            r"(save_metadata|trim_to_audio)"
        ),
        "severity": "WARN",
        "summary": "VideoHelperSuite log line. By policy WARN (not FATAL) -- "
                   "VHS emits this from its INPUT_TYPES validator on every "
                   "prompt that doesn't explicitly set save_metadata / "
                   "trim_to_audio, even when no VHS node is in the graph. "
                   "The graph-level check (no VHS_VideoCombine in the inline "
                   "graph) is owned by the notebook integrity test.",
    },
]

_REPEATED_MESSAGE = re.compile(r"Last message repeated (\d+) times")
# Threshold: original download.txt showed 1000+ repeats before Killed.
# Anything ≥ 50 is suspicious -- a single benign re-encode might repeat a
# handful of times; 50+ implies runaway log spam from a stuck decoder.
_REPEATED_THRESHOLD = 50

_PROMPT_EXECUTED = re.compile(
    r"Prompt executed in (\d+):(\d{2}):(\d{2})"
)


# =============================================================================
# SCANNER
# =============================================================================
def scan(text: str) -> List[Dict]:
    """Return a de-duplicated list of findings for `text`.

    Order: FATAL findings first (in the order they first appear in text),
    then WARN findings. Each finding has the line where the FIRST match
    occurs; repeated occurrences are counted but produce a single row.
    """
    findings: List[Dict] = []
    seen_ids: set = set()

    # --- FATAL pattern scan (one row per pattern, line of first match) ---
    for pat in _FATAL_PATTERNS:
        m = pat["regex"].search(text)
        if m:
            line_no = _line_of(text, m.start())
            count = len(pat["regex"].findall(text))
            findings.append({
                "id": pat["id"],
                "severity": pat["severity"],
                "line": line_no,
                "count": count,
                "summary": pat["summary"],
            })
            seen_ids.add(pat["id"])

    # --- runaway `Last message repeated N times` -- counts total repeats ---
    total_repeats = sum(int(m) for m in _REPEATED_MESSAGE.findall(text))
    if total_repeats >= _REPEATED_THRESHOLD:
        findings.append({
            "id": "runaway_repeated_messages",
            "severity": "FATAL",
            "line": _line_of(text, _REPEATED_MESSAGE.search(text).start()),
            "count": total_repeats,
            "summary": f"ffmpeg/decoder log shows {total_repeats}x "
                       f"`Last message repeated` (threshold "
                       f"{_REPEATED_THRESHOLD}). This is the original 79-min "
                       f"hang fingerprint -- the decoder is stuck emitting "
                       f"the same error forever before being Killed.",
        })

    # --- WARN pattern scan (VHS etc.) ---
    for pat in _WARN_PATTERNS:
        matches = list(pat["regex"].finditer(text))
        if matches:
            findings.append({
                "id": pat["id"],
                "severity": pat["severity"],
                "line": _line_of(text, matches[0].start()),
                "count": len(matches),
                "summary": pat["summary"],
            })

    return findings


def extract_prompt_timings(text: str) -> List[Dict]:
    """Parse all `Prompt executed in HH:MM:SS` lines, return seconds-per-prompt."""
    timings = []
    for idx, m in enumerate(_PROMPT_EXECUTED.finditer(text)):
        h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        timings.append({
            "index": idx,
            "match": m.group(0),
            "seconds": h * 3600 + mn * 60 + s,
        })
    return timings


def flag_slow_prompts(timings: List[Dict], factor: float = 1.5) -> List[Dict]:
    """Return any prompt whose wall-clock is > factor * median."""
    if len(timings) < 3:
        return []  # too few to talk about a baseline
    secs = sorted(t["seconds"] for t in timings)
    median = secs[len(secs) // 2]
    if median == 0:
        return []
    return [t for t in timings if t["seconds"] > factor * median]


# =============================================================================
# HELPERS
# =============================================================================
def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _format(findings: List[Dict]) -> str:
    if not findings:
        return "[kaggle_log] no findings -- log looks clean."
    rows = []
    for f in findings:
        marker = "FATAL" if f["severity"] == "FATAL" else "WARN "
        mult = f"({f['count']}x)" if f.get("count", 1) > 1 else ""
        rows.append(
            f"  [{marker}] L{f['line']:>5} {mult:<7} {f['id']}\n"
            f"            {f['summary']}"
        )
    return "\n".join(rows)


def scan_path(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return [{
            "id": "log_file_missing",
            "severity": "FATAL",
            "line": 0,
            "count": 1,
            "summary": f"Log file not found: {path}",
        }]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return scan(f.read())


# =============================================================================
# CLI
# =============================================================================
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scan Kaggle / ComfyUI runtime logs for defect "
                    "signatures (Invalid PNG / Killed / OOM / VHS warnings / "
                    "per-prompt timing regressions)."
    )
    ap.add_argument("--scan", action="append", default=[],
                    metavar="PATH",
                    help="Log file to scan. Repeatable.")
    ap.add_argument("--extract-prompts", metavar="PATH",
                    help="Print JSON list of {index, prompt, seconds} for "
                         "each `Prompt executed in HH:MM:SS`.")
    args = ap.parse_args(argv)

    rc = 0

    if args.extract_prompts:
        if not os.path.exists(args.extract_prompts):
            print(json.dumps({"error": "log file not found",
                              "path": args.extract_prompts}))
            return 1
        with open(args.extract_prompts, encoding="utf-8", errors="ignore") as f:
            timings = extract_prompt_timings(f.read())
        slow = flag_slow_prompts(timings)
        report = {
            "prompts": timings,
            "median_seconds": (sorted(t["seconds"] for t in timings)
                               [len(timings) // 2] if timings else None),
            "slow_prompts": slow,
        }
        print(json.dumps(report, indent=2))
        # Do not return rc=1 on slow prompts; that's informational.
        # rc is determined by --scan below if both flags pass.

    if args.scan:
        all_findings = []
        for p in args.scan:
            fnd = scan_path(p)
            all_findings.extend(fnd)
            print(f"\n=== {p} ===")
            print(_format(fnd))
        any_fatal = any(f["severity"] == "FATAL" for f in all_findings)
        if any_fatal:
            print("\n[kaggle_log] FAIL -- at least one FATAL finding.")
            rc = 1
        elif all_findings:
            print("\n[kaggle_log] OK with WARNINGS -- no FATAL findings; "
                  "review WARN rows above.")
        else:
            print("\n[kaggle_log] OK -- all scanned logs clean.")
        return rc

    # No --scan and no --extract-prompts: print usage with non-zero exit so
    # the test harness never confuses "no args given" with "all green".
    if not args.scan and not args.extract_prompts:
        ap.print_help()
        return 2

    return rc


if __name__ == "__main__":
    sys.exit(main())
