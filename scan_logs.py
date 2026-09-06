#!/usr/bin/env python3
"""
Kaggle Log Scanner — drop-in operator tool.

Usage (after any Kaggle ComfyUI session):
    python scan_logs.py [/path/to/comfyui.log]

If no path given, checks the default locations:
    /kaggle/working/comfyui.log
    /kaggle/working/comfyui_gpu0.log
    /kaggle/working/download.txt

Exit codes:
    0 = clean (no FATAL signatures)
    1 = FATAL signature(s) found
    2 = log file not found
"""

import sys
import os

# Ensure the kit's parser is on sys.path regardless of CWD
KIT_SRC = os.getenv("GODSHAND_KIT_SRC", "/kaggle/working/godshand/kit/src")
if KIT_SRC not in sys.path:
    sys.path.insert(0, KIT_SRC)

import kaggle_log_validator as klv


DEFAULT_LOGS = [
    "/kaggle/working/comfyui.log",
    "/kaggle/working/comfyui_gpu0.log",
    "/kaggle/working/download.txt",
]


def find_log(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for p in DEFAULT_LOGS:
        if os.path.exists(p):
            return p
    return None


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    log_path = find_log(argv[0] if argv else None)
    if not log_path:
        print("[scan] No log file found. Checked:")
        for p in DEFAULT_LOGS:
            print(f"  {p} — {'exists' if os.path.exists(p) else 'missing'}")
        if argv:
            print(f"  (explicit) {argv[0]} — missing")
        return 2

    print(f"[scan] Scanning: {log_path}")
    findings = klv.scan_path(log_path)

    any_fatal = False
    for f in findings:
        sev = "FATAL" if f["severity"] == "FATAL" else "WARN "
        mult = f"({f['count']}x)" if f.get("count", 1) > 1 else ""
        print(f"  [{sev}] L{f['line']:>5} {mult:<7} {f['id']}")
        if f["severity"] == "FATAL":
            any_fatal = True

    if not findings:
        print("  (no signatures detected)")

    if any_fatal:
        print("\n[scan] FAIL — FATAL signatures detected in log.")
        return 1
    else:
        print("\n[scan] OK — no FATAL signatures.")
        return 0


if __name__ == "__main__":
    sys.exit(main())