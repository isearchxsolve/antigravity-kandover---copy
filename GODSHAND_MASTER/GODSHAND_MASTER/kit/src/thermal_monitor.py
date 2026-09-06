#!/usr/bin/env python3
"""thermal_monitor.py -- Post-run telemetry dump + on-the-fly throttle.

Records a per-shot thermal profile (time, GPU idx, temp, util, free VRAM) so
post-run forensics can show whether throttling happened. If `block_if_hot` is
called, it sleeps until the named GPU drops below throttle temp -- useful
between heavy back-to-back renders on a 2xT4 chassis.
"""
import os, time, json
import vram_governor as vg

LOG_PATH = os.environ.get("KIT_THERMAL_LOG",
                         os.path.join("/tmp" if os.name != "nt" else os.environ.get("TEMP", "."),
                                      "thermal_profile.jsonl"))

_entries = []


def snapshot(gpu_index=0, tag=""):
    """Record one row in the in-memory + on-disk telemetry log."""
    gpus = vg.probe_all()
    cur = next((g for g in gpus if g["index"] == gpu_index), None)
    if not cur:
        return None
    row = {
        "ts": time.time(),
        "tag": tag,
        "gpu": gpu_index,
        "temp_c": cur["temp_c"],
        "util_pct": cur["util_pct"],
        "free_mb": cur["free_mb"],
    }
    _entries.append(row)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass
    return row


def block_if_hot(gpu_index=0, max_temp_c=83, timeout_sec=180, poll_sec=3):
    t0 = time.time()
    while True:
        gpus = vg.probe_all()
        cur = next((g for g in gpus if g["index"] == gpu_index), None)
        if cur is None or cur["temp_c"] <= max_temp_c:
            return cur
        if time.time() - t0 > timeout_sec:
            print(f"[thermal_monitor] GPU {gpu_index} still {cur['temp_c']}C "
                  f"after {timeout_sec}s -- proceeding.")
            return cur
        time.sleep(poll_sec)


def dump():
    """Return the in-memory profile as JSON; flushed on every snapshot()."""
    return list(_entries)


if __name__ == "__main__":
    print(json.dumps(snapshot(0, "manual"), indent=2))
