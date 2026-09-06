#!/usr/bin/env python3
"""
Kaggle GPU Smoke Test — run AFTER model downloads, BEFORE full notebook run.

This executes `selftest --motion --fast` on the actual Kaggle GPU to prove
the production render path (`render_motion.py` → WanImageToVideo → SaveImage)
works on real hardware with real model weights loaded.

Usage:
    python gpu_smoke.py

Exit codes:
    0 = green (production path renders on real GPU)
    1 = red (regression in production path)
    2 = environment not ready (no GPU, kit not found, etc.)
"""

import os
import sys
import subprocess

KIT_ROOT = os.getenv("GODSHAND_KIT_ROOT", "/kaggle/working/godshand/kit")
SRC = os.path.join(KIT_ROOT, "src")
PY = sys.executable


def check_env():
    # GPU present?
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if out.returncode != 0 or "T4" not in out.stdout:
            print("[gpu_smoke] No T4 GPU visible via nvidia-smi")
            return False
    except FileNotFoundError:
        print("[gpu_smoke] nvidia-smi not found")
        return False

    # Kit present?
    if not os.path.isdir(SRC):
        print(f"[gpu_smoke] Kit source not found at {SRC}")
        return False

    # Model weights present? (quick check)
    models = os.path.join(KIT_ROOT, "..", "models")
    if not os.path.exists(models):
        print(f"[gpu_smoke] Models dir not found at {models}")
        return False

    return True


def run_selftest_motion():
    from render_motion import tune_cuda, ensure_vram
    tune_cuda(granularity=128)
    if not ensure_vram(500):
        print("[gpu_smoke] WARNING: VRAM headroom insufficient before smoke test")
    env = os.environ.copy()
    env.update({
        "COMFY_MOCK": "0",
        "POST_MODE": "gpu",
        "KIT_CONFIG": "config_confrontation.yaml",
        "KIT_FAST_POST": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
    })
    cmd = [PY, os.path.join(SRC, "selftest.py"), "--motion", "--fast"]
    print(f"[gpu_smoke] Running: {' '.join(cmd)}")
    print(f"[gpu_smoke] CWD: {KIT_ROOT}")
    print(f"[gpu_smoke] COMFY_MOCK=0, POST_MODE=gpu, tune_cuda applied")

    proc = subprocess.run(cmd, cwd=KIT_ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)

    print(proc.stdout[-3000:] if proc.stdout else "(no stdout)")
    if proc.stderr:
        print(proc.stderr[-2000:], file=sys.stderr)

    return proc.returncode == 0, proc.stdout


def main():
    print("=" * 60)
    print("GODSHAND GPU SMOKE TEST")
    print("=" * 60)

    if not check_env():
        print("\n[gpu_smoke] Environment check FAILED (rc=2)")
        return 2

    print("\n[gpu_smoke] Environment OK. Running selftest --motion --fast on real GPU...")
    ok, out = run_selftest_motion()

    if ok and "ALL GREEN" in out:
        print("\n[gpu_smoke] ✅ GREEN — production path renders on real GPU")
        return 0
    else:
        print("\n[gpu_smoke] ❌ RED — production path regression detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())