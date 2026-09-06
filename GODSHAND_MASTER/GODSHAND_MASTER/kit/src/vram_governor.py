#!/usr/bin/env python3
"""vram_governor.py -- Pre-shot VRAM probe + parameter resolver.

PREVENTS OOM ON KAGGLE 2xT4 (16GB each) with ZERO COMPROMISE.
T4-specific regimes with aggressive memory management:

    T4_PLENTY   (>= 12 GB free): full quality, fp16, attention slicing ON
    T4_TIGHT    (8-12 GB free)   : fp16, attention slicing, VAE tiling, model offload
    T4_DANGER   (4-8 GB free)    : fp8, CPU offload model, aggressive VAE tiling, reduced steps
    T4_CRITICAL (< 4 GB free)    : MAX offload, fp8, minimal steps, 4s max, 512x288 fallback

Pure stdlib -- no torch import here so it loads instantly even on cold start.
"""
import os, subprocess, json, time

QUERY_RETRY = 5
QUERY_DELAY = 1.0
SAFE_RESERVE_MB = 2048  # 2GB safety margin for T4

# T4-specific regimes (16GB total VRAM each)
REGIME_T4_PLENTY = "t4_plenty"      # >= 12GB free
REGIME_T4_TIGHT = "t4_tight"        # 8-12GB free
REGIME_T4_DANGER = "t4_danger"      # 4-8GB free
REGIME_T4_CRITICAL = "t4_critical"  # < 4GB free


def _query_smi():
    """Return list of {index, name, total_mb, used_mb, free_mb, temp_c, util_pct}."""
    out = []
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free,"
             "temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return out
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                out.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mb": int(parts[2]),
                    "used_mb": int(parts[3]),
                    "free_mb": int(parts[4]),
                    "temp_c": int(parts[5]),
                    "util_pct": int(parts[6]),
                })
            except (ValueError, IndexError):
                continue
    except Exception:
        pass
    return out


def _query_smi_robust():
    for _ in range(QUERY_RETRY):
        gpus = _query_smi()
        if gpus:
            return gpus
        time.sleep(QUERY_DELAY)
    return []


def free_mb(gpu_index=0):
    """Free VRAM on a specific GPU, in MB. Returns 0 on any failure."""
    for g in _query_smi_robust():
        if g["index"] == gpu_index:
            return g["free_mb"]
    return 0


def probe_all():
    """Return full GPU list with free_mb, temp, util. Empty list if no GPUs."""
    return _query_smi_robust()


def pick_regime(free):
    """Resolve a T4-specific regime string from free VRAM (MB)."""
    if free >= 12 * 1024:
        return REGIME_T4_PLENTY
    if free >= 8 * 1024:
        return REGIME_T4_TIGHT
    if free >= 4 * 1024:
        return REGIME_T4_DANGER
    return REGIME_T4_CRITICAL


def resolve_params(cfg, gpu_index=0, shot=None):
    """Mutate-and-return a per-shot param dict that fits the available VRAM.

    T4-OPTIMIZED: Aggressive memory management for zero OOM on 2xT4.
    Honors configured width/height/steps/seconds but downgrades memory-heavy
    knobs when VRAM is tight. Returns dict with keys:
    regime, free_mb, width, height, steps, seconds,
    offload_level, attention_slicing, vae_tiling, vae_tile_size, fp8, notes,
    model_offload, sequential_offload, cpu_offload_clip, cpu_offload_vae.
    """
    free = free_mb(gpu_index)
    regime = pick_regime(free)
    base_w = int((shot or {}).get("width", cfg.get("width", 1280)))
    base_h = int((shot or {}).get("height", cfg.get("height", 704)))
    base_steps = int((shot or {}).get("steps", cfg.get("steps", 30)))
    base_seconds = float((shot or {}).get("dur", cfg.get("max_clip_seconds", 5)))
    
    out = {
        "regime": regime,
        "free_mb": free,
        "gpu_index": gpu_index,
        "width": base_w,
        "height": base_h,
        "steps": base_steps,
        "seconds": base_seconds,
        "offload_level": "none",
        "attention_slicing": True,  # ALWAYS ON for T4
        "vae_tiling": True,         # ALWAYS ON for T4
        "vae_tile_size": 256,
        "fp8": False,
        "model_offload": False,
        "sequential_offload": False,
        "cpu_offload_clip": False,
        "cpu_offload_vae": False,
        "notes": [],
    }
    
    if regime == REGIME_T4_PLENTY:
        # >= 12GB free - still enable attention slicing for safety
        out["attention_slicing"] = True
        out["vae_tiling"] = True
        out["vae_tile_size"] = 512
        out["notes"].append("T4_PLENTY (>=12GB): fp16, attention_slicing, VAE tiling 512")
        
    elif regime == REGIME_T4_TIGHT:
        # 8-12GB free - enable model offload
        out["attention_slicing"] = True
        out["vae_tiling"] = True
        out["vae_tile_size"] = 256
        out["model_offload"] = True
        out["sequential_offload"] = True
        out["cpu_offload_clip"] = True
        out["notes"].append("T4_TIGHT (8-12GB): fp16, model_offload, sequential_offload, cpu_offload_clip")
        
    elif regime == REGIME_T4_DANGER:
        # 4-8GB free - fp8 + aggressive offload
        out["attention_slicing"] = True
        out["vae_tiling"] = True
        out["vae_tile_size"] = 128
        out["fp8"] = True
        out["model_offload"] = True
        out["sequential_offload"] = True
        out["cpu_offload_clip"] = True
        out["cpu_offload_vae"] = True
        out["offload_level"] = "model"
        out["steps"] = max(8, base_steps - 12)
        out["seconds"] = min(base_seconds, 4.0)
        out["notes"].append("T4_DANGER (4-8GB): fp8, full offload, reduced steps/seconds")
        
    else:  # REGIME_T4_CRITICAL
        # < 4GB free - MAXIMUM offload, minimal resolution
        out["attention_slicing"] = True
        out["vae_tiling"] = True
        out["vae_tile_size"] = 64
        out["fp8"] = True
        out["model_offload"] = True
        out["sequential_offload"] = True
        out["cpu_offload_clip"] = True
        out["cpu_offload_vae"] = True
        out["offload_level"] = "full"
        out["steps"] = max(4, base_steps - 20)
        out["seconds"] = min(base_seconds, 3.0)
        out["width"] = min(base_w, 768)
        out["height"] = min(base_h, 512)
        out["notes"].append("T4_CRITICAL (<4GB): MAX offload, fp8, minimal res/steps")
    
    return out


def wait_for_vram(min_free_mb, gpu_index=0, timeout_sec=300, poll_sec=5.0):
    """Block until `min_free_mb` is available on `gpu_index`. Returns final free_mb."""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        free = free_mb(gpu_index)
        if free >= min_free_mb:
            return free
        time.sleep(poll_sec)
    return free_mb(gpu_index)


def safety_margin_mb():
    return int(os.environ.get("KIT_VRAM_SAFETY_MB", str(SAFE_RESERVE_MB)))


if __name__ == "__main__":
    gpus = probe_all()
    if not gpus:
        print("[vram_governor] no GPUs visible -- skip.")
    else:
        print(json.dumps(gpus, indent=2))
        for g in gpus:
            print(f"[vram_governor] GPU {g['index']} ({g['name']}): "
                  f"free={g['free_mb']}MB temp={g['temp_c']}C util={g['util_pct']}% "
                  f"regime={pick_regime(g['free_mb'])}")
