#!/usr/bin/env python3
# l4_model_setup.py -- faithful port of the PROVEN WAN 2.2 downloader from
# wan-latest.ipynb (the exact repos/quant-order/filenames that produced the
# successful 28-shot render). Downloads into $COMFYUI/models and writes
# wan_env.sh (WAN_HIGH/WAN_LOW/WAN_UMT5/WAN_VAE + step/cfg) for the renderer.
#
# ZERO-OOM T4 MODE: Prefers Q4_K_M quantization for 2xT4 (16GB each) to prevent OOM.
# Set KIT_QUANT=Q8_0 to override for higher quality on larger GPUs.
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"   # plain download; avoids Xet scratch space

COMFY = Path(os.environ.get("COMFYUI") or (Path.cwd() / "_run" / "ComfyUI"))
MODELS = COMFY / "models"
for sub in ("unet", "loras", "text_encoders", "vae"):
    (MODELS / sub).mkdir(parents=True, exist_ok=True)
_tok = os.environ.get("HF_TOKEN") or None

# Quantization preference: Q4_K_M for T4 zero-OOM, Q8_0 for quality on larger GPUs
KIT_QUANT = os.environ.get("KIT_QUANT", "Q4_K_M").lower()
# Quant order preference - Q4_K_M first for T4 zero-OOM
_QORDER = ["q4_k_m", "q4_k_s", "q5_k_m", "q3_k_m", "q4_0", "q8_0"]

# If user explicitly wants Q8, reorder
if "q8" in KIT_QUANT:
    _QORDER = ["q8_0", "q4_k_m", "q4_k_s", "q5_k_m", "q3_k_m", "q4_0"]

WAN_GGUF_REPO = "QuantStack/Wan2.2-I2V-A14B-GGUF"
WAN_COMFY_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

def _pick_gguf(files, tag):
    cand = [f for f in files if f.lower().endswith(".gguf") and tag in f.lower()]
    for q in _QORDER:
        for f in cand:
            if q in f.lower():
                return f
    return cand[0] if cand else None

def _dl(repo, fn, sub):
    lp = hf_hub_download(repo, filename=fn, local_dir=str(MODELS / sub), token=_tok)
    flat = MODELS / sub / Path(fn).name
    if not flat.exists():
        shutil.move(lp, flat)
    return flat.name

print("[wan] listing GGUF experts from", WAN_GGUF_REPO)
_g = list_repo_files(WAN_GGUF_REPO, token=_tok)
_hi = _pick_gguf(_g, "high"); _lo = _pick_gguf(_g, "low")
assert _hi and _lo, "no high/low GGUF experts in %s" % WAN_GGUF_REPO
WAN_HIGH = _dl(WAN_GGUF_REPO, _hi, "unet")
WAN_LOW = _dl(WAN_GGUF_REPO, _lo, "unet")

_c = list_repo_files(WAN_COMFY_REPO, token=_tok)
def _find(files, *needles):
    for f in files:
        fl = f.lower()
        if all(n in fl for n in needles):
            return f
    return None
_umt5 = _find(_c, "umt5", "fp8") or _find(_c, "umt5")
_vae = _find(_c, "wan_2.1_vae") or _find(_c, "vae", "2.1")
assert _umt5 and _vae, "umt5/vae not found in %s" % WAN_COMFY_REPO
WAN_UMT5 = _dl(WAN_COMFY_REPO, _umt5, "text_encoders")
WAN_VAE = _dl(WAN_COMFY_REPO, _vae, "vae")

LH = LL = ""
_lhi = _find(_c, "i2v", "lightx2v", "high") or _find(_c, "i2v", "4step", "high") or _find(_c, "i2v", "4steps", "high")
_llo = _find(_c, "i2v", "lightx2v", "low") or _find(_c, "i2v", "4step", "low") or _find(_c, "i2v", "4steps", "low")
if _lhi and _llo:
    try:
        LH = _dl(WAN_COMFY_REPO, _lhi, "loras")
        LL = _dl(WAN_COMFY_REPO, _llo, "loras")
    except Exception as e:
        LH = LL = ""
        print("[wan] 4-step LoRA download failed (%s); will run at higher steps." % e)

# Steps/CFG based on whether 4-step LoRAs are available
steps, cfg = ("4", "1.0") if (LH and LL) else ("20", "3.5")
# Override for zero-OOM T4 profile
if KIT_QUANT == "q4_k_m":
    steps, cfg = ("20", "5.5")  # More steps, lower CFG for quality at lower quant

env = (
    "export KIT_ENGINE=wan\n"
    'export WAN_HIGH="%s"\n'
    'export WAN_LOW="%s"\n'
    'export WAN_UMT5="%s"\n'
    'export WAN_VAE="%s"\n'
    'export WAN_LORA_HIGH="%s"\n'
    'export WAN_LORA_LOW="%s"\n'
    'export KIT_WAN_STEPS="%s"\n'
    'export KIT_WAN_CFG="%s"\n'
    'export KIT_QUANT="%s"\n'
) % (WAN_HIGH, WAN_LOW, WAN_UMT5, WAN_VAE, LH, LL, steps, cfg, KIT_QUANT.upper())
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan_env.sh"), "w").write(env)

for cdir in MODELS.rglob(".cache"):
    shutil.rmtree(cdir, ignore_errors=True)
print("[wan] READY | high=%s low=%s umt5=%s vae=%s loraH=%s loraL=%s steps=%s cfg=%s quant=%s"
      % (WAN_HIGH, WAN_LOW, WAN_UMT5, WAN_VAE, LH or "-", LL or "-", steps, cfg, KIT_QUANT.upper()))
