import json

# Read the full merged notebook code
with open(r'C:/Users/Admin/Downloads/rainman-wan-MERGED.ipynb', 'r', encoding='utf-8') as f:
    merged = json.load(f)

merged_code = merged['cells'][1]['source']

# Find the start of STEP 1
step1_idx = merged_code.find("# ==============================================================================\n# STEP 1: PROGRESSIVE CHUNKING CONTROLS & REAL GPU ENVIRONMENT")
rest_of_code = merged_code[step1_idx:]

# Build the corrected code cell
corrected_code = [
    "# ==============================================================================\n",
    "# DISK SPACE MANAGEMENT & CLEANUP UTILITIES (TUNED: PROACTIVE DISK GUARD)\n",
    "# ==============================================================================\n",
    "import os, sys, subprocess, json, shutil, time, gc\n",
    "from pathlib import Path\n",
    "\n",
    "def check_disk_space(path='/kaggle/working', min_gb=5):\n",
    '    """Check available disk space, return (free_gb, total_gb). Warn if below threshold."""\n',
    "    usage = shutil.disk_usage(path)\n",
    "    free_gb = usage.free / (1024**3)\n",
    "    total_gb = usage.total / (1024**3)\n",
    "    if free_gb < min_gb:\n",
    '        print(f"\u26a0\ufe0f  LOW DISK: {free_gb:.1f}GB free of {total_gb:.1f}GB at {path} (min {min_gb}GB)")\n',
    "    else:\n",
    '        print(f"\u2713 Disk OK: {free_gb:.1f}GB free of {total_gb:.1f}GB at {path}")\n',
    "    return free_gb, total_gb\n",
    "\n",
    "def cleanup_kaggle_working(aggressive=False):\n",
    '    """Proactive disk cleanup for Kaggle /kaggle/working (~20GB cap)."""\n',
    "    freed = 0\n",
    "    targets = [\n",
    "        # HF Hub caches\n",
    "        '/kaggle/working/.cache/huggingface',\n",
    "        '/kaggle/working/models/.cache',\n",
    "        '/kaggle/working/ComfyUI/models/.cache',\n",
    "        # ComfyUI output caches (per-GPU)\n",
    "        '/kaggle/working/ComfyUI/output_gpu0',\n",
    "        '/kaggle/working/ComfyUI/output_gpu1',\n",
    "        # split_files from model downloads\n",
    "        '/kaggle/working/ComfyUI/models/split_files',\n",
    "        # pip cache\n",
    "        '/root/.cache/pip',\n",
    "        # temp files\n",
    "        '/tmp/*.tmp',\n",
    "        '/tmp/hf_*',\n",
    "    ]\n",
    "    if aggressive:\n",
    "        targets += [\n",
    "            '/kaggle/working/confrontation_work',  # ONLY if force rerender\n",
    "        ]\n",
    "    for target in targets:\n",
    "        try:\n",
    "            if '*' in target:\n",
    "                import glob\n",
    "                for f in glob.glob(target):\n",
    "                    if os.path.isfile(f):\n",
    "                        sz = os.path.getsize(f)\n",
    "                        os.remove(f)\n",
    "                        freed += sz\n",
    "            elif os.path.isdir(target):\n",
    "                # Calculate size before removal\n",
    "                for root, dirs, files in os.walk(target):\n",
    "                    for f in files:\n",
    "                        fp = os.path.join(root, f)\n",
    "                        try:\n",
    "                            freed += os.path.getsize(fp)\n",
    "                        except OSError:\n",
    "                            pass\n",
    "                shutil.rmtree(target, ignore_errors=True)\n",
    "            elif os.path.isfile(target):\n",
    "                freed += os.path.getsize(target)\n",
    "                os.remove(target)\n",
    "        except Exception:\n",
    "            pass  # ignore errors\n",
    "    if freed > 0:\n",
    '        print(f"\U0001f9f9 Cleaned {freed / (1024**3):.2f} GB")\n',
    "    return freed\n",
    "\n",
    "def gpu_memory_cleanup():\n",
    '    """Clear GPU memory (VRAM + system RAM)."""\n',
    "    gc.collect()\n",
    "    try:\n",
    "        import torch\n",
    "        if torch.cuda.is_available():\n",
    "            torch.cuda.empty_cache()\n",
    "            torch.cuda.ipc_collect()\n",
    "    except Exception:\n",
    "        pass\n",
    "\n",
    "# Run initial cleanup and disk check\n",
    'print("=== PROACTIVE DISK GUARD ===")\n',
    "check_disk_space()\n",
    "cleanup_kaggle_working()\n",
    "gpu_memory_cleanup()\n",
    "check_disk_space()\n",
    "\n",
    "# ==============================================================================\n",
    "# HOLLYWOOD QUALITY OVERRIDES (TUNED FINAL ROUND)\n",
    "# ==============================================================================\n",
    "os.environ['GODSHAND_MODE'] = 'full'\n",
    "os.environ['GODSHAND_FORGE'] = 'anchors'\n",
    "os.environ['KIT_WAN_STEPS'] = '40'\n",
    "os.environ['KIT_WAN_CFG'] = '6.0'\n",
    "os.environ['WAN_LORA_HIGH'] = ''\n",
    "os.environ['WAN_LORA_LOW'] = ''\n",
    "os.environ['KIT_FAST_POST'] = '0'  # RESTORED: color grade + face-SR now run\n",
    "\n",
    "import urllib.request, urllib.parse, urllib.error\n",
    "import torch, re\n",
    "import numpy as np\n",
    "from PIL import Image\n",
]

# Append the rest of the merged code
corrected_code.extend(rest_of_code.splitlines(keepends=True))

# Create the new notebook
notebook = {
    "metadata": merged["metadata"],
    "nbformat_minor": merged["nbformat_minor"],
    "nbformat": merged["nbformat"],
    "cells": [
        merged["cells"][0],  # Keep the markdown cell
        {
            "cell_type": "code",
            "source": corrected_code,
            "metadata": {}
        }
    ]
}

# Write the corrected notebook
output_path = r'C:/Users/Admin/Downloads/antigravity-kandover/rainman-wan-MERGED-TUNED.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Written to {output_path}")
print(f"Total cells: {len(notebook['cells'])}")
print(f"Code length: {len(corrected_code)} lines")