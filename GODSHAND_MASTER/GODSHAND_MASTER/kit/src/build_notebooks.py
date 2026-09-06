#!/usr/bin/env python3
"""Generate the three Kaggle notebooks as valid .ipynb JSON from cell sources here.
Keeps the kit self-contained (no hand-authored JSON to corrupt).
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def nb(cells):
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": c if isinstance(c, list) else [c]} for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"},
                     "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }

PREFLIGHT = nb([
    "# 00 - PREFLIGHT: environment + input gate\n",
    "!nvidia-smi || echo 'No GPU: set Accelerator = GPU T4 x2'\n",
    "import os; os.chdir('/kaggle/working/film_kit_v3') if os.path.isdir('/kaggle/working/film_kit_v3') else None\n",
    "!python3 src/preflight.py\n",
])

BUILD = nb([
    "# BUILD MODEL DATASET (run once) - downloads + checksums weights, saves reusable Dataset\n",
    "%env HF_HUB_ENABLE_HF_TRANSFER=1\n",
    "!pip -q install huggingface_hub hf_transfer >/dev/null\n",
    "# Pull every repo in MANIFEST with resume + retry, into /kaggle/working/models\n"
    "import re, subprocess, os\n"
    "os.makedirs('/kaggle/working/models', exist_ok=True)\n"
    "for line in open('MANIFEST'):\n"
    "    line=line.strip()\n"
    "    if not line or line.startswith('#'): continue\n"
    "    parts=[p.strip() for p in line.split('|')]\n"
    "    if len(parts)<2: continue\n"
    "    repo=parts[1]\n"
    "    if repo.startswith('http'): continue\n"
    "    for attempt in range(3):\n"
    "        r=subprocess.run(['hf','download',repo,'--local-dir',f'/kaggle/working/models/{repo.split(chr(47))[-1]}'])\n"
    "        if r.returncode==0: break\n",
    "print('Now: Save Version -> as a Dataset, then mount it read-only at /kaggle/input/moment-models')\n",
])

RUN = nb([
    "# KAGGLE RUN - hardened, resumable, with a COST-METER GATE\n",
    "import os\n"
    "os.chdir('/kaggle/working/film_kit_v3')\n"
    "os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'\n"
    "os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'\n",
    "# Card profile: 'kaggle_2xt4' (FREE default) or 'lightning_l4'. On Kaggle leave as is.\n"
    "import yaml\n"
    "prof=os.environ.get('MOMENT_PROFILE','kaggle_2xt4')\n"
    "c=yaml.safe_load(open('config.yaml')); c['active_profile']=prof\n"
    "yaml.safe_dump(c, open('config.yaml','w'), sort_keys=False); print('active_profile ->', prof)\n",
    "# STEP 1 (safe): preflight -> cull -> de-age -> render ONE sample shot -> print projected time/cost, then HOLD.\n"
    "os.environ.setdefault('BUDGET_USD','4')  # optional spend ceiling for the meter warning\n"
    "!python3 src/filmforge.py\n",
    "# STEP 2: after reading the COST METER numbers above and you are happy, uncomment to render the whole film:\n"
    "# os.environ['CONFIRM_FULL_RUN']='1'\n"
    "# !python3 src/filmforge.py\n",
    "# STEP 3 (after STEP 2 finishes): package for download\n"
    "# !cd output && zip -qr /kaggle/working/THE_MOMENT_final.zip THE_MOMENT.mp4 && echo 'Download from Output tab.'\n",
])

OBSERVER = nb([
    "# THE OBSERVER (Film II) - hardened, resumable, COST-METER GATE. Generates its own character (no photos).\n",
    "import os\n"
    "os.chdir('/kaggle/working/film_kit_v3')\n"
    "os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'\n"
    "os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'\n"
    "os.environ['KIT_CONFIG']='config_observer.yaml'  # <- selects Film II\n",
    "# Card profile: 'kaggle_2xt4' (FREE default) or 'lightning_l4'.\n"
    "import yaml\n"
    "prof=os.environ.get('MOMENT_PROFILE','kaggle_2xt4')\n"
    "c=yaml.safe_load(open('config_observer.yaml')); c['active_profile']=prof\n"
    "yaml.safe_dump(c, open('config_observer.yaml','w'), sort_keys=False); print('active_profile ->', prof)\n",
    "# STEP 1 (safe): preflight -> FORGE the Observer -> render ONE sample shot -> COST METER, then HOLD.\n"
    "os.environ.setdefault('BUDGET_USD','4')\n"
    "!python3 src/filmforge_observer.py\n",
    "# STEP 2: happy with the cost-meter numbers? uncomment both lines to render the whole film:\n"
    "# os.environ['CONFIRM_FULL_RUN']='1'\n"
    "# !python3 src/filmforge_observer.py\n",
    "# STEP 3 (after STEP 2 finishes): package for download\n"
    "# !cd output_observer && zip -qr /kaggle/working/THE_OBSERVER_final.zip THE_OBSERVER.mp4 && echo 'Download from Output tab.'\n",
])

ONE = nb([
    "# THE ONE (the single merged film) - real memoir crosses into the metabolized creature. Needs your photos + forges the creature.\n",
    "import os\n"
    "os.chdir('/kaggle/working/film_kit_v3')\n"
    "os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'\n"
    "os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'\n"
    "os.environ['KIT_CONFIG']='config_one.yaml'  # <- selects THE ONE\n",
    "import yaml\n"
    "prof=os.environ.get('MOMENT_PROFILE','kaggle_2xt4')\n"
    "c=yaml.safe_load(open('config_one.yaml')); c['active_profile']=prof\n"
    "yaml.safe_dump(c, open('config_one.yaml','w'), sort_keys=False); print('active_profile ->', prof)\n",
    "# STEP 1 (safe): preflight -> forge creature -> cull/de-age your photos -> ONE sample shot -> COST METER, then HOLD.\n"
    "os.environ.setdefault('BUDGET_USD','4')\n"
    "!python3 src/filmforge_one.py\n",
    "# STEP 2: happy with the numbers? uncomment both lines to render the whole film:\n"
    "# os.environ['CONFIRM_FULL_RUN']='1'\n"
    "# !python3 src/filmforge_one.py\n",
    "# STEP 3 (after STEP 2 finishes): package for download\n"
    "# !cd output_one && zip -qr /kaggle/working/THE_ONE_final.zip THE_ONE.mp4 && echo 'Download from Output tab.'\n",
])

def main():
    json.dump(PREFLIGHT, open(os.path.join(ROOT, "00_preflight.ipynb"), "w"), indent=1)
    json.dump(BUILD, open(os.path.join(ROOT, "build_model_dataset.ipynb"), "w"), indent=1)
    json.dump(RUN, open(os.path.join(ROOT, "kaggle_run.ipynb"), "w"), indent=1)
    json.dump(OBSERVER, open(os.path.join(ROOT, "kaggle_run_observer.ipynb"), "w"), indent=1)
    json.dump(ONE, open(os.path.join(ROOT, "kaggle_run_one.ipynb"), "w"), indent=1)
    print("[ok] wrote notebooks incl. kaggle_run_one.ipynb (THE ONE)")

if __name__ == "__main__":
    main()
