# ==============================================================================
# GODSHAND / THE_CONFRONTATION_PIPELINE — PROGRESSIVE DUAL-T4 CHUNK ENGINE
# REAL GPU INFERENCE: CHUNK 1 (SHOTS 1-5) + NUMPY/OPENCV PATCH
# ==============================================================================
import os, sys, subprocess, json, shutil, time
import urllib.request, urllib.parse, urllib.error
from pathlib import Path
import torch, gc, re
import numpy as np
from PIL import Image

# ==============================================================================
# STEP 1: PROGRESSIVE CHUNKING CONTROLS & REAL GPU ENVIRONMENT
# ==============================================================================
KIT_ROOT = '/kaggle/working'
os.chdir(KIT_ROOT)

# --- HANDS-OFF PROGRESSIVE RENDER (no per-session knob to touch) ---
# You never edit a chunk number. Each session renders the WHOLE film's
# remaining shots until it must stop before Kaggle's ~12h cap; finished
# shots are skipped next run. Just Run All again next session to continue.
KIT_CHUNK_INDEX = 1            # retained only so downstream refs keep working
KIT_CHUNK_SIZE  = 1           # unused in hands-off mode (kept for compatibility)
TOTAL_SHOTS     = 435          # PROVISIONAL; REAL total auto-derived from the story below
os.environ['KIT_MAX_HOURS']  = '11'          # stop cleanly before the ~12h session cap
os.environ['KIT_FAST_POST']   = '1'           # skip 60fps CPU interpolation per shot;
                                               # use upscale_hd_kaggle.ipynb on CPU after download
os.environ['KIT_SHOT_START'] = '1'           # full range; render_motion skips already-done shots
os.environ['KIT_SHOT_END']   = str(TOTAL_SHOTS)
os.environ['KIT_CHUNK_LABEL']= 'Beholder_FULL'
_lo, _hi = 1, TOTAL_SHOTS
print('[auto] HANDS-OFF render: full range, resume-on-rerun, budget=%sh (no chunk knob).' % os.environ['KIT_MAX_HOURS'])

os.environ['KIT_CONFIG']                = 'config_confrontation.yaml'
os.environ['KIT_RENDER_MODE']           = 'motion_stills'
os.environ['PYTHONPATH']                = 'src:' + os.environ.get('PYTHONPATH', '')
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']   = 'expandable_segments:True,max_split_size_mb:128'

# --- REAL GPU MODE ---
os.environ['COMFY_MOCK']                = '0'    # 0 = REAL inference (flip to '1' for a free mock dry-run)
os.environ['KIT_FORCE_RERENDER']        = '0'    # 0 = KEEP finished clips so re-running RESUMES (never re-render done shots)
os.environ['CONFIRM_FULL_RUN']          = '1'
os.environ['KIT_MOTION_STEPS']          = '28'
os.environ['KIT_LTXV_FPS']              = '18'   # lower LTXV frame_rate = more motion (balanced vs melting)

# --- Hugging Face auth from Kaggle Secrets (higher rate limits + faster downloads) ---
try:
    from kaggle_secrets import UserSecretsClient
    _usc = UserSecretsClient(); _hf = None
    for _lbl in ('HF_TOKEN','HUGGINGFACE_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','hf_token'):
        try:
            _v = _usc.get_secret(_lbl)
        except Exception:
            _v = None
        if _v:
            _hf = _v.strip(); print("[hf] loaded token from Kaggle secret '%s'" % _lbl); break
    if _hf:
        for _k in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN'):
            os.environ[_k] = _hf
        try:
            from huggingface_hub import login as _hf_login, get_token as _hf_get
            _hf_login(token=_hf, add_to_git_credential=False)
            print('[hf] login() ok; get_token sees token: %s' % bool(_hf_get()))
        except Exception as _le:
            print('[hf] login() skipped (%s)' % _le)
        print('[hf] Hugging Face authenticated for this session.')
    else:
        print("[hf] No HF token secret found -- add a Kaggle Secret named HF_TOKEN and attach it via Add-ons > Secrets. Continuing unauthenticated.")
except Exception as _e:
    print('[hf] Kaggle secrets unavailable (%s); continuing unauthenticated.' % _e)

print(f"--- 1. INITIALIZING ENGINE: REAL RENDERING SHOTS {os.environ['KIT_SHOT_START']} TO {os.environ['KIT_SHOT_END']} ---")
gc.collect(); torch.cuda.empty_cache()

# ==============================================================================
# STEP 2: MOUNT AND SYNC DATASET KIT ("GODSHAND")
# ==============================================================================
kit_src = None
for root, dirs, files in os.walk('/kaggle/input'):
    if os.path.basename(root) == 'src' and 'render_motion.py' in files:
        kit_src = os.path.dirname(root)
        break
assert kit_src, 'ERROR: no attached dataset contains src/ — attach GODSHAND dataset.'
print(f'[setup] GODSHAND kit mounted from: {kit_src}')

for entry in os.listdir(kit_src):
    s = os.path.join(kit_src, entry)
    d = os.path.join(KIT_ROOT, entry)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

for req in ['src/render_motion.py', 'src/comfy_client.py', 'workflows/flf2v.json',
            'config_confrontation.yaml', 'shots_confrontation.json']:
    assert os.path.exists(req), f'MISSING REQUIRED FILE: {req}'
print('[setup] kit file structure verified OK.')

# ==============================================================================
# EDIT YOUR STORY HERE -- the ONLY place to change the film.
#   Each shot = id / secs / who / camera / song / action / palette.
#   'action' is what happens on screen. Top-level 'look' & 'avoid' set the
#   style + negatives for EVERY shot (add 'look:'/'avoid:' inside a shot only
#   to override that one shot). Edit below and re-run the whole notebook --
#   shots_confrontation.json is rebuilt automatically. Never touch the big JSON.
# ==============================================================================
STORY_YAML = r'''
film: THE SAGA OF THE BEHOLDER
note: The Journey of the Ordinary Man and the Angel. A memoir of forty-seven years in the purest form. jack = the ordinary man; julie = the woman/angel.
look: grounded cinematic realism in its purest form, no polish, no gloss, 35mm film grain, natural motivated lighting, restrained intimacy, expressionist mood, a lonely modern fable of one ordinary man
avoid: nudity, sexual content, cartoon, anime, cgi, 3d render, video game, deformed anatomy, extra fingers, extra limbs, distorted face, plastic skin, melting, warping, morphing, smearing, dissolving, wobbling, liquefying, blurry, ghosting, flickering, temporal artifacts, unstable geometry, text, watermark, logo, subtitles, flat lifeless daylight
render:
  fps: 24
  width: 1280
  height: 720
  base_model: Wan2.2-S2V
  quant: Q8
  steps: 30
  cfg: 5.5
  sampler: uni_pc
  scheduler: beta
  interpolate_to_fps: 48
  upscale: 1.75
  rejection_samples: 3
characters:
  jack:
    anchor: man_anchor.png
    look: an ordinary lean tired man in his mid thirties to forties, slightly unkempt dark hair, faded olive-green jacket over a plain white tee, no armor, grounded and human
  julie:
    anchor: woman_anchor.png
    look: a warm captivating woman with genuine human warmth, later in soft timeless angelic robes, expressive eyes, an ordinary human being
shots:
- id: i01_jungle_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward a lone ordinary man of forty-seven in a faded olive-green jacket over a plain white tee, dwarfed by towering brutalist concrete architecture of a cold city, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey-blue concrete, overcast
  voiceover: true
  talking: false
- id: i01_jungle_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across a lone ordinary man of forty-seven in a faded olive-green jacket over a plain white tee, dwarfed by towering brutalist concrete architecture of a cold city, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey-blue concrete, overcast
  voiceover: true
  talking: false
- id: i01_jungle_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on a lone ordinary man of forty-seven in a faded olive-green jacket over a plain white tee, dwarfed by towering brutalist concrete architecture of a cold city, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey-blue concrete, overcast
  voiceover: true
  talking: false
- id: i01_jungle_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over a lone ordinary man of forty-seven in a faded olive-green jacket over a plain white tee, dwarfed by towering brutalist concrete architecture of a cold city, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey-blue concrete, overcast
  voiceover: true
  talking: false
- id: i01_jungle_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on a lone ordinary man of forty-seven in a faded olive-green jacket over a plain white tee, dwarfed by towering brutalist concrete architecture of a cold city, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey-blue concrete, overcast
  voiceover: true
  talking: false
- id: i01_jungle_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward a lone ordinary man of forty-seven in a faded olive-green jacket over a plain white tee, dwarfed by towering brutalist concrete architecture of a cold city, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey-blue concrete, overcast
  voiceover: true
  talking: false
- id: i02_war_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the tired man standing rigid, torn between two impulses, pressing the brakes and the accelerator of his own mind at once in a quiet internal war, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: muted steel, hard shadow
  voiceover: true
  talking: false
- id: i02_war_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the tired man standing rigid, torn between two impulses, pressing the brakes and the accelerator of his own mind at once in a quiet internal war, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: muted steel, hard shadow
  voiceover: true
  talking: false
- id: i02_war_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the tired man standing rigid, torn between two impulses, pressing the brakes and the accelerator of his own mind at once in a quiet internal war, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: muted steel, hard shadow
  voiceover: true
  talking: false
- id: i02_war_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the tired man standing rigid, torn between two impulses, pressing the brakes and the accelerator of his own mind at once in a quiet internal war, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: muted steel, hard shadow
  voiceover: true
  talking: false
- id: i02_war_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the tired man standing rigid, torn between two impulses, pressing the brakes and the accelerator of his own mind at once in a quiet internal war, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: muted steel, hard shadow
  voiceover: true
  talking: false
- id: i03_framework_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the disciplined man moving through a rigid daily routine inside a sterile fluorescent office, surviving the weight of his own nervous system, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fluorescent office, desaturated
  voiceover: true
  talking: false
- id: i03_framework_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the disciplined man moving through a rigid daily routine inside a sterile fluorescent office, surviving the weight of his own nervous system, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fluorescent office, desaturated
  voiceover: true
  talking: false
- id: i03_framework_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the disciplined man moving through a rigid daily routine inside a sterile fluorescent office, surviving the weight of his own nervous system, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fluorescent office, desaturated
  voiceover: true
  talking: false
- id: i03_framework_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the disciplined man moving through a rigid daily routine inside a sterile fluorescent office, surviving the weight of his own nervous system, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fluorescent office, desaturated
  voiceover: true
  talking: false
- id: i03_framework_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the disciplined man moving through a rigid daily routine inside a sterile fluorescent office, surviving the weight of his own nervous system, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fluorescent office, desaturated
  voiceover: true
  talking: false
- id: i04_she_enters_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward a warm captivating woman arriving into the office as a new co-worker, her easy genuine warmth like sudden fire in a freezing room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm amber against cold office
  voiceover: true
  talking: false
- id: i04_she_enters_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across a warm captivating woman arriving into the office as a new co-worker, her easy genuine warmth like sudden fire in a freezing room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm amber against cold office
  voiceover: true
  talking: false
- id: i04_she_enters_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on a warm captivating woman arriving into the office as a new co-worker, her easy genuine warmth like sudden fire in a freezing room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm amber against cold office
  voiceover: true
  talking: false
- id: i04_she_enters_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over a warm captivating woman arriving into the office as a new co-worker, her easy genuine warmth like sudden fire in a freezing room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm amber against cold office
  voiceover: true
  talking: false
- id: i04_she_enters_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on a warm captivating woman arriving into the office as a new co-worker, her easy genuine warmth like sudden fire in a freezing room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm amber against cold office
  voiceover: true
  talking: false
- id: i04_she_enters_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward a warm captivating woman arriving into the office as a new co-worker, her easy genuine warmth like sudden fire in a freezing room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm amber against cold office
  voiceover: true
  talking: false
- id: i05_between_desks_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the man and the woman working at nearby desks as a slow profound shift grows in the quiet spaces between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: office neutral, soft warmth
  voiceover: true
  talking: false
- id: i05_between_desks_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the man and the woman working at nearby desks as a slow profound shift grows in the quiet spaces between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: office neutral, soft warmth
  voiceover: true
  talking: false
- id: i05_between_desks_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the man and the woman working at nearby desks as a slow profound shift grows in the quiet spaces between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: office neutral, soft warmth
  voiceover: true
  talking: false
- id: i05_between_desks_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the man and the woman working at nearby desks as a slow profound shift grows in the quiet spaces between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: office neutral, soft warmth
  voiceover: true
  talking: false
- id: i05_between_desks_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the man and the woman working at nearby desks as a slow profound shift grows in the quiet spaces between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: office neutral, soft warmth
  voiceover: true
  talking: false
- id: i05_between_desks_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the man and the woman working at nearby desks as a slow profound shift grows in the quiet spaces between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: office neutral, soft warmth
  voiceover: true
  talking: false
- id: i06_captivation_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man stealing quiet glances, his affection slowly intensifying into reverence for the woman, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft focus warm
  voiceover: true
  talking: false
- id: i06_captivation_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man stealing quiet glances, his affection slowly intensifying into reverence for the woman, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft focus warm
  voiceover: true
  talking: false
- id: i06_captivation_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man stealing quiet glances, his affection slowly intensifying into reverence for the woman, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft focus warm
  voiceover: true
  talking: false
- id: i06_captivation_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man stealing quiet glances, his affection slowly intensifying into reverence for the woman, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft focus warm
  voiceover: true
  talking: false
- id: i06_captivation_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man stealing quiet glances, his affection slowly intensifying into reverence for the woman, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft focus warm
  voiceover: true
  talking: false
- id: i07_pedestal_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man building an invisible pedestal in his mind, elevating the woman into an angel he worships, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ethereal soft light
  voiceover: true
  talking: false
- id: i07_pedestal_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man building an invisible pedestal in his mind, elevating the woman into an angel he worships, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ethereal soft light
  voiceover: true
  talking: false
- id: i07_pedestal_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man building an invisible pedestal in his mind, elevating the woman into an angel he worships, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ethereal soft light
  voiceover: true
  talking: false
- id: i07_pedestal_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man building an invisible pedestal in his mind, elevating the woman into an angel he worships, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ethereal soft light
  voiceover: true
  talking: false
- id: i07_pedestal_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man building an invisible pedestal in his mind, elevating the woman into an angel he worships, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ethereal soft light
  voiceover: true
  talking: false
- id: i07_pedestal_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man building an invisible pedestal in his mind, elevating the woman into an angel he worships, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ethereal soft light
  voiceover: true
  talking: false
- id: i08_weight_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the woman quietly sensing the crushing weight of the silent reverence placed upon her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool tender
  voiceover: true
  talking: false
- id: i08_weight_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the woman quietly sensing the crushing weight of the silent reverence placed upon her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool tender
  voiceover: true
  talking: false
- id: i08_weight_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the woman quietly sensing the crushing weight of the silent reverence placed upon her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool tender
  voiceover: true
  talking: false
- id: i08_weight_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the woman quietly sensing the crushing weight of the silent reverence placed upon her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool tender
  voiceover: true
  talking: false
- id: i08_weight_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the woman quietly sensing the crushing weight of the silent reverence placed upon her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool tender
  voiceover: true
  talking: false
- id: i09_she_tries_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the woman gently trying to break the pedestal, showing him she is merely human, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm human
  voiceover: true
  talking: false
- id: i09_she_tries_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the woman gently trying to break the pedestal, showing him she is merely human, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm human
  voiceover: true
  talking: false
- id: i09_she_tries_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the woman gently trying to break the pedestal, showing him she is merely human, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm human
  voiceover: true
  talking: false
- id: i09_she_tries_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the woman gently trying to break the pedestal, showing him she is merely human, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm human
  voiceover: true
  talking: false
- id: i09_she_tries_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the woman gently trying to break the pedestal, showing him she is merely human, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm human
  voiceover: true
  talking: false
- id: i10_she_changes_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the woman reading his unspoken traditional heart and quietly transforming, covering herself in soft timeless angelic robes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft white and gold, angelic
  voiceover: true
  talking: false
- id: i10_she_changes_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the woman reading his unspoken traditional heart and quietly transforming, covering herself in soft timeless angelic robes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft white and gold, angelic
  voiceover: true
  talking: false
- id: i10_she_changes_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the woman reading his unspoken traditional heart and quietly transforming, covering herself in soft timeless angelic robes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft white and gold, angelic
  voiceover: true
  talking: false
- id: i10_she_changes_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the woman reading his unspoken traditional heart and quietly transforming, covering herself in soft timeless angelic robes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft white and gold, angelic
  voiceover: true
  talking: false
- id: i10_she_changes_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the woman reading his unspoken traditional heart and quietly transforming, covering herself in soft timeless angelic robes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft white and gold, angelic
  voiceover: true
  talking: false
- id: i10_she_changes_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the woman reading his unspoken traditional heart and quietly transforming, covering herself in soft timeless angelic robes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft white and gold, angelic
  voiceover: true
  talking: false
- id: i10_she_changes_07
  secs: 4
  who:
  - julie
  camera: low-angle crane up
  song: prelude
  action: a low-angle shot slowly craning up at the woman reading his unspoken traditional heart and quietly transforming, covering herself in soft timeless angelic robes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft white and gold, angelic
  voiceover: true
  talking: false
- id: i11_beside_desk_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward one fine morning the woman in soft angelic robes materializing and standing quietly beside the man at his desk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm morning light, office
  voiceover: true
  talking: false
- id: i11_beside_desk_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across one fine morning the woman in soft angelic robes materializing and standing quietly beside the man at his desk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm morning light, office
  voiceover: true
  talking: false
- id: i11_beside_desk_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on one fine morning the woman in soft angelic robes materializing and standing quietly beside the man at his desk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm morning light, office
  voiceover: true
  talking: false
- id: i11_beside_desk_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over one fine morning the woman in soft angelic robes materializing and standing quietly beside the man at his desk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm morning light, office
  voiceover: true
  talking: false
- id: i11_beside_desk_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on one fine morning the woman in soft angelic robes materializing and standing quietly beside the man at his desk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm morning light, office
  voiceover: true
  talking: false
- id: i11_beside_desk_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward one fine morning the woman in soft angelic robes materializing and standing quietly beside the man at his desk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm morning light, office
  voiceover: true
  talking: false
- id: i11_beside_desk_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: prelude
  action: a low-angle shot slowly craning up at one fine morning the woman in soft angelic robes materializing and standing quietly beside the man at his desk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm morning light, office
  voiceover: true
  talking: false
- id: i12_glued_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the man glued to his glowing screen, a violent desperate restructuring happening inside his chest, unable to look up, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold screen glow
  voiceover: true
  talking: false
- id: i12_glued_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the man glued to his glowing screen, a violent desperate restructuring happening inside his chest, unable to look up, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold screen glow
  voiceover: true
  talking: false
- id: i12_glued_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the man glued to his glowing screen, a violent desperate restructuring happening inside his chest, unable to look up, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold screen glow
  voiceover: true
  talking: false
- id: i12_glued_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the man glued to his glowing screen, a violent desperate restructuring happening inside his chest, unable to look up, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold screen glow
  voiceover: true
  talking: false
- id: i12_glued_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the man glued to his glowing screen, a violent desperate restructuring happening inside his chest, unable to look up, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold screen glow
  voiceover: true
  talking: false
- id: i12_glued_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the man glued to his glowing screen, a violent desperate restructuring happening inside his chest, unable to look up, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold screen glow
  voiceover: true
  talking: false
- id: i13_khilona_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the fragile heart of the man imagined as a delicate clay toy, a khilona, that even a careless touch could shatter, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fragile warm, shallow focus
  voiceover: true
  talking: false
- id: i13_khilona_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the fragile heart of the man imagined as a delicate clay toy, a khilona, that even a careless touch could shatter, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fragile warm, shallow focus
  voiceover: true
  talking: false
- id: i13_khilona_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the fragile heart of the man imagined as a delicate clay toy, a khilona, that even a careless touch could shatter, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fragile warm, shallow focus
  voiceover: true
  talking: false
- id: i13_khilona_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the fragile heart of the man imagined as a delicate clay toy, a khilona, that even a careless touch could shatter, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fragile warm, shallow focus
  voiceover: true
  talking: false
- id: i13_khilona_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the fragile heart of the man imagined as a delicate clay toy, a khilona, that even a careless touch could shatter, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: fragile warm, shallow focus
  voiceover: true
  talking: false
- id: i14_packaging_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the man frozen, trying to package his love into a vessel worthy of her, refusing to look up out of paralyzing reverence, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tense warm-cold
  voiceover: true
  talking: false
- id: i14_packaging_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the man frozen, trying to package his love into a vessel worthy of her, refusing to look up out of paralyzing reverence, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tense warm-cold
  voiceover: true
  talking: false
- id: i14_packaging_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the man frozen, trying to package his love into a vessel worthy of her, refusing to look up out of paralyzing reverence, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tense warm-cold
  voiceover: true
  talking: false
- id: i14_packaging_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the man frozen, trying to package his love into a vessel worthy of her, refusing to look up out of paralyzing reverence, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tense warm-cold
  voiceover: true
  talking: false
- id: i14_packaging_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the man frozen, trying to package his love into a vessel worthy of her, refusing to look up out of paralyzing reverence, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tense warm-cold
  voiceover: true
  talking: false
- id: i14_packaging_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the man frozen, trying to package his love into a vessel worthy of her, refusing to look up out of paralyzing reverence, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tense warm-cold
  voiceover: true
  talking: false
- id: i15_two_clocks_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the woman standing beside him waiting, the silence stretching, two different clocks ticking in the same room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: still suspended air
  voiceover: true
  talking: false
- id: i15_two_clocks_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the woman standing beside him waiting, the silence stretching, two different clocks ticking in the same room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: still suspended air
  voiceover: true
  talking: false
- id: i15_two_clocks_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the woman standing beside him waiting, the silence stretching, two different clocks ticking in the same room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: still suspended air
  voiceover: true
  talking: false
- id: i15_two_clocks_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the woman standing beside him waiting, the silence stretching, two different clocks ticking in the same room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: still suspended air
  voiceover: true
  talking: false
- id: i15_two_clocks_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the woman standing beside him waiting, the silence stretching, two different clocks ticking in the same room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: still suspended air
  voiceover: true
  talking: false
- id: i15_two_clocks_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the woman standing beside him waiting, the silence stretching, two different clocks ticking in the same room, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: still suspended air
  voiceover: true
  talking: false
- id: i16_the_plea_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i16_the_plea_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i16_the_plea_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i16_the_plea_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i16_the_plea_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i16_the_plea_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i16_the_plea_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: confrontation
  action: a low-angle shot slowly craning up at the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i16_the_plea_08
  secs: 4
  who:
  - jack
  - julie
  camera: slow tracking
  song: confrontation
  action: a slow tracking shot moving alongside the woman finally and heartbroken asking him the burning question, do you not look at people in the eyes when you talk, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: charged intimate
  voiceover: true
  talking: false
- id: i17_freezes_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the shock breaking his trance, the man beginning to turn his head to realize the enormity of the moment, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: sharp cold
  voiceover: true
  talking: false
- id: i17_freezes_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the shock breaking his trance, the man beginning to turn his head to realize the enormity of the moment, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: sharp cold
  voiceover: true
  talking: false
- id: i17_freezes_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the shock breaking his trance, the man beginning to turn his head to realize the enormity of the moment, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: sharp cold
  voiceover: true
  talking: false
- id: i17_freezes_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the shock breaking his trance, the man beginning to turn his head to realize the enormity of the moment, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: sharp cold
  voiceover: true
  talking: false
- id: i17_freezes_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the shock breaking his trance, the man beginning to turn his head to realize the enormity of the moment, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: sharp cold
  voiceover: true
  talking: false
- id: i18_too_late_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the man opening his mouth to apologize and turning to find the space beside him empty, she is gone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow cold
  voiceover: true
  talking: false
- id: i18_too_late_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the man opening his mouth to apologize and turning to find the space beside him empty, she is gone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow cold
  voiceover: true
  talking: false
- id: i18_too_late_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the man opening his mouth to apologize and turning to find the space beside him empty, she is gone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow cold
  voiceover: true
  talking: false
- id: i18_too_late_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the man opening his mouth to apologize and turning to find the space beside him empty, she is gone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow cold
  voiceover: true
  talking: false
- id: i18_too_late_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the man opening his mouth to apologize and turning to find the space beside him empty, she is gone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow cold
  voiceover: true
  talking: false
- id: i18_too_late_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the man opening his mouth to apologize and turning to find the space beside him empty, she is gone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow cold
  voiceover: true
  talking: false
- id: i19_she_leaves_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the woman walking out of the room carrying the devastating belief that he did not love her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding cold
  voiceover: true
  talking: false
- id: i19_she_leaves_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the woman walking out of the room carrying the devastating belief that he did not love her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding cold
  voiceover: true
  talking: false
- id: i19_she_leaves_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the woman walking out of the room carrying the devastating belief that he did not love her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding cold
  voiceover: true
  talking: false
- id: i19_she_leaves_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the woman walking out of the room carrying the devastating belief that he did not love her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding cold
  voiceover: true
  talking: false
- id: i19_she_leaves_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the woman walking out of the room carrying the devastating belief that he did not love her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding cold
  voiceover: true
  talking: false
- id: i19_she_leaves_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the woman walking out of the room carrying the devastating belief that he did not love her, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding cold
  voiceover: true
  talking: false
- id: i20_collapse_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man entire internal framework collapsing as he abandons his desk and his world, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: crumbling grey
  voiceover: true
  talking: false
- id: i20_collapse_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man entire internal framework collapsing as he abandons his desk and his world, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: crumbling grey
  voiceover: true
  talking: false
- id: i20_collapse_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man entire internal framework collapsing as he abandons his desk and his world, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: crumbling grey
  voiceover: true
  talking: false
- id: i20_collapse_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man entire internal framework collapsing as he abandons his desk and his world, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: crumbling grey
  voiceover: true
  talking: false
- id: i20_collapse_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man entire internal framework collapsing as he abandons his desk and his world, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: crumbling grey
  voiceover: true
  talking: false
- id: i21_nowhere_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man falling into a locationless abyss, a station of the soul called Nowhere, utterly alone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim void, isolate
  voiceover: true
  talking: false
- id: i21_nowhere_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man falling into a locationless abyss, a station of the soul called Nowhere, utterly alone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim void, isolate
  voiceover: true
  talking: false
- id: i21_nowhere_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man falling into a locationless abyss, a station of the soul called Nowhere, utterly alone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim void, isolate
  voiceover: true
  talking: false
- id: i21_nowhere_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man falling into a locationless abyss, a station of the soul called Nowhere, utterly alone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim void, isolate
  voiceover: true
  talking: false
- id: i21_nowhere_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man falling into a locationless abyss, a station of the soul called Nowhere, utterly alone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim void, isolate
  voiceover: true
  talking: false
- id: i21_nowhere_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man falling into a locationless abyss, a station of the soul called Nowhere, utterly alone, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim void, isolate
  voiceover: true
  talking: false
- id: ii01_road_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the lone man standing on a silent sun-blasted small-town road at midday, empty asphalt shimmering like a desert under an overexposed white sky, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached white, heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii01_road_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the lone man standing on a silent sun-blasted small-town road at midday, empty asphalt shimmering like a desert under an overexposed white sky, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached white, heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii01_road_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the lone man standing on a silent sun-blasted small-town road at midday, empty asphalt shimmering like a desert under an overexposed white sky, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached white, heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii01_road_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the lone man standing on a silent sun-blasted small-town road at midday, empty asphalt shimmering like a desert under an overexposed white sky, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached white, heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii01_road_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the lone man standing on a silent sun-blasted small-town road at midday, empty asphalt shimmering like a desert under an overexposed white sky, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached white, heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii01_road_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the lone man standing on a silent sun-blasted small-town road at midday, empty asphalt shimmering like a desert under an overexposed white sky, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached white, heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii01_road_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: search
  action: a low-angle shot slowly craning up at the lone man standing on a silent sun-blasted small-town road at midday, empty asphalt shimmering like a desert under an overexposed white sky, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached white, heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii02_siren_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the empty small town in oppressive stillness, the faint unanswered wail of a distant police siren hanging in the air, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: washed-out, tense
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii02_siren_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the empty small town in oppressive stillness, the faint unanswered wail of a distant police siren hanging in the air, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: washed-out, tense
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii02_siren_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the empty small town in oppressive stillness, the faint unanswered wail of a distant police siren hanging in the air, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: washed-out, tense
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii02_siren_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the empty small town in oppressive stillness, the faint unanswered wail of a distant police siren hanging in the air, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: washed-out, tense
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii02_siren_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the empty small town in oppressive stillness, the faint unanswered wail of a distant police siren hanging in the air, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: washed-out, tense
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii03_approaches_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the woman approaching unhurried out of the shimmering haze, open palms extended, wearing a disarming genuine human smile, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: golden haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii03_approaches_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the woman approaching unhurried out of the shimmering haze, open palms extended, wearing a disarming genuine human smile, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: golden haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii03_approaches_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the woman approaching unhurried out of the shimmering haze, open palms extended, wearing a disarming genuine human smile, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: golden haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii03_approaches_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the woman approaching unhurried out of the shimmering haze, open palms extended, wearing a disarming genuine human smile, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: golden haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii03_approaches_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the woman approaching unhurried out of the shimmering haze, open palms extended, wearing a disarming genuine human smile, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: golden haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii03_approaches_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the woman approaching unhurried out of the shimmering haze, open palms extended, wearing a disarming genuine human smile, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: golden haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii03_approaches_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: search
  action: a low-angle shot slowly craning up at the woman approaching unhurried out of the shimmering haze, open palms extended, wearing a disarming genuine human smile, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: golden haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii04_tender_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward for a beat in the golden afternoon haze the world feeling tender safe and deeply intimate between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm gold, soft
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii04_tender_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across for a beat in the golden afternoon haze the world feeling tender safe and deeply intimate between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm gold, soft
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii04_tender_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on for a beat in the golden afternoon haze the world feeling tender safe and deeply intimate between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm gold, soft
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii04_tender_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over for a beat in the golden afternoon haze the world feeling tender safe and deeply intimate between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm gold, soft
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii04_tender_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on for a beat in the golden afternoon haze the world feeling tender safe and deeply intimate between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm gold, soft
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii04_tender_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward for a beat in the golden afternoon haze the world feeling tender safe and deeply intimate between them, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm gold, soft
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii05_guard_drops_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man guard dropping completely, leaning toward her ready to hand over the unbroken clay toy of his chest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm, vulnerable
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii05_guard_drops_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man guard dropping completely, leaning toward her ready to hand over the unbroken clay toy of his chest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm, vulnerable
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii05_guard_drops_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man guard dropping completely, leaning toward her ready to hand over the unbroken clay toy of his chest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm, vulnerable
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii05_guard_drops_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man guard dropping completely, leaning toward her ready to hand over the unbroken clay toy of his chest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm, vulnerable
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii05_guard_drops_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man guard dropping completely, leaning toward her ready to hand over the unbroken clay toy of his chest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm, vulnerable
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii05_guard_drops_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man guard dropping completely, leaning toward her ready to hand over the unbroken clay toy of his chest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm, vulnerable
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii06_the_turn_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the psychological turn as the golden light drains into a harsh bleached-white glare, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: harsh white glare
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii06_the_turn_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the psychological turn as the golden light drains into a harsh bleached-white glare, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: harsh white glare
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii06_the_turn_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the psychological turn as the golden light drains into a harsh bleached-white glare, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: harsh white glare
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii06_the_turn_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the psychological turn as the golden light drains into a harsh bleached-white glare, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: harsh white glare
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii06_the_turn_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the psychological turn as the golden light drains into a harsh bleached-white glare, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: harsh white glare
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii06_the_turn_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the psychological turn as the golden light drains into a harsh bleached-white glare, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: harsh white glare
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii07_curdles_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the woman tender smile holding a beat too long, curdling from human affection into an artificial predatory mask, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold white, flat
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii07_curdles_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the woman tender smile holding a beat too long, curdling from human affection into an artificial predatory mask, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold white, flat
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii07_curdles_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the woman tender smile holding a beat too long, curdling from human affection into an artificial predatory mask, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold white, flat
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii07_curdles_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the woman tender smile holding a beat too long, curdling from human affection into an artificial predatory mask, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold white, flat
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii07_curdles_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the woman tender smile holding a beat too long, curdling from human affection into an artificial predatory mask, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold white, flat
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii07_curdles_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the woman tender smile holding a beat too long, curdling from human affection into an artificial predatory mask, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold white, flat
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii07_curdles_07
  secs: 4
  who:
  - julie
  camera: low-angle crane up
  song: confrontation
  action: a low-angle shot slowly craning up at the woman tender smile holding a beat too long, curdling from human affection into an artificial predatory mask, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold white, flat
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii08_dead_eyes_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the woman eyes gone completely flat, dead, hollow and ice-cold, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: icy, drained
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii08_dead_eyes_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the woman eyes gone completely flat, dead, hollow and ice-cold, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: icy, drained
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii08_dead_eyes_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the woman eyes gone completely flat, dead, hollow and ice-cold, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: icy, drained
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii08_dead_eyes_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the woman eyes gone completely flat, dead, hollow and ice-cold, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: icy, drained
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii08_dead_eyes_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the woman eyes gone completely flat, dead, hollow and ice-cold, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: icy, drained
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii08_dead_eyes_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the woman eyes gone completely flat, dead, hollow and ice-cold, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: icy, drained
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii09_realization_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the man realizing with a chill down his spine that real danger wears a sundress, an angel face and an inviting smile, warmth used as a weapon, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold dread, bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii09_realization_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the man realizing with a chill down his spine that real danger wears a sundress, an angel face and an inviting smile, warmth used as a weapon, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold dread, bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii09_realization_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the man realizing with a chill down his spine that real danger wears a sundress, an angel face and an inviting smile, warmth used as a weapon, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold dread, bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii09_realization_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the man realizing with a chill down his spine that real danger wears a sundress, an angel face and an inviting smile, warmth used as a weapon, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold dread, bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii09_realization_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the man realizing with a chill down his spine that real danger wears a sundress, an angel face and an inviting smile, warmth used as a weapon, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold dread, bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii09_realization_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the man realizing with a chill down his spine that real danger wears a sundress, an angel face and an inviting smile, warmth used as a weapon, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold dread, bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii09_realization_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: confrontation
  action: a low-angle shot slowly craning up at the man realizing with a chill down his spine that real danger wears a sundress, an angel face and an inviting smile, warmth used as a weapon, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold dread, bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii10_no_hide_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the woman stepping agonizingly close, surrounding his vulnerability with nowhere left to hide, the air turning to freezing ice under the midday sun, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii10_no_hide_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the woman stepping agonizingly close, surrounding his vulnerability with nowhere left to hide, the air turning to freezing ice under the midday sun, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii10_no_hide_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the woman stepping agonizingly close, surrounding his vulnerability with nowhere left to hide, the air turning to freezing ice under the midday sun, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii10_no_hide_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the woman stepping agonizingly close, surrounding his vulnerability with nowhere left to hide, the air turning to freezing ice under the midday sun, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii10_no_hide_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the woman stepping agonizingly close, surrounding his vulnerability with nowhere left to hide, the air turning to freezing ice under the midday sun, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii10_no_hide_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the woman stepping agonizingly close, surrounding his vulnerability with nowhere left to hide, the air turning to freezing ice under the midday sun, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen bright
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii11_cut_black_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward just before contact the man mind executing a hard cut to black, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: black void
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii11_cut_black_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across just before contact the man mind executing a hard cut to black, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: black void
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii11_cut_black_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on just before contact the man mind executing a hard cut to black, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: black void
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii11_cut_black_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over just before contact the man mind executing a hard cut to black, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: black void
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii11_cut_black_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on just before contact the man mind executing a hard cut to black, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: black void
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii12_empty_road_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the light returning to reveal the empty road and the shattered human silhouette left behind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached emptiness
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii12_empty_road_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the light returning to reveal the empty road and the shattered human silhouette left behind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached emptiness
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii12_empty_road_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the light returning to reveal the empty road and the shattered human silhouette left behind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached emptiness
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii12_empty_road_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the light returning to reveal the empty road and the shattered human silhouette left behind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached emptiness
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii12_empty_road_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the light returning to reveal the empty road and the shattered human silhouette left behind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached emptiness
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii12_empty_road_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the light returning to reveal the empty road and the shattered human silhouette left behind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: bleached emptiness
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii13_walks_away_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the woman walking away unhurried into the distance, never once turning around, never looking back, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii13_walks_away_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the woman walking away unhurried into the distance, never once turning around, never looking back, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii13_walks_away_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the woman walking away unhurried into the distance, never once turning around, never looking back, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii13_walks_away_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the woman walking away unhurried into the distance, never once turning around, never looking back, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii13_walks_away_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the woman walking away unhurried into the distance, never once turning around, never looking back, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii13_walks_away_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the woman walking away unhurried into the distance, never once turning around, never looking back, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: receding heat-haze
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii14_cold_spine_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man alone understanding the lethal pain, that they never promise and never return, that is why they are the cold down the spine, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow bleached
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii14_cold_spine_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man alone understanding the lethal pain, that they never promise and never return, that is why they are the cold down the spine, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow bleached
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii14_cold_spine_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man alone understanding the lethal pain, that they never promise and never return, that is why they are the cold down the spine, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow bleached
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii14_cold_spine_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man alone understanding the lethal pain, that they never promise and never return, that is why they are the cold down the spine, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow bleached
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii14_cold_spine_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man alone understanding the lethal pain, that they never promise and never return, that is why they are the cold down the spine, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow bleached
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: ii14_cold_spine_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man alone understanding the lethal pain, that they never promise and never return, that is why they are the cold down the spine, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow bleached
  voiceover: true
  talking: false
  look: bleached overexposed midday desert light, blinding white sky, heat-haze shimmer, sun-blasted empty small-town road, harsh high-key, 35mm grain
- id: iii01_insomnia_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward months of insomnia, the exhausted man lying awake in the dark small hours as decades of war culminate, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim blue-grey night
  voiceover: true
  talking: false
- id: iii01_insomnia_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across months of insomnia, the exhausted man lying awake in the dark small hours as decades of war culminate, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim blue-grey night
  voiceover: true
  talking: false
- id: iii01_insomnia_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on months of insomnia, the exhausted man lying awake in the dark small hours as decades of war culminate, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim blue-grey night
  voiceover: true
  talking: false
- id: iii01_insomnia_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over months of insomnia, the exhausted man lying awake in the dark small hours as decades of war culminate, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim blue-grey night
  voiceover: true
  talking: false
- id: iii01_insomnia_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on months of insomnia, the exhausted man lying awake in the dark small hours as decades of war culminate, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim blue-grey night
  voiceover: true
  talking: false
- id: iii01_insomnia_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward months of insomnia, the exhausted man lying awake in the dark small hours as decades of war culminate, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim blue-grey night
  voiceover: true
  talking: false
- id: iii02_no_clock_out_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man mind refusing to clock out, carrying its trauma up into the clouds, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: murky dream-grey
  voiceover: true
  talking: false
- id: iii02_no_clock_out_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man mind refusing to clock out, carrying its trauma up into the clouds, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: murky dream-grey
  voiceover: true
  talking: false
- id: iii02_no_clock_out_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man mind refusing to clock out, carrying its trauma up into the clouds, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: murky dream-grey
  voiceover: true
  talking: false
- id: iii02_no_clock_out_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man mind refusing to clock out, carrying its trauma up into the clouds, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: murky dream-grey
  voiceover: true
  talking: false
- id: iii02_no_clock_out_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man mind refusing to clock out, carrying its trauma up into the clouds, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: murky dream-grey
  voiceover: true
  talking: false
- id: iii03_flight_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man flying in a dream above the brutalist city, not to fire weapons but to seek an aerial vantage to understand, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel-grey sky, vast
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii03_flight_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man flying in a dream above the brutalist city, not to fire weapons but to seek an aerial vantage to understand, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel-grey sky, vast
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii03_flight_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man flying in a dream above the brutalist city, not to fire weapons but to seek an aerial vantage to understand, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel-grey sky, vast
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii03_flight_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man flying in a dream above the brutalist city, not to fire weapons but to seek an aerial vantage to understand, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel-grey sky, vast
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii03_flight_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man flying in a dream above the brutalist city, not to fire weapons but to seek an aerial vantage to understand, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel-grey sky, vast
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii03_flight_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man flying in a dream above the brutalist city, not to fire weapons but to seek an aerial vantage to understand, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel-grey sky, vast
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii03_flight_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: search
  action: a low-angle shot slowly craning up at the man flying in a dream above the brutalist city, not to fire weapons but to seek an aerial vantage to understand, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel-grey sky, vast
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii04_rifles_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward every time he tries to land his exhausted mind, armed rifles point up at him from below, denying him the ground, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey, threat
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii04_rifles_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across every time he tries to land his exhausted mind, armed rifles point up at him from below, denying him the ground, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey, threat
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii04_rifles_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on every time he tries to land his exhausted mind, armed rifles point up at him from below, denying him the ground, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey, threat
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii04_rifles_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over every time he tries to land his exhausted mind, armed rifles point up at him from below, denying him the ground, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey, threat
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii04_rifles_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on every time he tries to land his exhausted mind, armed rifles point up at him from below, denying him the ground, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey, threat
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii04_rifles_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward every time he tries to land his exhausted mind, armed rifles point up at him from below, denying him the ground, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold grey, threat
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii05_circles_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man circling for hours in the dark sky, denied any place to land, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dark steel, weary
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii05_circles_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man circling for hours in the dark sky, denied any place to land, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dark steel, weary
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii05_circles_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man circling for hours in the dark sky, denied any place to land, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dark steel, weary
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii05_circles_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man circling for hours in the dark sky, denied any place to land, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dark steel, weary
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii05_circles_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man circling for hours in the dark sky, denied any place to land, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dark steel, weary
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii05_circles_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man circling for hours in the dark sky, denied any place to land, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dark steel, weary
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii06_sky_parts_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward from his high vantage the sky parting to reveal an apocalyptic truth, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ominous grey, parting
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii06_sky_parts_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across from his high vantage the sky parting to reveal an apocalyptic truth, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ominous grey, parting
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii06_sky_parts_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on from his high vantage the sky parting to reveal an apocalyptic truth, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ominous grey, parting
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii06_sky_parts_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over from his high vantage the sky parting to reveal an apocalyptic truth, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ominous grey, parting
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii06_sky_parts_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on from his high vantage the sky parting to reveal an apocalyptic truth, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ominous grey, parting
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii06_sky_parts_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward from his high vantage the sky parting to reveal an apocalyptic truth, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: ominous grey, parting
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: confrontation
  action: a low-angle shot slowly craning up at an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii07_motherships_08
  secs: 4
  who:
  - jack
  camera: slow tracking
  song: confrontation
  action: a slow tracking shot moving alongside an overwhelming armada of colossal silent motherships, the Liberators, descending through the clouds over the concrete jungle, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: vast steel-grey, awe
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii08_dwarf_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the massive silent hulls moving in formation, dwarfing the skyscrapers far below, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: immense scale, cold
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii08_dwarf_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the massive silent hulls moving in formation, dwarfing the skyscrapers far below, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: immense scale, cold
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii08_dwarf_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the massive silent hulls moving in formation, dwarfing the skyscrapers far below, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: immense scale, cold
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii08_dwarf_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the massive silent hulls moving in formation, dwarfing the skyscrapers far below, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: immense scale, cold
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii08_dwarf_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the massive silent hulls moving in formation, dwarfing the skyscrapers far below, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: immense scale, cold
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii08_dwarf_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the massive silent hulls moving in formation, dwarfing the skyscrapers far below, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: immense scale, cold
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: confrontation
  action: a low-angle shot slowly craning up at the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii09_eclipse_08
  secs: 4
  who:
  - jack
  camera: slow tracking
  song: confrontation
  action: a slow tracking shot moving alongside the flagship passing overhead, its immense shadow physically eclipsing the sun and plunging the world into freezing desaturated steel-grey twilight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: eclipse shadow, twilight
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii10_symbol_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward standing on a rooftop below the fleet, the towering symbol he built over years, an image of the angel he worshipped, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: monument, cold grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii10_symbol_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across standing on a rooftop below the fleet, the towering symbol he built over years, an image of the angel he worshipped, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: monument, cold grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii10_symbol_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on standing on a rooftop below the fleet, the towering symbol he built over years, an image of the angel he worshipped, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: monument, cold grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii10_symbol_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over standing on a rooftop below the fleet, the towering symbol he built over years, an image of the angel he worshipped, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: monument, cold grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii10_symbol_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on standing on a rooftop below the fleet, the towering symbol he built over years, an image of the angel he worshipped, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: monument, cold grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii10_symbol_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward standing on a rooftop below the fleet, the towering symbol he built over years, an image of the angel he worshipped, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: monument, cold grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_07
  secs: 4
  who:
  - julie
  camera: low-angle crane up
  song: confrontation
  action: a low-angle shot slowly craning up at as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii11_shatter_08
  secs: 4
  who:
  - julie
  camera: slow tracking
  song: confrontation
  action: a slow tracking shot moving alongside as the shadow swallows the light the revered symbol cracking and shattering into pieces, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: shattering marble, grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii12_marble_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: confrontation
  action: a wide shot slowly pushing in toward the woman who dressed for him and asked for his eyes standing under the alien ships, her smile frozen into cold marble, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen pale grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii12_marble_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: confrontation
  action: a wide slow dolly gliding across the woman who dressed for him and asked for his eyes standing under the alien ships, her smile frozen into cold marble, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen pale grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii12_marble_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: confrontation
  action: a medium shot slowly pushing in on the woman who dressed for him and asked for his eyes standing under the alien ships, her smile frozen into cold marble, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen pale grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii12_marble_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: confrontation
  action: an intimate close-up slowly drifting over the woman who dressed for him and asked for his eyes standing under the alien ships, her smile frozen into cold marble, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen pale grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii12_marble_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: confrontation
  action: an extreme close-up with a slow rack focus on the woman who dressed for him and asked for his eyes standing under the alien ships, her smile frozen into cold marble, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen pale grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii12_marble_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: confrontation
  action: a slow over-the-shoulder move toward the woman who dressed for him and asked for his eyes standing under the alien ships, her smile frozen into cold marble, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: frozen pale grey
  voiceover: true
  talking: false
  look: apocalyptic steel-grey twilight, colossal motherships eclipsing the sun, desaturated cold, vast oppressive scale, ominous, 35mm grain
- id: iii13_grief_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man profound grief that has no funeral, losing her twice, the person and then the world he believed was kind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow steel
  voiceover: true
  talking: false
- id: iii13_grief_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man profound grief that has no funeral, losing her twice, the person and then the world he believed was kind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow steel
  voiceover: true
  talking: false
- id: iii13_grief_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man profound grief that has no funeral, losing her twice, the person and then the world he believed was kind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow steel
  voiceover: true
  talking: false
- id: iii13_grief_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man profound grief that has no funeral, losing her twice, the person and then the world he believed was kind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow steel
  voiceover: true
  talking: false
- id: iii13_grief_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man profound grief that has no funeral, losing her twice, the person and then the world he believed was kind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow steel
  voiceover: true
  talking: false
- id: iii13_grief_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man profound grief that has no funeral, losing her twice, the person and then the world he believed was kind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hollow steel
  voiceover: true
  talking: false
- id: iii14_shivers_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man shivering in the cold, because ice does not shiver, proof the cold has not consumed him and he has survived the harvest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold with a faint warm ember
  voiceover: true
  talking: false
- id: iii14_shivers_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man shivering in the cold, because ice does not shiver, proof the cold has not consumed him and he has survived the harvest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold with a faint warm ember
  voiceover: true
  talking: false
- id: iii14_shivers_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man shivering in the cold, because ice does not shiver, proof the cold has not consumed him and he has survived the harvest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold with a faint warm ember
  voiceover: true
  talking: false
- id: iii14_shivers_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man shivering in the cold, because ice does not shiver, proof the cold has not consumed him and he has survived the harvest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold with a faint warm ember
  voiceover: true
  talking: false
- id: iii14_shivers_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man shivering in the cold, because ice does not shiver, proof the cold has not consumed him and he has survived the harvest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold with a faint warm ember
  voiceover: true
  talking: false
- id: iii14_shivers_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man shivering in the cold, because ice does not shiver, proof the cold has not consumed him and he has survived the harvest, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cold with a faint warm ember
  voiceover: true
  talking: false
- id: iv01_why_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man of forty-seven, battered by society and sentenced to aloneness, asking why he opened his chest so completely to a stranger, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim reflective, dawn hint
  voiceover: true
  talking: false
- id: iv01_why_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man of forty-seven, battered by society and sentenced to aloneness, asking why he opened his chest so completely to a stranger, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim reflective, dawn hint
  voiceover: true
  talking: false
- id: iv01_why_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man of forty-seven, battered by society and sentenced to aloneness, asking why he opened his chest so completely to a stranger, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim reflective, dawn hint
  voiceover: true
  talking: false
- id: iv01_why_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man of forty-seven, battered by society and sentenced to aloneness, asking why he opened his chest so completely to a stranger, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim reflective, dawn hint
  voiceover: true
  talking: false
- id: iv01_why_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man of forty-seven, battered by society and sentenced to aloneness, asking why he opened his chest so completely to a stranger, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim reflective, dawn hint
  voiceover: true
  talking: false
- id: iv01_why_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man of forty-seven, battered by society and sentenced to aloneness, asking why he opened his chest so completely to a stranger, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: dim reflective, dawn hint
  voiceover: true
  talking: false
- id: iv02_dawn_desk_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward dawn breaking after the night of the dogfight, the man sitting at his writing desk seeing the architecture of his own soul, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool dawn, warm lamp
  voiceover: true
  talking: false
- id: iv02_dawn_desk_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across dawn breaking after the night of the dogfight, the man sitting at his writing desk seeing the architecture of his own soul, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool dawn, warm lamp
  voiceover: true
  talking: false
- id: iv02_dawn_desk_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on dawn breaking after the night of the dogfight, the man sitting at his writing desk seeing the architecture of his own soul, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool dawn, warm lamp
  voiceover: true
  talking: false
- id: iv02_dawn_desk_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over dawn breaking after the night of the dogfight, the man sitting at his writing desk seeing the architecture of his own soul, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool dawn, warm lamp
  voiceover: true
  talking: false
- id: iv02_dawn_desk_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on dawn breaking after the night of the dogfight, the man sitting at his writing desk seeing the architecture of his own soul, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool dawn, warm lamp
  voiceover: true
  talking: false
- id: iv02_dawn_desk_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward dawn breaking after the night of the dogfight, the man sitting at his writing desk seeing the architecture of his own soul, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: cool dawn, warm lamp
  voiceover: true
  talking: false
- id: iv03_machine_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the man imagining the world as a machine that demands perfection, wanting men to be cold lethal precise Terminators, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hard industrial grey
  voiceover: true
  talking: false
- id: iv03_machine_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the man imagining the world as a machine that demands perfection, wanting men to be cold lethal precise Terminators, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hard industrial grey
  voiceover: true
  talking: false
- id: iv03_machine_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the man imagining the world as a machine that demands perfection, wanting men to be cold lethal precise Terminators, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hard industrial grey
  voiceover: true
  talking: false
- id: iv03_machine_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the man imagining the world as a machine that demands perfection, wanting men to be cold lethal precise Terminators, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hard industrial grey
  voiceover: true
  talking: false
- id: iv03_machine_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the man imagining the world as a machine that demands perfection, wanting men to be cold lethal precise Terminators, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hard industrial grey
  voiceover: true
  talking: false
- id: iv03_machine_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the man imagining the world as a machine that demands perfection, wanting men to be cold lethal precise Terminators, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: hard industrial grey
  voiceover: true
  talking: false
- id: iv04_markets_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the cold precision the markets and the races reward, instruments that never hesitate and never miss, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel, clinical
  voiceover: true
  talking: false
- id: iv04_markets_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the cold precision the markets and the races reward, instruments that never hesitate and never miss, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel, clinical
  voiceover: true
  talking: false
- id: iv04_markets_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the cold precision the markets and the races reward, instruments that never hesitate and never miss, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel, clinical
  voiceover: true
  talking: false
- id: iv04_markets_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the cold precision the markets and the races reward, instruments that never hesitate and never miss, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel, clinical
  voiceover: true
  talking: false
- id: iv04_markets_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the cold precision the markets and the races reward, instruments that never hesitate and never miss, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: steel, clinical
  voiceover: true
  talking: false
- id: iv05_no_memoir_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the truth that a Terminator writes no memoir, never weeps in the dark and cannot feel red rose petals dancing in the wind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grey with a single red petal
  voiceover: true
  talking: false
- id: iv05_no_memoir_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the truth that a Terminator writes no memoir, never weeps in the dark and cannot feel red rose petals dancing in the wind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grey with a single red petal
  voiceover: true
  talking: false
- id: iv05_no_memoir_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the truth that a Terminator writes no memoir, never weeps in the dark and cannot feel red rose petals dancing in the wind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grey with a single red petal
  voiceover: true
  talking: false
- id: iv05_no_memoir_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the truth that a Terminator writes no memoir, never weeps in the dark and cannot feel red rose petals dancing in the wind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grey with a single red petal
  voiceover: true
  talking: false
- id: iv05_no_memoir_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the truth that a Terminator writes no memoir, never weeps in the dark and cannot feel red rose petals dancing in the wind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grey with a single red petal
  voiceover: true
  talking: false
- id: iv05_no_memoir_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the truth that a Terminator writes no memoir, never weeps in the dark and cannot feel red rose petals dancing in the wind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grey with a single red petal
  voiceover: true
  talking: false
- id: iv06_glitch_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward what turned him from a cold machine into a human being, a glitch, a bug of vulnerability, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm flaw in cold
  voiceover: true
  talking: false
- id: iv06_glitch_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across what turned him from a cold machine into a human being, a glitch, a bug of vulnerability, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm flaw in cold
  voiceover: true
  talking: false
- id: iv06_glitch_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on what turned him from a cold machine into a human being, a glitch, a bug of vulnerability, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm flaw in cold
  voiceover: true
  talking: false
- id: iv06_glitch_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over what turned him from a cold machine into a human being, a glitch, a bug of vulnerability, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm flaw in cold
  voiceover: true
  talking: false
- id: iv06_glitch_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on what turned him from a cold machine into a human being, a glitch, a bug of vulnerability, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm flaw in cold
  voiceover: true
  talking: false
- id: iv06_glitch_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward what turned him from a cold machine into a human being, a glitch, a bug of vulnerability, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm flaw in cold
  voiceover: true
  talking: false
- id: iv07_bug_body_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the hesitation at his desk, the tightening throat, the nervous system feeling every micro-tone of grief, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: intimate warm-cold
  voiceover: true
  talking: false
- id: iv07_bug_body_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the hesitation at his desk, the tightening throat, the nervous system feeling every micro-tone of grief, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: intimate warm-cold
  voiceover: true
  talking: false
- id: iv07_bug_body_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the hesitation at his desk, the tightening throat, the nervous system feeling every micro-tone of grief, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: intimate warm-cold
  voiceover: true
  talking: false
- id: iv07_bug_body_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the hesitation at his desk, the tightening throat, the nervous system feeling every micro-tone of grief, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: intimate warm-cold
  voiceover: true
  talking: false
- id: iv07_bug_body_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the hesitation at his desk, the tightening throat, the nervous system feeling every micro-tone of grief, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: intimate warm-cold
  voiceover: true
  talking: false
- id: iv08_love_bug_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the man confession, that he loves that bug and fell for it at first sight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender warm
  voiceover: true
  talking: false
- id: iv08_love_bug_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the man confession, that he loves that bug and fell for it at first sight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender warm
  voiceover: true
  talking: false
- id: iv08_love_bug_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the man confession, that he loves that bug and fell for it at first sight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender warm
  voiceover: true
  talking: false
- id: iv08_love_bug_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the man confession, that he loves that bug and fell for it at first sight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender warm
  voiceover: true
  talking: false
- id: iv08_love_bug_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the man confession, that he loves that bug and fell for it at first sight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender warm
  voiceover: true
  talking: false
- id: iv08_love_bug_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the man confession, that he loves that bug and fell for it at first sight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender warm
  voiceover: true
  talking: false
- id: iv08_love_bug_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: prelude
  action: a low-angle shot slowly craning up at the man confession, that he loves that bug and fell for it at first sight, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender warm
  voiceover: true
  talking: false
- id: iv09_her_bug_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the recognition that she too was a bug in the machine, an imperfect beautiful human who felt too much and tried to break pedestals, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft human warmth
  voiceover: true
  talking: false
- id: iv09_her_bug_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the recognition that she too was a bug in the machine, an imperfect beautiful human who felt too much and tried to break pedestals, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft human warmth
  voiceover: true
  talking: false
- id: iv09_her_bug_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the recognition that she too was a bug in the machine, an imperfect beautiful human who felt too much and tried to break pedestals, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft human warmth
  voiceover: true
  talking: false
- id: iv09_her_bug_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the recognition that she too was a bug in the machine, an imperfect beautiful human who felt too much and tried to break pedestals, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft human warmth
  voiceover: true
  talking: false
- id: iv09_her_bug_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the recognition that she too was a bug in the machine, an imperfect beautiful human who felt too much and tried to break pedestals, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft human warmth
  voiceover: true
  talking: false
- id: iv09_her_bug_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the recognition that she too was a bug in the machine, an imperfect beautiful human who felt too much and tried to break pedestals, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft human warmth
  voiceover: true
  talking: false
- id: iv10_wrong_second_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the tragedy of two glitches meeting at the wrong second, she spoke while he packaged, he looked at the screen while she offered her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: two clocks, bittersweet
  voiceover: true
  talking: false
- id: iv10_wrong_second_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the tragedy of two glitches meeting at the wrong second, she spoke while he packaged, he looked at the screen while she offered her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: two clocks, bittersweet
  voiceover: true
  talking: false
- id: iv10_wrong_second_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the tragedy of two glitches meeting at the wrong second, she spoke while he packaged, he looked at the screen while she offered her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: two clocks, bittersweet
  voiceover: true
  talking: false
- id: iv10_wrong_second_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the tragedy of two glitches meeting at the wrong second, she spoke while he packaged, he looked at the screen while she offered her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: two clocks, bittersweet
  voiceover: true
  talking: false
- id: iv10_wrong_second_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the tragedy of two glitches meeting at the wrong second, she spoke while he packaged, he looked at the screen while she offered her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: two clocks, bittersweet
  voiceover: true
  talking: false
- id: iv10_wrong_second_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the tragedy of two glitches meeting at the wrong second, she spoke while he packaged, he looked at the screen while she offered her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: two clocks, bittersweet
  voiceover: true
  talking: false
- id: iv10_wrong_second_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: search
  action: a low-angle shot slowly craning up at the tragedy of two glitches meeting at the wrong second, she spoke while he packaged, he looked at the screen while she offered her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: two clocks, bittersweet
  voiceover: true
  talking: false
- id: v01_truce_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the forty-seven-year war inside his skull ending, the man stopping the brakes and the accelerator and signing a peace with his own mind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: calm, warming
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v01_truce_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the forty-seven-year war inside his skull ending, the man stopping the brakes and the accelerator and signing a peace with his own mind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: calm, warming
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v01_truce_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the forty-seven-year war inside his skull ending, the man stopping the brakes and the accelerator and signing a peace with his own mind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: calm, warming
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v01_truce_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the forty-seven-year war inside his skull ending, the man stopping the brakes and the accelerator and signing a peace with his own mind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: calm, warming
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v01_truce_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the forty-seven-year war inside his skull ending, the man stopping the brakes and the accelerator and signing a peace with his own mind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: calm, warming
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v01_truce_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the forty-seven-year war inside his skull ending, the man stopping the brakes and the accelerator and signing a peace with his own mind, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: calm, warming
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v02_open_cage_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward each evening the man opening the cage and letting the wild brilliant eagle of his intellect fly free to the highest altitude, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: neon glow, open sky
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v02_open_cage_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across each evening the man opening the cage and letting the wild brilliant eagle of his intellect fly free to the highest altitude, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: neon glow, open sky
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v02_open_cage_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on each evening the man opening the cage and letting the wild brilliant eagle of his intellect fly free to the highest altitude, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: neon glow, open sky
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v02_open_cage_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over each evening the man opening the cage and letting the wild brilliant eagle of his intellect fly free to the highest altitude, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: neon glow, open sky
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v02_open_cage_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on each evening the man opening the cage and letting the wild brilliant eagle of his intellect fly free to the highest altitude, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: neon glow, open sky
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v02_open_cage_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward each evening the man opening the cage and letting the wild brilliant eagle of his intellect fly free to the highest altitude, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: neon glow, open sky
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v02_open_cage_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: prelude
  action: a low-angle shot slowly craning up at each evening the man opening the cage and letting the wild brilliant eagle of his intellect fly free to the highest altitude, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: neon glow, open sky
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v03_dimensions_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the eagle of his mind gliding through dimensions, sometimes a poet, sometimes a philosopher, sometimes a conductor, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: iridescent twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v03_dimensions_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the eagle of his mind gliding through dimensions, sometimes a poet, sometimes a philosopher, sometimes a conductor, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: iridescent twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v03_dimensions_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the eagle of his mind gliding through dimensions, sometimes a poet, sometimes a philosopher, sometimes a conductor, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: iridescent twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v03_dimensions_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the eagle of his mind gliding through dimensions, sometimes a poet, sometimes a philosopher, sometimes a conductor, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: iridescent twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v03_dimensions_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the eagle of his mind gliding through dimensions, sometimes a poet, sometimes a philosopher, sometimes a conductor, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: iridescent twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v03_dimensions_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the eagle of his mind gliding through dimensions, sometimes a poet, sometimes a philosopher, sometimes a conductor, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: iridescent twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v04_returns_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward at the appointed hour, without a struggle, the eagle returning home to the ordinary man, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm dusk, settling
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v04_returns_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across at the appointed hour, without a struggle, the eagle returning home to the ordinary man, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm dusk, settling
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v04_returns_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on at the appointed hour, without a struggle, the eagle returning home to the ordinary man, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm dusk, settling
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v04_returns_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over at the appointed hour, without a struggle, the eagle returning home to the ordinary man, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm dusk, settling
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v04_returns_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on at the appointed hour, without a struggle, the eagle returning home to the ordinary man, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm dusk, settling
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v04_returns_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward at the appointed hour, without a struggle, the eagle returning home to the ordinary man, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm dusk, settling
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v05_beholder_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the man staying grounded, an ordinary man with an ordinary body, becoming the Beholder, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grounded warm neon
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v05_beholder_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the man staying grounded, an ordinary man with an ordinary body, becoming the Beholder, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grounded warm neon
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v05_beholder_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the man staying grounded, an ordinary man with an ordinary body, becoming the Beholder, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grounded warm neon
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v05_beholder_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the man staying grounded, an ordinary man with an ordinary body, becoming the Beholder, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grounded warm neon
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v05_beholder_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the man staying grounded, an ordinary man with an ordinary body, becoming the Beholder, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grounded warm neon
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v05_beholder_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the man staying grounded, an ordinary man with an ordinary body, becoming the Beholder, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: grounded warm neon
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v06_her_shoes_01
  secs: 5
  who:
  - jack
  - julie
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward the supreme act of love, the man stepping out of his own shoes and into hers and looking back at himself through her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender neon, dual
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v06_her_shoes_02
  secs: 4
  who:
  - jack
  - julie
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across the supreme act of love, the man stepping out of his own shoes and into hers and looking back at himself through her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender neon, dual
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v06_her_shoes_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on the supreme act of love, the man stepping out of his own shoes and into hers and looking back at himself through her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender neon, dual
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v06_her_shoes_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over the supreme act of love, the man stepping out of his own shoes and into hers and looking back at himself through her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender neon, dual
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v06_her_shoes_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on the supreme act of love, the man stepping out of his own shoes and into hers and looking back at himself through her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender neon, dual
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v06_her_shoes_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward the supreme act of love, the man stepping out of his own shoes and into hers and looking back at himself through her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender neon, dual
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v06_her_shoes_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: search
  action: a low-angle shot slowly craning up at the supreme act of love, the man stepping out of his own shoes and into hers and looking back at himself through her eyes, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: tender neon, dual
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v07_her_eyes_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: search
  action: a wide shot slowly pushing in toward from her vantage across the room, finally seeing the silent man at the screen and feeling her courage, her vulnerability and the sting of her misunderstanding, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft reverse, warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v07_her_eyes_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: search
  action: a wide slow dolly gliding across from her vantage across the room, finally seeing the silent man at the screen and feeling her courage, her vulnerability and the sting of her misunderstanding, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft reverse, warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v07_her_eyes_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: search
  action: a medium shot slowly pushing in on from her vantage across the room, finally seeing the silent man at the screen and feeling her courage, her vulnerability and the sting of her misunderstanding, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft reverse, warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v07_her_eyes_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: search
  action: an intimate close-up slowly drifting over from her vantage across the room, finally seeing the silent man at the screen and feeling her courage, her vulnerability and the sting of her misunderstanding, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft reverse, warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v07_her_eyes_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: search
  action: an extreme close-up with a slow rack focus on from her vantage across the room, finally seeing the silent man at the screen and feeling her courage, her vulnerability and the sting of her misunderstanding, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft reverse, warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v07_her_eyes_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: search
  action: a slow over-the-shoulder move toward from her vantage across the room, finally seeing the silent man at the screen and feeling her courage, her vulnerability and the sting of her misunderstanding, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: soft reverse, warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v08_forgives_01
  secs: 5
  who:
  - julie
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the man loving her for attempting the crossing at all, becoming the woman to his own memory, standing on the dirt looking up and watching the creature fly free, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: forgiving glow
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v08_forgives_02
  secs: 4
  who:
  - julie
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the man loving her for attempting the crossing at all, becoming the woman to his own memory, standing on the dirt looking up and watching the creature fly free, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: forgiving glow
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v08_forgives_03
  secs: 4
  who:
  - julie
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the man loving her for attempting the crossing at all, becoming the woman to his own memory, standing on the dirt looking up and watching the creature fly free, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: forgiving glow
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v08_forgives_04
  secs: 4
  who:
  - julie
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the man loving her for attempting the crossing at all, becoming the woman to his own memory, standing on the dirt looking up and watching the creature fly free, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: forgiving glow
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v08_forgives_05
  secs: 4
  who:
  - julie
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the man loving her for attempting the crossing at all, becoming the woman to his own memory, standing on the dirt looking up and watching the creature fly free, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: forgiving glow
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v08_forgives_06
  secs: 4
  who:
  - julie
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the man loving her for attempting the crossing at all, becoming the woman to his own memory, standing on the dirt looking up and watching the creature fly free, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: forgiving glow
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v08_forgives_07
  secs: 4
  who:
  - julie
  camera: low-angle crane up
  song: prelude
  action: a low-angle shot slowly craning up at the man loving her for attempting the crossing at all, becoming the woman to his own memory, standing on the dirt looking up and watching the creature fly free, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: forgiving glow
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v09_twilight_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the harsh white midday sun fading and the freezing blue shadows softening as the warm analog neon twilight of memory descends over an open road, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm synthwave neon twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v09_twilight_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the harsh white midday sun fading and the freezing blue shadows softening as the warm analog neon twilight of memory descends over an open road, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm synthwave neon twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v09_twilight_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the harsh white midday sun fading and the freezing blue shadows softening as the warm analog neon twilight of memory descends over an open road, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm synthwave neon twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v09_twilight_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the harsh white midday sun fading and the freezing blue shadows softening as the warm analog neon twilight of memory descends over an open road, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm synthwave neon twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v09_twilight_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the harsh white midday sun fading and the freezing blue shadows softening as the warm analog neon twilight of memory descends over an open road, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm synthwave neon twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v09_twilight_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the harsh white midday sun fading and the freezing blue shadows softening as the warm analog neon twilight of memory descends over an open road, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm synthwave neon twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v09_twilight_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: prelude
  action: a low-angle shot slowly craning up at the harsh white midday sun fading and the freezing blue shadows softening as the warm analog neon twilight of memory descends over an open road, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: warm synthwave neon twilight
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v10_writes_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the man at his desk picking up his pen and beginning to write the saga of what he had seen, the framework the machine that turns the flood into art, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: lamp warm, neon window
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v10_writes_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the man at his desk picking up his pen and beginning to write the saga of what he had seen, the framework the machine that turns the flood into art, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: lamp warm, neon window
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v10_writes_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the man at his desk picking up his pen and beginning to write the saga of what he had seen, the framework the machine that turns the flood into art, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: lamp warm, neon window
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v10_writes_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the man at his desk picking up his pen and beginning to write the saga of what he had seen, the framework the machine that turns the flood into art, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: lamp warm, neon window
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v10_writes_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the man at his desk picking up his pen and beginning to write the saga of what he had seen, the framework the machine that turns the flood into art, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: lamp warm, neon window
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v10_writes_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the man at his desk picking up his pen and beginning to write the saga of what he had seen, the framework the machine that turns the flood into art, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: lamp warm, neon window
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v11_channel_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the ordinary man as a humble channel to whom the divine handed a gift, his character wall intact after forty-seven years of pure form, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: serene warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v11_channel_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the ordinary man as a humble channel to whom the divine handed a gift, his character wall intact after forty-seven years of pure form, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: serene warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v11_channel_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the ordinary man as a humble channel to whom the divine handed a gift, his character wall intact after forty-seven years of pure form, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: serene warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v11_channel_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the ordinary man as a humble channel to whom the divine handed a gift, his character wall intact after forty-seven years of pure form, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: serene warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v11_channel_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the ordinary man as a humble channel to whom the divine handed a gift, his character wall intact after forty-seven years of pure form, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: serene warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v11_channel_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the ordinary man as a humble channel to whom the divine handed a gift, his character wall intact after forty-seven years of pure form, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: serene warm
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v12_final_01
  secs: 5
  who:
  - jack
  camera: wide slow push-in
  song: prelude
  action: a wide shot slowly pushing in toward the man laying the pen down and whispering the final words to the empty room, received recorded returned, not owned not authored given back, peace, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: quiet neon fade
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v12_final_02
  secs: 4
  who:
  - jack
  camera: wide slow dolly
  song: prelude
  action: a wide slow dolly gliding across the man laying the pen down and whispering the final words to the empty room, received recorded returned, not owned not authored given back, peace, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: quiet neon fade
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v12_final_03
  secs: 4
  who:
  - jack
  camera: medium push-in
  song: prelude
  action: a medium shot slowly pushing in on the man laying the pen down and whispering the final words to the empty room, received recorded returned, not owned not authored given back, peace, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: quiet neon fade
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v12_final_04
  secs: 4
  who:
  - jack
  camera: close-up drift
  song: prelude
  action: an intimate close-up slowly drifting over the man laying the pen down and whispering the final words to the empty room, received recorded returned, not owned not authored given back, peace, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: quiet neon fade
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v12_final_05
  secs: 4
  who:
  - jack
  camera: extreme close-up rack
  song: prelude
  action: an extreme close-up with a slow rack focus on the man laying the pen down and whispering the final words to the empty room, received recorded returned, not owned not authored given back, peace, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: quiet neon fade
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v12_final_06
  secs: 4
  who:
  - jack
  camera: over-the-shoulder move
  song: prelude
  action: a slow over-the-shoulder move toward the man laying the pen down and whispering the final words to the empty room, received recorded returned, not owned not authored given back, peace, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: quiet neon fade
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
- id: v12_final_07
  secs: 4
  who:
  - jack
  camera: low-angle crane up
  song: prelude
  action: a low-angle shot slowly craning up at the man laying the pen down and whispering the final words to the empty room, received recorded returned, not owned not authored given back, peace, smooth steady camera movement, gentle natural motion, subtle drifting atmosphere and soft shifting light, stable coherent shapes, solid consistent forms, a living breathing cinematic shot, no freeze
  palette: quiet neon fade
  voiceover: true
  talking: false
  look: warm analog neon twilight, soft synth-wave glow, magenta and amber haze, gentle grain, nostalgic peace, 35mm
'''

def _build_shots_from_story(story_text, out_json):
    import yaml, json, os
    story = yaml.safe_load(story_text)
    base = {}
    if os.path.exists(out_json):
        try:
            base = json.load(open(out_json))
        except Exception:
            base = {}
    base_g = base.get("globals", {})
    base_cd = base.get("character_design", {})
    look = story.get("look", base_g.get("tone", ""))
    avoid = story.get("avoid", base_g.get("negative_prompt", ""))
    DEFAULT_RENDER = {"fps": 24, "width": 1280, "height": 720, "base_model": "Wan2.2-S2V", "quant": "Q8", "steps": 30, "cfg": 5.5, "sampler": "uni_pc", "scheduler": "beta", "interpolate_to_fps": 48, "upscale": 1.75, "rejection_samples": 3}
    g = dict(DEFAULT_RENDER)
    g.update({k: v for k, v in base_g.items() if k not in ("tone", "negative_prompt")})
    g.update(story.get("render", {}) or {})
    g["tone"] = look
    g["negative_prompt"] = avoid
    characters, character_design = {}, {}
    for name, c in (story.get("characters", {}) or {}).items():
        clook = c.get("look", "")
        characters[name] = {"refs": "refs/" + name, "tokens": clook, "anchor": c.get("anchor", name + ".png")}
        character_design[name] = {"summary": clook} if clook else (base_cd.get(name) or {"summary": ""})
    shots = []
    for s in story.get("shots", []):
        shot = {"id": s["id"], "dur": s.get("secs", 4), "chars": list(s.get("who", [])), "talking": bool(s.get("talking", False)), "vo": bool(s.get("voiceover", False)), "camera": s.get("camera", ""), "mode": s.get("mode", "A"), "song": s.get("song", ""), "prompt": s.get("action", ""), "palette": s.get("palette", ""), "negative": s.get("avoid", avoid), "tone": s.get("look", look), "movement": s.get("movement", s.get("song", "")), "line": s.get("line", s.get("say", s.get("dialogue", ""))), "vo_text": s.get("line", s.get("say", s.get("dialogue", "")))}
        if s.get("fx") is not None:
            shot["fx"] = s["fx"]
        if s.get("intensity") is not None:
            shot["intensity"] = s["intensity"]
        if s.get("takes") is not None:
            shot["rejection_samples"] = s["takes"]
        shots.append(shot)
    out = {"film": story.get("film", base.get("film", "UNTITLED")), "note": story.get("note", base.get("note", "")), "globals": g, "character_design": character_design, "characters": characters, "shots": shots}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return out

try:
    _story_json = os.path.join(KIT_ROOT, 'shots_confrontation.json')
    _built = _build_shots_from_story(STORY_YAML, _story_json)
    with open(os.path.join(KIT_ROOT, 'story.yaml'), 'w') as _f:
        _f.write(STORY_YAML)
    print("[OK] STORY built from notebook block: %d shots, %d characters -> shots_confrontation.json" % (len(_built['shots']), len(_built['characters'])))
    print("     characters:", ", ".join("%s<-%s" % (k, v.get('anchor')) for k, v in _built['characters'].items()))
    # --- 2-MINUTE CUT: keep only the opening whole scenes up to ~2 min ---
    _CUT_MAX_SECS = 125
    def _scene_of(_sid):
        return '_'.join(_sid.split('_')[:2])
    _scene_order = []
    for _s in _built['shots']:
        _sc = _scene_of(_s['id'])
        if not _scene_order or _scene_order[-1][0] != _sc:
            _scene_order.append((_sc, []))
        _scene_order[-1][1].append(_s)
    _kept = []; _tot = 0
    for _sc, _lst in _scene_order:
        _d = sum(int(x.get('dur', 4)) for x in _lst)
        if _kept and _tot + _d > _CUT_MAX_SECS:
            break
        _kept += _lst; _tot += _d
    _built['shots'] = _kept
    json.dump(_built, open(_story_json, 'w'), indent=1, ensure_ascii=False)
    print('[2min] Opening-act cut: %d shots, ~%ds (%d:%02d).' % (len(_kept), _tot, _tot // 60, _tot % 60))
    # --- AUTO-DERIVE real shot count -> re-clamp the chunk range so a chunk
    #     can NEVER exceed the actual story (overrides the provisional 435 above) ---
    _real_total = len(_built['shots'])
    os.environ['KIT_SHOT_START'] = '1'
    os.environ['KIT_SHOT_END']   = str(_real_total)
    os.environ['KIT_CHUNK_LABEL']= 'Beholder_FULL'
    print('[auto] AUTO-DERIVED from story: total=%d shots -> HANDS-OFF full-range render (resume-on-rerun, budget=%sh).' % (_real_total, os.environ.get('KIT_MAX_HOURS','11')))
except Exception as _e:
    print("[WARN] STORY build failed (%s); using the kit shots_confrontation.json as-is." % _e)

# --- SPEED: single take per shot (disable rejection sampling) for the short cut ---
import yaml as _yaml_rs
_rs_cfgp = os.path.join(KIT_ROOT, 'config_confrontation.yaml')
try:
    _rs_cfg = _yaml_rs.safe_load(open(_rs_cfgp))
    _rs_cfg['rejection_samples'] = 1
    _rs_ap = _rs_cfg.get('active_profile')
    if _rs_ap and isinstance(_rs_cfg.get('profiles'), dict) and _rs_ap in _rs_cfg['profiles']:
        _rs_cfg['profiles'][_rs_ap]['rejection_samples'] = 1
    _yaml_rs.safe_dump(_rs_cfg, open(_rs_cfgp, 'w'), sort_keys=False)
    print('[speed] rejection_samples=1 (single take/shot); active_profile=%s' % _rs_ap)
except Exception as _rse:
    print('[speed][WARN] could not set rejection_samples=1: %s' % _rse)
# ===== MERGE: PROOF-FIRST GATE (oneclick ZERO-BURN playbook) =====
# Default renders ONLY the first shots (the "prelude" proof) so env/models/
# identity/quality are validated cheaply before committing full-film GPU spend.
# Set GODSHAND_MODE=full to render every shot. Resume-on-rerun still applies.
import json as _mj
GODSHAND_MODE = os.environ.get("GODSHAND_MODE", "proof").lower()
try:
    _mtotal = len(_mj.load(open(os.path.join(KIT_ROOT, "shots_confrontation.json")))["shots"])
except Exception:
    _mtotal = int(os.environ.get("KIT_SHOT_END", "4") or "4")
if GODSHAND_MODE == "proof":
    _pn = min(int(os.environ.get("GODSHAND_PROOF_SHOTS", "4")), _mtotal)
    os.environ["KIT_SHOT_START"] = "1"
    os.environ["KIT_SHOT_END"]   = str(_pn)
    os.environ["KIT_CHUNK_LABEL"] = "PROOF_prelude"
    print("[merge] PROOF MODE -> first %d shot(s) only (the 'prelude'). Set GODSHAND_MODE=full for all %d." % (_pn, _mtotal))
else:
    os.environ["KIT_SHOT_START"] = "1"
    os.environ["KIT_SHOT_END"]   = str(_mtotal)
    os.environ["KIT_CHUNK_LABEL"] = "FULL"
    print("[merge] FULL MODE -> all %d shots." % _mtotal)


# ==============================================================================
# STEP 3: INSTALL / UPDATE COMFYUI & DEPENDENCIES
# ==============================================================================
COMFY = Path('/kaggle/working/ComfyUI')

def sh(cmd, cwd=None):
    print(f'$ {cmd}')
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        raise RuntimeError(f'Command failed: {cmd}')

if not COMFY.exists():
    sh('git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git', cwd='/kaggle/working')
else:
    sh('git pull', cwd=str(COMFY))

sh('pip install -q -r requirements.txt', cwd=str(COMFY))
cn = COMFY / 'custom_nodes'; cn.mkdir(exist_ok=True)
for name, url in [
    ('ComfyUI-GGUF', 'https://github.com/city96/ComfyUI-GGUF.git'),
    ('ComfyUI-VideoHelperSuite', 'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git'),
]:
    dst = cn / name
    if not dst.exists():
        sh(f'git clone --depth 1 {url}', cwd=str(cn))
    else:
        sh('git pull', cwd=str(dst))
sh('pip install -q gguf sentencepiece protobuf einops')
print('[install] ComfyUI and custom nodes ready.')

# ==============================================================================
# STEP 4: DOWNLOAD LTX-VIDEO 13B 0.9.7 DISTILLED CHECKPOINT & T5-XXL TEXT ENCODER
# ==============================================================================
from huggingface_hub import hf_hub_download, list_repo_files

# Store large weights on the roomier /kaggle/temp disk to dodge the ~20GB /kaggle/working cap.
os.environ['HF_HUB_DISABLE_XET'] = '1'  # plain download; avoids Xet reconstruction scratch space
_MODELS_REAL = Path('/kaggle/temp/models'); _MODELS_REAL.mkdir(parents=True, exist_ok=True)
MODELS = COMFY / 'models'
if not MODELS.is_symlink():
    if MODELS.exists():
        shutil.rmtree(MODELS, ignore_errors=True)
    MODELS.symlink_to(_MODELS_REAL)
for sub in ['unet', 'vae', 'text_encoders', 'clip_vision', 'diffusion_models', 'checkpoints', 'upscale_models', 'loras']:
    (MODELS / sub).mkdir(parents=True, exist_ok=True)

# --- LTX-Video 13B 0.9.7 DISTILLED checkpoint (all-in-one: video transformer + VAE) ---
LTXV_REPO = 'Lightricks/LTX-Video'
LTXV_CANDIDATES = ['ltxv-13b-0.9.7-distilled-fp8.safetensors', 'ltxv-13b-0.9.7-dev-fp8.safetensors', 'ltxv-13b-0.9.7-dev.safetensors', 'ltxv-13b-0.9.7-distilled.safetensors', 'ltx-video-2b-v0.9.5.safetensors']
repo_files = set(list_repo_files(LTXV_REPO, token=os.environ.get('HF_TOKEN')))
CKPT_PICK = next((f for f in LTXV_CANDIDATES if f in repo_files), None)
assert CKPT_PICK, f'ERROR: no known LTX-Video checkpoint in {LTXV_REPO}; .safetensors seen: {sorted(f for f in repo_files if f.endswith(".safetensors"))[:8]}'
local = hf_hub_download(LTXV_REPO, filename=CKPT_PICK, local_dir=str(MODELS / 'checkpoints'), token=os.environ.get('HF_TOKEN'))
flat = MODELS / 'checkpoints' / Path(CKPT_PICK).name
if not flat.exists():
    shutil.move(local, flat)
CKPT_NAME = flat.name
IS_13B = '13b' in CKPT_NAME.lower(); IS_DISTILLED = 'distilled' in CKPT_NAME.lower()
if IS_DISTILLED:
    os.environ['KIT_MOTION_STEPS'] = '8'     # LTXV distilled renders in ~8 steps
    os.environ['KIT_LTXV_CFG'] = '1.0'       # distilled model -> CFG 1.0 (no guidance)
else:
    os.environ['KIT_MOTION_STEPS'] = '30'    # dev 13B: full sampling for stability (kills opening morph)
    os.environ['KIT_LTXV_CFG'] = '3.0'       # dev model uses real CFG guidance
print('[model] loaded %s | 13B=%s distilled=%s steps=%s cfg=%s' % (CKPT_NAME, IS_13B, IS_DISTILLED, os.environ['KIT_MOTION_STEPS'], os.environ['KIT_LTXV_CFG']))

# --- T5-XXL text encoder required by LTX-Video ---
T5_REPO = 'comfyanonymous/flux_text_encoders'
_T5_CANDIDATES = ['t5xxl_fp8_e4m3fn.safetensors', 't5xxl_fp16.safetensors']
T5_NAME = next((f for f in _T5_CANDIDATES if f in set(list_repo_files(T5_REPO, token=os.environ.get('HF_TOKEN')))), 't5xxl_fp16.safetensors')
local = hf_hub_download(T5_REPO, filename=T5_NAME, local_dir=str(MODELS), token=os.environ.get('HF_TOKEN'))
flat_t5 = MODELS / 'text_encoders' / T5_NAME
if not flat_t5.exists():
    shutil.move(local, flat_t5)

# --- Spatial upscaler model (RealESRGAN) for near-HD crispness; guarded so failure never breaks the render ---
UPSCALE_MODEL_NAME = None
_UP_CANDIDATES = [('ai-forever/Real-ESRGAN', 'RealESRGAN_x2.pth'), ('ai-forever/Real-ESRGAN', 'RealESRGAN_x4.pth')]
for _rp, _fn in _UP_CANDIDATES:
    try:
        _up_local = hf_hub_download(_rp, filename=_fn, local_dir=str(MODELS / 'upscale_models'), token=os.environ.get('HF_TOKEN'))
        _up_flat = MODELS / 'upscale_models' / Path(_fn).name
        if not _up_flat.exists(): shutil.move(_up_local, _up_flat)
        UPSCALE_MODEL_NAME = _up_flat.name
        print('[download] spatial upscaler ready: %s (from %s)' % (UPSCALE_MODEL_NAME, _rp)); break
    except Exception as _ue:
        print('[download] upscaler candidate %s/%s failed (%s)' % (_rp, _fn, _ue))
if not UPSCALE_MODEL_NAME:
    print('[download] no upscaler model available; rendering at native res.')
open('/kaggle/working/.gguf_name', 'w').write(CKPT_NAME)
for _c in MODELS.rglob('.cache'):
    shutil.rmtree(_c, ignore_errors=True)
shutil.rmtree(MODELS / 'split_files', ignore_errors=True)
print(f'[download] Assets ready. LTX-Video ckpt: {CKPT_NAME} | T5: {T5_NAME}')
# === WAN2.2 14B I2V ENGINE DOWNLOADS (GGUF MoE experts + umt5 + wan2.1 vae + 4-step loras) ===
os.environ.setdefault("KIT_ENGINE", "wan")
if os.environ.get("KIT_ENGINE") == "wan":
    print("\n--- 4w. FETCHING WAN2.2 14B I2V (GGUF, MoE) ASSETS ---")
    try:
        (MODELS / "unet").mkdir(parents=True, exist_ok=True)
        (MODELS / "loras").mkdir(parents=True, exist_ok=True)
        _tok = os.environ.get("HF_TOKEN")
        WAN_GGUF_REPO = "QuantStack/Wan2.2-I2V-A14B-GGUF"
        WAN_COMFY_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
        _gfiles = list_repo_files(WAN_GGUF_REPO, token=_tok)
        _QORDER = ["q4_k_m", "q4_k_s", "q5_k_m", "q3_k_m", "q4_0", "q8_0"]
        def _pick_gguf(files, tag):
            cand = [f for f in files if f.lower().endswith(".gguf") and tag in f.lower()]
            for q in _QORDER:
                for f in cand:
                    if q in f.lower():
                        return f
            return cand[0] if cand else None
        _hi = _pick_gguf(_gfiles, "high"); _lo = _pick_gguf(_gfiles, "low")
        assert _hi and _lo, "no high/low GGUF experts in %s" % WAN_GGUF_REPO
        def _dl(repo, fn, sub):
            _lp = hf_hub_download(repo, filename=fn, local_dir=str(MODELS / sub), token=_tok)
            _flat = MODELS / sub / Path(fn).name
            if not _flat.exists():
                shutil.move(_lp, _flat)
            return _flat.name
        WAN_HIGH = _dl(WAN_GGUF_REPO, _hi, "unet")
        WAN_LOW = _dl(WAN_GGUF_REPO, _lo, "unet")
        _cfiles = list_repo_files(WAN_COMFY_REPO, token=_tok)
        def _find(files, *needles):
            for f in files:
                fl = f.lower()
                if all(n in fl for n in needles):
                    return f
            return None
        _umt5 = _find(_cfiles, "umt5", "fp8") or _find(_cfiles, "umt5")
        _vae = _find(_cfiles, "wan_2.1_vae") or _find(_cfiles, "vae", "2.1")
        assert _umt5 and _vae, "umt5/vae not found in %s" % WAN_COMFY_REPO
        WAN_UMT5 = _dl(WAN_COMFY_REPO, _umt5, "text_encoders")
        WAN_VAE = _dl(WAN_COMFY_REPO, _vae, "vae")
        WAN_LORA_HIGH = WAN_LORA_LOW = ""
        _lhi = _find(_cfiles, "i2v", "lightx2v", "high") or _find(_cfiles, "i2v", "4step", "high") or _find(_cfiles, "i2v", "4steps", "high")
        _llo = _find(_cfiles, "i2v", "lightx2v", "low") or _find(_cfiles, "i2v", "4step", "low") or _find(_cfiles, "i2v", "4steps", "low")
        if _lhi and _llo:
            try:
                WAN_LORA_HIGH = _dl(WAN_COMFY_REPO, _lhi, "loras")
                WAN_LORA_LOW = _dl(WAN_COMFY_REPO, _llo, "loras")
            except Exception as _le:
                WAN_LORA_HIGH = WAN_LORA_LOW = ""
                print("[wan] 4-step LoRA download failed (%s); will run at higher steps." % _le)
        os.environ["WAN_HIGH"] = WAN_HIGH; os.environ["WAN_LOW"] = WAN_LOW
        os.environ["WAN_UMT5"] = WAN_UMT5; os.environ["WAN_VAE"] = WAN_VAE
        os.environ["WAN_LORA_HIGH"] = WAN_LORA_HIGH; os.environ["WAN_LORA_LOW"] = WAN_LORA_LOW
        if WAN_LORA_HIGH and WAN_LORA_LOW:
            os.environ.setdefault("KIT_WAN_STEPS", "4"); os.environ.setdefault("KIT_WAN_CFG", "1.0")
        else:
            os.environ.setdefault("KIT_WAN_STEPS", "20"); os.environ.setdefault("KIT_WAN_CFG", "3.5")
        os.environ["KIT_ENGINE"] = "wan"
        for _c in MODELS.rglob(".cache"):
            shutil.rmtree(_c, ignore_errors=True)
        print("[wan] engine READY | high=%s low=%s umt5=%s vae=%s loraH=%s loraL=%s steps=%s cfg=%s" % (WAN_HIGH, WAN_LOW, WAN_UMT5, WAN_VAE, WAN_LORA_HIGH or "-", WAN_LORA_LOW or "-", os.environ["KIT_WAN_STEPS"], os.environ["KIT_WAN_CFG"]))
    except Exception as _wde:
        os.environ["KIT_ENGINE"] = "ltxv"
        print("[wan] setup FAILED (%s); reverting to LTXV engine (already downloaded)." % _wde)

# ==============================================================================
# STEP 5: PATCH CONFIG + INGEST ANCHORS FROM ATTACHED DATASET
# ==============================================================================
config_path = os.path.join(KIT_ROOT, 'config_confrontation.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_data = f.read()
    cfg_data = re.sub(r'^(\s*width:\s*)\d+',  r'\g<1>768', cfg_data, count=0, flags=re.M)
    cfg_data = re.sub(r'^(\s*height:\s*)\d+', r'\g<1>512', cfg_data, count=0, flags=re.M)
    cfg_data = re.sub(r'^(\s*steps:\s*)\d+',  r'\g<1>8',    cfg_data, count=1, flags=re.M)
    cfg_data = re.sub(r'^(\s*cfg:\s*)[\d.]+', r'\g<1>1.0', cfg_data, count=0, flags=re.M)
    cfg_data = re.sub(r'^(\s*rejection_samples:\s*)\d+', r'\g<1>2', cfg_data, count=0, flags=re.M)
    cfg_data = re.sub(r'^(\s*identity_retry:\s*)\d+', r'\g<1>0', cfg_data, count=0, flags=re.M)
    cfg_data = re.sub(r'^(\s*upscale:\s*)[\d.]+', r'\g<1>1.0', cfg_data, count=0, flags=re.M)  # graph upscaler handles this now
    print("[config] takes patched: rejection_samples=2, identity_retry=0 (2 takes/shot, pick best).")
    cfg_data = re.sub(r'^(active_profile:\s*).*$', r'\g<1>kaggle_2xt4', cfg_data, count=1, flags=re.M)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(cfg_data)
    print("[config] profile patched for VRAM safety.")

print("\n--- 5b. INGESTING CHARACTER ANCHORS FROM ATTACHED DATASET ---")
comfy_input_dir = COMFY / 'input'; comfy_input_dir.mkdir(parents=True, exist_ok=True)

# Accept hyphen OR underscore naming for each anchor.
_MAN_NAMES = ('man_anchor.png', 'man-anchor.png')
_WOMAN_NAMES = ('woman_anchor.png', 'woman-anchor.png')

# Gather EVERY dir under /kaggle/input that has both a man and a woman anchor.
_candidates = []
for root, dirs, files in os.walk('/kaggle/input'):
    _m = next((n for n in _MAN_NAMES if n in files), None)
    _w = next((n for n in _WOMAN_NAMES if n in files), None)
    if _m and _w:
        _candidates.append((root, _m, _w))

# Prefer the user's real dataset ('Saga Custom Characters') over the bundled
# GODSHAND kit placeholders (which sort earlier in os.walk and caused mush).
def _score(path):
    p = path.lower(); s = 0
    if any(k in p for k in ('saga', 'custom', 'character')): s += 2
    if 'godshand' in p: s -= 2
    return s
_candidates.sort(key=lambda c: _score(c[0]), reverse=True)

# ===== MERGE: identity source -- auto-forge (default) vs real anchors =====
# GODSHAND_FORGE=auto (default): INVENT jack/julie via SDXL text-forge; no
#   dataset photos required (the SDXL checkpoint is fetched before the forge
#   stage further below).
# GODSHAND_FORGE=anchors: original behavior -- require + bind the Saga Custom
#   Characters photo anchors, and text-forge is skipped for bound characters.
GODSHAND_FORGE = os.environ.get("GODSHAND_FORGE", "auto").lower()
_char_anchor = {}
anchor_src_dir = None
if GODSHAND_FORGE == "anchors":
    if not _candidates:
        print("\u274c ERROR: Could not locate man/woman anchors (underscore or hyphen) in /kaggle/input/!")
        print("\U0001f449 Verify the 'Saga Custom Characters' dataset is attached in the right-hand panel.")
        raise FileNotFoundError("Missing anchor images in /kaggle/input.")
    anchor_src_dir, _man_file, _woman_file = _candidates[0]
    print(f"\u2713 Selected character dataset: {anchor_src_dir}  (man='{_man_file}', woman='{_woman_file}')")
    if len(_candidates) > 1:
        print(f"  ({len(_candidates)} candidate dirs found; picked highest-priority. Others ignored: {[c[0] for c in _candidates[1:]]})")
    for _src_name, _dest_name in ((_man_file, 'man_anchor.png'), (_woman_file, 'woman_anchor.png')):
        _src_p = os.path.join(anchor_src_dir, _src_name)
        shutil.copy2(_src_p, os.path.join(KIT_ROOT, _dest_name))
        shutil.copy2(_src_p, os.path.join(comfy_input_dir, _dest_name))
        try:
            _im = Image.open(_src_p).convert('RGB')
            _arr = np.asarray(_im, dtype=np.float32)
            _std = float(_arr.std())
            _flag = "\u26a0\ufe0f LOOKS FLAT / POSSIBLE PLACEHOLDER" if _std < 8.0 else "ok (real detail)"
            print(f"   {_dest_name}: {_im.size[0]}x{_im.size[1]} px, pixel std={_std:.1f} -> {_flag}")
        except Exception as _e:
            print(f"   {_dest_name}: copied, but validation failed: {_e}")
    print("\u2705 Ingested REAL character anchors into project root and ComfyUI/input/.")
    _char_anchor = {os.environ.get("KIT_MAN_CHAR", "jack"): "man_anchor.png",
                    os.environ.get("KIT_WOMAN_CHAR", "julie"): "woman_anchor.png"}
    for _char, _afile in _char_anchor.items():
        _cdir = os.path.join(KIT_ROOT, "refs", _char)
        if os.path.isdir(_cdir):
            shutil.rmtree(_cdir, ignore_errors=True)  # wipe any stale text-forged stills
        _face = os.path.join(_cdir, "face"); os.makedirs(_face, exist_ok=True)
        shutil.copy2(os.path.join(KIT_ROOT, _afile), os.path.join(_face, _afile))
        print(f"   refs/{_char}/face/{_afile} <- real anchor (text-forge will be skipped for {_char})")
    print("\u2705 Character anchors BOUND: jack<-man_anchor, julie<-woman_anchor. Text-forge overridden.")
else:
    print("[merge] AUTO-FORGE mode: skipping dataset-anchor ingest; jack/julie will be INVENTED via SDXL text-forge (set GODSHAND_FORGE=anchors to use real photos).")
    if not _candidates:
        print("[merge] (no dataset anchors present -- expected in auto-forge mode.)")
    # Ensure forge can actually generate: clear any stale/placeholder face refs for the leads.
    for _char in (os.environ.get("KIT_MAN_CHAR", "jack"), os.environ.get("KIT_WOMAN_CHAR", "julie")):
        _fd = os.path.join(KIT_ROOT, "refs", _char, "face")
        if os.path.isdir(_fd):
            for _fn2 in os.listdir(_fd):
                if _fn2.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        os.remove(os.path.join(_fd, _fn2))
                    except Exception:
                        pass

# --- N-CHARACTER SUPPORT (additive): install a real anchor for EVERY story
#     character that declares one, so creators are not limited to jack/julie.
#     Characters with no matching image fall back to text-forge (unchanged).
try:
    import yaml as _yaml
    _story_nc = _yaml.safe_load(STORY_YAML) or {}
    _chars_nc = (_story_nc.get("characters") or {})
    _img_index = {}
    for _r, _d, _fs in os.walk('/kaggle/input'):
        for _fn in _fs:
            if _fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                _img_index.setdefault(_fn, os.path.join(_r, _fn))
                _img_index.setdefault(_fn.lower(), os.path.join(_r, _fn))
    _installed_nc, _missing_nc = [], []
    for _name, _cfg in _chars_nc.items():
        if _name in (os.environ.get("KIT_MAN_CHAR", "jack"), os.environ.get("KIT_WOMAN_CHAR", "julie")):
            continue  # already identity-bound above via man/woman anchors
        _anchor = (_cfg or {}).get("anchor") or (_name + ".png")
        _srcp = _img_index.get(_anchor) or _img_index.get(_anchor.lower())
        if not _srcp:
            _missing_nc.append((_name, _anchor)); continue
        _cdir = os.path.join(KIT_ROOT, "refs", _name)
        shutil.rmtree(_cdir, ignore_errors=True)
        _face = os.path.join(_cdir, "face"); os.makedirs(_face, exist_ok=True)
        _canon = _name + "_anchor.png"
        shutil.copy2(_srcp, os.path.join(_face, _canon))
        shutil.copy2(_srcp, os.path.join(KIT_ROOT, _canon))
        shutil.copy2(_srcp, os.path.join(comfy_input_dir, _canon))
        _installed_nc.append((_name, _anchor))
    for _n, _a in _installed_nc:
        print(f"   refs/{_n}/face/{_n}_anchor.png <- real anchor '{_a}' (identity-locked)")
    for _n, _a in _missing_nc:
        print(f"   [warn] character '{_n}' has no anchor image '{_a}' in /kaggle/input -> text-forge from look tokens")
    if _chars_nc:
        print(f"[OK] N-character pass: {len(_installed_nc)} extra anchored, {len(_missing_nc)} text-forged.")
except Exception as _e_nc:
    print(f"[warn] N-character anchor pass skipped (non-fatal): {_e_nc}")

blank_img = Image.fromarray(np.zeros((480, 720, 3), dtype=np.uint8))
for anchor_name in ['start.png']:
    ap = comfy_input_dir / anchor_name
    if not ap.exists(): blank_img.save(ap)
    lp = Path(KIT_ROOT) / anchor_name
    if not lp.exists(): blank_img.save(lp)

# ==============================================================================
# STEP 6: DEPLOY UPGRADED COMFY_CLIENT.PY
# ==============================================================================
client_path = '/kaggle/working/src/comfy_client.py'
os.makedirs(os.path.dirname(client_path), exist_ok=True)

upgraded_client_code = '''#!/usr/bin/env python3
"""Headless ComfyUI API client with All-Terrain Media Retrieval & Error Interception."""
import glob, json, os, shutil, subprocess, time, urllib.request, urllib.parse, urllib.error, uuid

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
VID_EXT = (".mp4", ".webm", ".mkv", ".mov")

class ComfyClient:
    def __init__(self, host=None, port=None, fps=12):
        host = host or os.environ.get("COMFY_HOST", "127.0.0.1")
        port = int(port or os.environ.get("COMFY_PORT", "8188"))
        self.base = f"http://{host}:{port}"
        self.cid = str(uuid.uuid4())
        self.fps = fps
        self.mock = os.environ.get("COMFY_MOCK") == "1"

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120))

    def _get(self, path):
        return json.load(urllib.request.urlopen(self.base + path, timeout=120))

    @staticmethod
    def _first_image(d):
        if not d or not os.path.isdir(d):
            return None
        for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if f.lower().endswith(IMG_EXT):
                return f
        return None

    def apply_overrides(self, graph, overrides):
        bind = graph.get("_bind", {})
        for key, val in (overrides or {}).items():
            target = bind.get(key)
            if not target:
                continue
            node_id, input_name = target
            node = graph.get(node_id, {}).get("inputs")
            if node is None:
                continue
            if key == "seconds":
                val = int(round(float(val) * self.fps)); val = val - (val % 8) + 1
            elif key == "ref_dirs" and isinstance(val, dict):
                img = None
                for d in val.values():
                    img = self._first_image(d)
                    if img:
                        break
                if not img:
                    continue
                if img and os.path.exists(img):
                    target_name = os.path.basename(img)
                    comfy_in = "/kaggle/working/ComfyUI/input"
                    os.makedirs(comfy_in, exist_ok=True)
                    dest_path = os.path.join(comfy_in, target_name)
                    if not os.path.exists(dest_path):
                        shutil.copy2(img, dest_path)
                    val = target_name
            elif key == "identity_images" and isinstance(val, (list, tuple)):
                if not val:
                    continue
                val = val[0]
            node[input_name] = val
        return graph

    @staticmethod
    def _clean(graph):
        return {k: v for k, v in graph.items() if not k.startswith("_")}

    def run(self, graph, overrides=None, tries=3):
        if getattr(self, "mock", False):
            return {"_mock": {"videos": []}}
        graph = json.loads(json.dumps(graph))
        if overrides:
            graph = self.apply_overrides(graph, overrides)
        prompt = self._clean(graph)
        last = None
        for attempt in range(tries):
            try:
                r = self._post("/prompt", {"prompt": prompt, "client_id": self.cid})
                pid = r["prompt_id"]
                while True:
                    hist = self._get(f"/history/{pid}")
                    if pid in hist:
                        job_data = hist[pid]
                        status = job_data.get("status", {})
                        if status.get("status_str") == "error" or not job_data.get("outputs"):
                            print(f"\\n❌ COMFYUI RUNTIME EXECUTION ERROR DETECTED!")
                            log_path = "/kaggle/working/comfyui.log"
                            if os.path.exists(log_path):
                                print("\\n--- LAST 50 LINES OF COMFYUI SERVER LOG ---")
                                with open(log_path, "r") as lf:
                                    print("".join(lf.readlines()[-50:]))
                            raise RuntimeError(f"ComfyUI job {pid} failed during execution!")
                        return job_data["outputs"]
                    time.sleep(2)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8', errors='ignore')
                print(f"\\n❌ COMFYUI API REJECTED GRAPH (HTTP {e.code}):\\n{err_body}\\n")
                last = f"HTTP {e.code}: {err_body}"
                time.sleep(5 * (attempt + 1))
            except Exception as e:
                last = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"ComfyUI run failed after {tries} tries: {last}")

    def save_image(self, outputs, dest):
        if getattr(self, "mock", False):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if dest.lower().endswith(IMG_EXT):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                                "-i", "testsrc2=size=512x288:rate=1:duration=1",
                                "-frames:v", "1", dest], check=True)
            else:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                                "-i", "testsrc2=size=512x288:rate=24:duration=2",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", dest], check=True)
            return dest

        os.makedirs(os.path.dirname(dest), exist_ok=True)

        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for kind in ("images", "gifs", "videos", "filenames", "output", "media"):
                for media in node_out.get(kind, []):
                    if isinstance(media, str) and os.path.exists(media):
                        shutil.copy2(media, dest)
                        return dest
                    if isinstance(media, dict) and "filename" in media:
                        q = urllib.parse.urlencode({"filename": media["filename"],
                                                    "subfolder": media.get("subfolder", ""),
                                                    "type": media.get("type", "output")})
                        raw = urllib.request.urlopen(self.base + "/view?" + q, timeout=120).read()
                        open(dest, "wb").write(raw)
                        return dest

        comfy_out_dir = os.environ.get("COMFY_OUTPUT_DIR", "/kaggle/working/ComfyUI/output")
        if os.path.exists(comfy_out_dir):
            generated = sorted(glob.glob(os.path.join(comfy_out_dir, "**", "*.*"), recursive=True),
                                     key=os.path.getmtime, reverse=True)
            want = VID_EXT if dest.lower().endswith(VID_EXT) else IMG_EXT
            for f in generated:
                if f.lower().endswith(want):
                    print(f"✓ Retrieved output from disk fallback: {f}")
                    shutil.copy2(f, dest)
                    return dest

        raise RuntimeError(f"No image/video found in ComfyUI outputs! Received: {json.dumps(outputs)}")
'''

with open(client_path, 'w', encoding='utf-8') as f:
    f.write(upgraded_client_code)
print("[client] upgraded comfy_client.py deployed.")

# ==============================================================================
# STEP 7: STITCH.PY STEREO LAYOUT FIX
# ==============================================================================
stitch_file = 'src/stitch.py'
if os.path.exists(stitch_file):
    with open(stitch_file, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(
        "amix=inputs=2:normalize=0,aformat=channel_layouts=stereo[key_in]",
        "amix=inputs=2:normalize=0,pan=stereo|c0=c0|c1=c1[key_in]")
    content = content.replace(
        "aformat=channel_layouts=stereo,volume=-4dB[music_gain]",
        "aformat=channel_layouts=stereo,volume=-4dB,pan=stereo|c0=c0|c1=c1[music_gain]")
    with open(stitch_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("[stitch] FFmpeg channel-layout fix applied.")

# ==============================================================================
# STEP 7.5: SOURCE PATCHES (cv2 import, numpy evaluation patch, progressive chunks)
# ==============================================================================
import re as _re

# --- write CORRECTED source files (root-cause fixes baked in) ---
#     grade_match.py : truly non-fatal wrapper + isolated/guarded face step
#     face_enhance.py: fixes `if not face_rects` ValueError on numpy (Haar) return
import base64 as _b64
_SRC_DIR = "/kaggle/working/src"
os.makedirs(_SRC_DIR, exist_ok=True)
_FIXED_FILES = {
    "grade_match.py": "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJHcmFkZS1tYXRjaCB2NSDigJQgQ0lORU1BVElDIEFSQ0hJVEVDVFVSRS4KICAgICAgICAKVHJhbnNmb3JtIHBpcGVsaW5lOgogIGlucHV0IOKGkiB0ZW1wb3JhbCBkZW5vaXNlIOKGkiBmaWxtIExVVCAocHJvY2VkdXJhbCkg4oaSIG11bHRpLWZyYW1lIGJsZW5kIAogICAgICAgIOKGkiB1bnNoYXJwL0NBUyBkZXRhaWwg4oaSIGhhbGF0aW9uIGJsb29tIOKGkiBncmFpbiDihpIgb3V0cHV0CgpLZXkgaW5ub3ZhdGlvbnMgKGFsbCBDUFUsIGFsbCBudW1weSwgbm8gc2NpcHkpOgogICogUHJvY2VkdXJhbCBmaWxtIHByaW50IExVVHMg4oCUIGtvZGFjaHJvbWUva29kYWsvYXJyaSBlbXVsYXRpb24gZ2VuZXJhdGVkIGF0IHJ1bnRpbWUKICAqIE11bHRpLWZyYW1lIGdhdXNzaWFuLXdlaWdodGVkIHRlbXBvcmFsIGJsZW5kaW5nIGZvciBjb2hlcmVuY2UKICAqIENvbnRyYXN0LWFkYXB0aXZlIHNoYXJwZW5pbmcgKENBUykg4oCUIGRldGFpbCBzeW50aGVzaXMgd2l0aG91dCByaW5naW5nCiAgKiBIYWxhdGlvbiBibG9vbSB3aXRoIHJlYWxpc3RpYyBoaWdobGlnaHQgcm9sbG9mZgogICogVGVtcG9yYWxseS1jb2hlcmVudCBncmFpbiB3aXRoIGZyYWN0YWwgc3ViLWxheWVyCiAgICAgICAgCkZ1bGx5IG5vbi1mYXRhbDogZmFpbHVyZSDihpIgY29waWVzIGlucHV0IHRvIG91dHB1dCB1bmNoYW5nZWQuCiIiIgppbXBvcnQgc3lzLCBvcywganNvbiwgZ2xvYiwgdGVtcGZpbGUsIHN1YnByb2Nlc3MsIHN0cnVjdCwgc2h1dGlsCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgY3YyCgoKIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQojIEZJTE0gUFJJTlQgTFVUIOKAlCBwcm9jZWR1cmFsIGVtdWxhdGlvbiwgemVybyBleHRlcm5hbCBmaWxlcwojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CgojIEJ1aWx0LWluIGZpbG0gcHJlc2V0czogZWFjaCBkZWZpbmVzIGEgY29sb3IgdHJhbnNmb3JtIGZyb20gbG9nL3NSR0IgdG8gcHJpbnQKRklMTV9QUkVTRVRTID0gewogICAgImtvZGFjaHJvbWUiOiB7CiAgICAgICAgImNvbnRyYXN0IjogMS4xNSwgInNhdHVyYXRpb24iOiAxLjIsCiAgICAgICAgIndhcm10aCI6IFsxLjAsIDAuOTQsIDAuODVdLAogICAgICAgICJzaGFkb3dzIjogWzAuMDIsIDAuMDEsIDAuMDE1XSwKICAgICAgICAiaGlnaGxpZ2h0cyI6IFswLjk4LCAwLjk2LCAwLjkyXSwKICAgIH0sCiAgICAia29kYWtfNTI3OSI6IHsKICAgICAgICAiY29udHJhc3QiOiAxLjEwLCAic2F0dXJhdGlvbiI6IDEuMDUsCiAgICAgICAgIndhcm10aCI6IFsxLjAsIDAuOTgsIDAuOTddLAogICAgICAgICJzaGFkb3dzIjogWzAuMDEsIDAuMDE1LCAwLjAyXSwKICAgICAgICAiaGlnaGxpZ2h0cyI6IFswLjk3LCAwLjk4LCAwLjk5XSwKICAgIH0sCiAgICAiYXJyaV9hY2VzIjogewogICAgICAgICJjb250cmFzdCI6IDEuMDUsICJzYXR1cmF0aW9uIjogMS4wLAogICAgICAgICJ3YXJtdGgiOiBbMS4wLCAxLjAsIDEuMF0sCiAgICAgICAgInNoYWRvd3MiOiBbMC4wMSwgMC4wMSwgMC4wMV0sCiAgICAgICAgImhpZ2hsaWdodHMiOiBbMC45NSwgMC45NiwgMC45N10sCiAgICB9LAogICAgImJsZWFjaF9ieXBhc3MiOiB7CiAgICAgICAgImNvbnRyYXN0IjogMS4zMCwgInNhdHVyYXRpb24iOiAwLjYsCiAgICAgICAgIndhcm10aCI6IFsxLjAsIDAuOTgsIDAuOTVdLAogICAgICAgICJzaGFkb3dzIjogWzAuMDMsIDAuMDMsIDAuMDNdLAogICAgICAgICJoaWdobGlnaHRzIjogWzAuOTUsIDAuOTMsIDAuOTBdLAogICAgfSwKICAgICJ0ZWFsX29yYW5nZSI6IHsKICAgICAgICAiY29udHJhc3QiOiAxLjIwLCAic2F0dXJhdGlvbiI6IDEuMywKICAgICAgICAid2FybXRoIjogWzEuMDUsIDAuOTYsIDAuODVdLAogICAgICAgICJzaGFkb3dzIjogWzAuMCwgMC4wMiwgMC4wNF0sCiAgICAgICAgImhpZ2hsaWdodHMiOiBbMS4wLCAwLjk1LCAwLjkwXSwKICAgIH0sCn0KCgpkZWYgYXBwbHlfZmlsbV9sdXQoZnJhbWUsIHByZXNldD0ia29kYWNocm9tZSIpOgogICAgIiIiQXBwbHkgZmlsbSBwcmludCBlbXVsYXRpb24gdG8gYW4gUkdCIGZyYW1lIFtILFcsM10gZmxvYXQzMiBbMCwxXS4iIiIKICAgIHAgPSBGSUxNX1BSRVNFVFMuZ2V0KHByZXNldCwgRklMTV9QUkVTRVRTWyJrb2RhY2hyb21lIl0pCiAgICBsdXQgPSBmcmFtZS5jb3B5KCkKICAgIAogICAgIyBTLWN1cnZlIGNvbnRyYXN0CiAgICBsdXQgPSBucC5wb3dlcihsdXQsIDEuMCAvIHBbImNvbnRyYXN0Il0pCiAgICAKICAgICMgU2hhZG93cy9oaWdobGlnaHRzIHJvbGwtb2ZmIHBlciBjaGFubmVsCiAgICBmb3IgYyBpbiByYW5nZSgzKToKICAgICAgICBzLCBoID0gcFsic2hhZG93cyJdW2NdLCBwWyJoaWdobGlnaHRzIl1bY10KICAgICAgICBsdXRbOiw6LGNdID0gbnAuaW50ZXJwKGx1dFs6LDosY10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbMC4wLCBzLCAwLjUsIDAuOCwgMS4wXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFswLjAsIDAuMCwgMC41LCAwLjgqaCwgaF0pCiAgICAKICAgICMgQ29sb3Igd2FybXRoCiAgICB3YXJtdGggPSBucC5hcnJheShwWyJ3YXJtdGgiXSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIGx1dCAqPSB3YXJtdGgKICAgIAogICAgIyBTYXR1cmF0aW9uCiAgICBncmF5ID0gbnAubWVhbihsdXQsIGF4aXM9LTEsIGtlZXBkaW1zPVRydWUpCiAgICBsdXQgPSBncmF5ICsgcFsic2F0dXJhdGlvbiJdICogKGx1dCAtIGdyYXkpCiAgICAKICAgIHJldHVybiBucC5jbGlwKGx1dCwgMCwgMSkKCgojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiMgTVVMVEktRlJBTUUgR0FVU1NJQU4gVEVNUE9SQUwgQkxFTkRJTkcKIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQoKZGVmIHRlbXBvcmFsX2JsZW5kKGZyYW1lcywgd2luZG93PTUsIHNpZ21hPTEuNSk6CiAgICAiIiJNdWx0aS1mcmFtZSBnYXVzc2lhbi13ZWlnaHRlZCB0ZW1wb3JhbCBibGVuZGluZy4KICAgIEVhY2ggb3V0cHV0IGZyYW1lIGlzIGEgd2VpZ2h0ZWQgYmxlbmQgb2YgwrF3aW5kb3cvMiBhZGphY2VudCBmcmFtZXMuCiAgICBFbGltaW5hdGVzIHRlbXBvcmFsIGZsaWNrZXIgd2hpbGUgcHJlc2VydmluZyBsZWdpdGltYXRlIG1vdGlvbi4KICAgICIiIgogICAgaWYgbGVuKGZyYW1lcykgPCAzOgogICAgICAgIHJldHVybiBmcmFtZXMKICAgIHdzID0gbWluKHdpbmRvdywgbGVuKGZyYW1lcykpCiAgICBpZiB3cyAlIDIgPT0gMDoKICAgICAgICB3cyArPSAxCiAgICBoYWxmID0gd3MgLy8gMgogICAgd2VpZ2h0cyA9IG5wLmV4cCgtMC41ICogKChucC5hcmFuZ2Uod3MpIC0gaGFsZikgLyBzaWdtYSkgKiogMikKICAgIHdlaWdodHMgLz0gd2VpZ2h0cy5zdW0oKQogICAgCiAgICBwYWRkZWQgPSBucC5wYWQoZnJhbWVzLCAoKGhhbGYsIGhhbGYpLCAoMCwgMCksICgwLCAwKSwgKDAsIDApKSwgbW9kZT0nZWRnZScpCiAgICBvdXQgPSBucC56ZXJvc19saWtlKGZyYW1lcykKICAgIGZvciBpIGluIHJhbmdlKGxlbihmcmFtZXMpKToKICAgICAgICBmb3IgaiBpbiByYW5nZSh3cyk6CiAgICAgICAgICAgIG91dFtpXSArPSBwYWRkZWRbaSArIGpdICogd2VpZ2h0c1tqXQogICAgcmV0dXJuIG91dAoKCiMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KIyBDT05UUkFTVC1BREFQVElWRSBTSEFSUEVOSU5HIChDQVMpCiMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KCmRlZiBjYXNfc2hhcnBlbihmcmFtZSwgc3RyZW5ndGg9MC41KToKICAgICIiIkNvbnRyYXN0LWFkYXB0aXZlIHNoYXJwZW5pbmcg4oCUIGVuaGFuY2VzIGRldGFpbCB3aXRob3V0IHJpbmdpbmcuCiAgICBQdXJlIG51bXB5IGltcGxlbWVudGF0aW9uIG9mIEFNRCBGaWRlbGl0eUZYIENBUy4KICAgIE9ubHkgc2hhcnBlbnMgbWVkaXVtLWNvbnRyYXN0IGVkZ2VzLCBsZWF2ZXMgc21vb3RoIGFyZWFzIGFsb25lLgogICAgIiIiCiAgICBncmF5ID0gbnAubWVhbihmcmFtZSwgYXhpcz0tMSkKICAgICMgTG9jYWwgbWluL21heCBmb3IgZWRnZSBkZXRlY3Rpb24g4oCUIG1hbnVhbCBzbGlkaW5nIHdpbmRvdwogICAgaCwgdyA9IGZyYW1lLnNoYXBlWzoyXQogICAgCiAgICAjIFNpbXBsZSBMYXBsYWNpYW4gZWRnZSBkZXRlY3Rpb24KICAgIGxhcCA9IChmcmFtZVs6LTIsIDE6LTFdICsgZnJhbWVbMjosIDE6LTFdICsgCiAgICAgICAgICAgZnJhbWVbMTotMSwgOi0yXSArIGZyYW1lWzE6LTEsIDI6XSAtIDQgKiBmcmFtZVsxOi0xLCAxOi0xXSkKICAgIGVkZ2VfaW50ZW5zaXR5ID0gbnAuYWJzKG5wLm1lYW4obGFwLCBheGlzPS0xKSkKICAgIAogICAgIyBBZGFwdGl2ZSBnYWluOiBoaWdoIG9uIG1lZGl1bSBlZGdlcywgbG93IG9uIGZsYXQvc2F0dXJhdGVkIGVkZ2VzCiAgICBnYWluID0gc3RyZW5ndGggKiBucC5jbGlwKDEuMCAtIG5wLmV4cCgtZWRnZV9pbnRlbnNpdHkgKiAyMCksIDAsIDEpCiAgICBnYWluICo9IG5wLmNsaXAoMS4wIC0gbnAuZXhwKC1lZGdlX2ludGVuc2l0eSAqIDUpLCAwLjEsIDEpICAjIHN1cHByZXNzIHZlcnkgc3Ryb25nIGVkZ2VzCiAgICAKICAgIHJlc3VsdCA9IGZyYW1lLmNvcHkoKQogICAgZm9yIGMgaW4gcmFuZ2UoMyk6CiAgICAgICAgc2hhcnAgPSBmcmFtZVs6LDosY10uY29weSgpCiAgICAgICAgIyBVbnNoYXJwIG1hc2sgdmlhIGN2Mi5HYXVzc2lhbkJsdXIgKHZlY3Rvcml6ZWQgQysrLCAxMDAweCBmYXN0ZXIpCiAgICAgICAgYmx1ciA9IGN2Mi5HYXVzc2lhbkJsdXIoc2hhcnAsICgwLCAwKSwgMS41KQogICAgICAgIGRldGFpbCA9IHNoYXJwIC0gYmx1cgogICAgICAgIGdhaW5fbWFwID0gbnAucGFkKGdhaW4sIDEsIG1vZGU9J2VkZ2UnKQogICAgICAgIHJlc3VsdFsxOi0xLDE6LTEsY10gKz0gZGV0YWlsWzE6LTEsMTotMV0gKiBnYWluX21hcFsxOi0xLDE6LTFdCiAgICAKICAgIHJldHVybiBucC5jbGlwKHJlc3VsdCwgMCwgMSkKCgojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiMgSEFMQVRJT04gQkxPT00KIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQoKZGVmIGhhbGF0aW9uX2Jsb29tKGZyYW1lLCBzdHJlbmd0aD0wLjE1KToKICAgICIiIkZpbG0tbGlrZSBoYWxhdGlvbiBibG9vbTogcmVkL29yYW5nZSBnbG93IGFyb3VuZCBicmlnaHQgaGlnaGxpZ2h0cy4KICAgIEVtdWxhdGVzIGxpZ2h0IHNjYXR0ZXJpbmcgdGhyb3VnaCBmaWxtIGVtdWxzaW9uIGxheWVycy4KICAgICIiIgogICAgIyBMdW1pbmFuY2UgbWFzawogICAgZ3JheSA9IG5wLm1lYW4oZnJhbWUsIGF4aXM9LTEpCiAgICBicmlnaHQgPSBucC5jbGlwKGdyYXkgLSAwLjYsIDAsIDAuNCkgLyAwLjQgICMgaGlnaGxpZ2h0cyBvbmx5CiAgICAKICAgICMgV2FybS1jb2xvcmVkIGJsb29tCiAgICBibG9vbSA9IG5wLnplcm9zX2xpa2UoZnJhbWUpCiAgICBmb3IgYyBpbiByYW5nZSgzKToKICAgICAgICB3ZWlnaHQgPSBbMS4wLCAwLjcsIDAuM11bY10gICMgcmVkLWhlYXZ5IGZvciB3YXJtIGdsb3cKICAgICAgICBibG9vbVs6LDosY10gPSBicmlnaHQgKiB3ZWlnaHQKICAgIAogICAgIyBHYXVzc2lhbiBibHVyIHZpYSBjdjIgKHZlY3Rvcml6ZWQsIG9wdGltaXplZCkKICAgIGtzaXplID0gbWF4KDMsIGludChtaW4oZnJhbWUuc2hhcGVbOjJdKSAqIDAuMDMpKQogICAgaWYga3NpemUgJSAyID09IDA6CiAgICAgICAga3NpemUgKz0gMQogICAgZm9yIGMgaW4gcmFuZ2UoMyk6CiAgICAgICAgYmxvb21bOiw6LGNdID0gY3YyLkdhdXNzaWFuQmx1cihibG9vbVs6LDosY10sIChrc2l6ZSwga3NpemUpLCBrc2l6ZSAqIDAuMykKICAgIAogICAgYmxlbmRlZCA9IGZyYW1lICsgYmxvb20gKiBzdHJlbmd0aAogICAgcmV0dXJuIG5wLmNsaXAoYmxlbmRlZCwgMCwgMSkKCgpkZWYgX3NlcGFyYWJsZV9ibHVyKGltZywga3NpemUpOgogICAgIiIiRmFzdCBzZXBhcmFibGUgYm94IGJsdXIgKHZlcnRpY2FsICsgaG9yaXpvbnRhbCkuIiIiCiAgICBoLCB3ID0gaW1nLnNoYXBlWzoyXQogICAgcmVzdWx0ID0gaW1nLmNvcHkoKQogICAgazFkID0gbnAub25lcyhrc2l6ZSwgZHR5cGU9bnAuZmxvYXQzMikgLyBrc2l6ZQogICAgIyBIb3Jpem9udGFsCiAgICBmb3IgYyBpbiByYW5nZSgzKToKICAgICAgICBmb3IgeSBpbiByYW5nZShoKToKICAgICAgICAgICAgcmVzdWx0W3ksOixjXSA9IG5wLmNvbnZvbHZlKGltZ1t5LDosY10sIGsxZCwgbW9kZT0nc2FtZScpCiAgICAjIFZlcnRpY2FsCiAgICBmb3IgYyBpbiByYW5nZSgzKToKICAgICAgICBmb3IgeCBpbiByYW5nZSh3KToKICAgICAgICAgICAgcmVzdWx0WzoseCxjXSA9IG5wLmNvbnZvbHZlKHJlc3VsdFs6LHgsY10sIGsxZCwgbW9kZT0nc2FtZScpCiAgICByZXR1cm4gcmVzdWx0CgoKIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQojIEZJTE0gR1JBSU4gKGZyYWN0YWwgc3ViLWxheWVyKQojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CgpkZWYgZmlsbV9ncmFpbihudW1fZnJhbWVzLCBoLCB3LCBzdHJlbmd0aD0zLjUsIGNocm9tYT0wLjUsIHNlZWQ9NDIpOgogICAgIiIiVGVtcG9yYWxseS1jb2hlcmVudCBmcmFjdGFsIGZpbG0gZ3JhaW4uIAogICAgVXNlcyBvdmVybGFwcGluZyBvY3RhdmVzIGZvciBuYXR1cmFsLWxvb2tpbmcgZ3JhaW4gdGV4dHVyZS4KICAgICIiIgogICAgcm5nID0gbnAucmFuZG9tLmRlZmF1bHRfcm5nKHNlZWQpCiAgICAjIEdlbmVyYXRlIGJhc2UgZ3JhaW4gKyAyIG9jdGF2ZXMgZm9yIGZyYWN0YWwgZGV0YWlsCiAgICBncmFpbiA9IG5wLnplcm9zKChudW1fZnJhbWVzLCBoLCB3LCAzKSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIGZvciBvY3RhdmUgaW4gcmFuZ2UoMyk6CiAgICAgICAgc2NhbGUgPSAyICoqIG9jdGF2ZQogICAgICAgIHNoLCBzdyA9IG1heCgxLCBoIC8vIHNjYWxlKSwgbWF4KDEsIHcgLy8gc2NhbGUpCiAgICAgICAgYmFzZSA9IHJuZy5ub3JtYWwoMCwgMSwgKG51bV9mcmFtZXMsIHNoLCBzdykpLmFzdHlwZShucC5mbG9hdDMyKQogICAgICAgICMgVGVtcG9yYWwgc21vb3RoaW5nCiAgICAgICAgYWxwaGEgPSAwLjQgKyBvY3RhdmUgKiAwLjA1CiAgICAgICAgZm9yIGYgaW4gcmFuZ2UoMSwgbnVtX2ZyYW1lcyk6CiAgICAgICAgICAgIGJhc2VbZl0gPSBiYXNlW2ZdICogKDEgLSBhbHBoYSkgKyBiYXNlW2YtMV0gKiBhbHBoYQogICAgICAgICMgVXBzYW1wbGUgdG8gZnVsbCByZXMgdmlhIGN2Mi5yZXNpemUgKHZlY3Rvcml6ZWQsIDEwMDB4IGZhc3RlciB0aGFuIFB5dGhvbiBsb29wKQogICAgICAgIHVwID0gbnAuemVyb3MoKG51bV9mcmFtZXMsIGgsIHcpLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgICAgIGZvciBmIGluIHJhbmdlKG51bV9mcmFtZXMpOgogICAgICAgICAgICB1cFtmXSA9IGN2Mi5yZXNpemUoYmFzZVtmXSwgKHcsIGgpLCBpbnRlcnBvbGF0aW9uPWN2Mi5JTlRFUl9MSU5FQVIpCiAgICAgICAgIyBBZGQgb2N0YXZlIHdpdGggZGVjcmVhc2luZyBzdHJlbmd0aAogICAgICAgIG9jdF9zdHIgPSBzdHJlbmd0aCAvICgyICoqIG9jdGF2ZSkKICAgICAgICBmb3IgYyBpbiByYW5nZSgzKToKICAgICAgICAgICAgY2hyb21hX3cgPSBjaHJvbWEgaWYgYyA+IDAgZWxzZSAxLjAgICMgbHVtYSA9IGZ1bGwsIGNocm9tYSA9IHBhcnRpYWwKICAgICAgICAgICAgZ3JhaW5bOiw6LDosY10gKz0gdXAgKiBvY3Rfc3RyICogY2hyb21hX3cgKiAwLjAxCiAgICAKICAgIHJldHVybiBncmFpbgoKCiMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KIyBNQUlOIFBJUEVMSU5FCiMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KCmRlZiBncmFkZV92aWRlbyhpbl9wYXRoLCBvdXRfcGF0aCwgcHJlc2V0PSJrb2RhY2hyb21lIiwgc3RyZW5ndGg9MC44LCBncmFpbl9zdHI9My41KToKICAgICIiIk5vbi1mYXRhbCB3cmFwcGVyOiBvbiBBTlkgZXJyb3IsIGNvcHkgaW5wdXQgLT4gb3V0cHV0IHVuY2hhbmdlZCAoYXMgdGhlCiAgICBtb2R1bGUgZG9jc3RyaW5nIHByb21pc2VzKS4gR3VhcmFudGVlcyBhIGZhaWxlZCBncmFkZSBuZXZlciBhYm9ydHMgdGhlIHJlbmRlcgogICAgb3IgbGVhdmVzIGEgbWlzc2luZyBvdXRwdXQgZmlsZS4iIiIKICAgIHRyeToKICAgICAgICBfZ3JhZGVfdmlkZW9faW1wbChpbl9wYXRoLCBvdXRfcGF0aCwgcHJlc2V0LCBzdHJlbmd0aCwgZ3JhaW5fc3RyKQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgIGltcG9ydCB0cmFjZWJhY2sKICAgICAgICBwcmludChmIltncmFkZV0gV0FSTklORzogZ3JhZGUgZmFpbGVkICh7ZSFyfSk7IGNvcHlpbmcgaW5wdXQgdW5jaGFuZ2VkIC0+IHtvdXRfcGF0aH0iKQogICAgICAgIHRyYWNlYmFjay5wcmludF9leGMoKQogICAgICAgIHRyeToKICAgICAgICAgICAgc2h1dGlsLmNvcHkyKGluX3BhdGgsIG91dF9wYXRoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZTI6CiAgICAgICAgICAgIHByaW50KGYiW2dyYWRlXSBFUlJPUjogZmFsbGJhY2sgY29weSBhbHNvIGZhaWxlZDoge2UyIXJ9IikKCgpkZWYgX2dyYWRlX3ZpZGVvX2ltcGwoaW5fcGF0aCwgb3V0X3BhdGgsIHByZXNldD0ia29kYWNocm9tZSIsIHN0cmVuZ3RoPTAuOCwgZ3JhaW5fc3RyPTMuNSk6CiAgICAiIiJGdWxsIGNpbmVtYXRpYyBncmFkZSBwaXBlbGluZSB1c2luZyBwdXJlIGZmbXBlZyArIG51bXB5LiIiIgogICAgdG1wID0gdGVtcGZpbGUubWtkdGVtcCgpCiAgICB0cnk6CiAgICAgICAgIyBFeHRyYWN0IGZyYW1lcyBhcyBQTkcKICAgICAgICBmcHMgPSBfZ2V0X2Zwcyhpbl9wYXRoKQogICAgICAgIHN1YnByb2Nlc3MucnVuKFsiZmZtcGVnIiwgIi15IiwgIi1sb2dsZXZlbCIsICJlcnJvciIsICItaSIsIGluX3BhdGgsCiAgICAgICAgICAgICAgICAgICAgICAgIG9zLnBhdGguam9pbih0bXAsICJmXyUwNmQucG5nIildLCBjaGVjaz1UcnVlKQogICAgICAgIGZyYW1lc19pbiA9IHNvcnRlZChnbG9iLmdsb2Iob3MucGF0aC5qb2luKHRtcCwgImZfKi5wbmciKSkpCiAgICAgICAgaWYgbm90IGZyYW1lc19pbjoKICAgICAgICAgICAgc2h1dGlsLmNvcHkyKGluX3BhdGgsIG91dF9wYXRoKQogICAgICAgICAgICByZXR1cm4KICAgICAgICAKICAgICAgICAjIExvYWQgZnJhbWVzCiAgICAgICAgaW1wb3J0IGN2MgogICAgICAgIGZyYW1lcyA9IFtdCiAgICAgICAgZm9yIGYgaW4gZnJhbWVzX2luOgogICAgICAgICAgICBpbWcgPSBjdjIuaW1yZWFkKGYsIGN2Mi5JTVJFQURfQ09MT1IpCiAgICAgICAgICAgIGltZyA9IGN2Mi5jdnRDb2xvcihpbWcsIGN2Mi5DT0xPUl9CR1IyUkdCKS5hc3R5cGUobnAuZmxvYXQzMikgLyAyNTUuMAogICAgICAgICAgICBmcmFtZXMuYXBwZW5kKGltZykKICAgICAgICAKICAgICAgICBmcmFtZXMgPSBucC5zdGFjayhmcmFtZXMpCiAgICAgICAgbiwgaCwgdyA9IGZyYW1lcy5zaGFwZVs6M10KICAgICAgICAKICAgICAgICAjIDEpIEZpbG0gTFVUIGdyYWRlCiAgICAgICAgcHJpbnQoZiJbZ3JhZGVdIHtwcmVzZXR9IGZpbG0gcHJpbnQgZW11bGF0aW9uLi4uIikKICAgICAgICBmb3IgaSBpbiByYW5nZShuKToKICAgICAgICAgICAgZnJhbWVzW2ldID0gYXBwbHlfZmlsbV9sdXQoZnJhbWVzW2ldLCBwcmVzZXQpCiAgICAgICAgCiAgICAgICAgIyAyKSBNdWx0aS1mcmFtZSB0ZW1wb3JhbCBibGVuZGluZwogICAgICAgIHByaW50KGYiW2dyYWRlXSB0ZW1wb3JhbCBibGVuZGluZyAoe259IGZyYW1lcykuLi4iKQogICAgICAgIGZyYW1lcyA9IHRlbXBvcmFsX2JsZW5kKGZyYW1lcywgd2luZG93PTUsIHNpZ21hPTEuNSkKICAgICAgICAKICAgICAgICAjIDMpIENBUyBkZXRhaWwgZW5oYW5jZW1lbnQKICAgICAgICBwcmludChmIltncmFkZV0gY29udHJhc3QtYWRhcHRpdmUgc2hhcnBlbmluZy4uLiIpCiAgICAgICAgZm9yIGkgaW4gcmFuZ2Uobik6CiAgICAgICAgICAgIGZyYW1lc1tpXSA9IGNhc19zaGFycGVuKGZyYW1lc1tpXSwgc3RyZW5ndGggKiAwLjMpCiAgICAgICAgCiAgICAgICAgIyA0KSBIYWxhdGlvbiBibG9vbQogICAgICAgIHByaW50KGYiW2dyYWRlXSBoYWxhdGlvbiBibG9vbS4uLiIpCiAgICAgICAgZm9yIGkgaW4gcmFuZ2Uobik6CiAgICAgICAgICAgIGZyYW1lc1tpXSA9IGhhbGF0aW9uX2Jsb29tKGZyYW1lc1tpXSwgMC4xNSAqIHN0cmVuZ3RoKQogICAgICAgIAogICAgICAgICMgNSkgRmlsbSBncmFpbgogICAgICAgIHByaW50KGYiW2dyYWRlXSBmcmFjdGFsIGZpbG0gZ3JhaW4uLi4iKQogICAgICAgIGdyYWluID0gZmlsbV9ncmFpbihuLCBoLCB3LCBzdHJlbmd0aD1ncmFpbl9zdHIsIGNocm9tYT0wLjUpCiAgICAgICAgZnJhbWVzID0gbnAuY2xpcChmcmFtZXMgKyBncmFpbiwgMCwgMSkKICAgICAgICAKICAgICAgICAjIDYpIEZhY2Utc3BlY2lmaWMgc3VwZXItcmVzb2x1dGlvbiAoNHggZmFjZSB1cHNjYWxlKQogICAgICAgICMgSXNvbGF0ZWQgKyBvcHRpb25hbDogaWYgdGhpcyBzdGVwIHJhaXNlcyAoZS5nLiB0aGUgbnVtcHkgImFtYmlndW91cyB0cnV0aAogICAgICAgICMgdmFsdWUiIFZhbHVlRXJyb3IgaW5zaWRlIGZhY2VfZW5oYW5jZSksIHRoZSByZXN0IG9mIHRoZSBncmFkZSBpcyBrZXB0CiAgICAgICAgIyBpbnN0ZWFkIG9mIHRoZSB3aG9sZSBjbGlwIGZhbGxpbmcgYmFjayB0byB1bmdyYWRlZC4gU2V0CiAgICAgICAgIyBLSVRfTk9fRkFDRV9FTkhBTkNFPTEgdG8gc2tpcCBpdCBlbnRpcmVseSAoYWxzbyBzYXZlcyBSQU0gb24gdGlnaHQgaG9zdHMpLgogICAgICAgIGlmIG9zLmVudmlyb24uZ2V0KCJLSVRfTk9fRkFDRV9FTkhBTkNFIiwgIjAiKSAhPSAiMSI6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHByaW50KGYiW2dyYWRlXSBmYWNlIGVuaGFuY2VtZW50Li4uIikKICAgICAgICAgICAgICAgIGltcG9ydCBmYWNlX2VuaGFuY2UKICAgICAgICAgICAgICAgIGZvciBpIGluIHJhbmdlKG4pOgogICAgICAgICAgICAgICAgICAgIGZyYW1lX2JnciA9IGN2Mi5jdnRDb2xvcigoZnJhbWVzW2ldICogMjU1KS5hc3R5cGUobnAudWludDgpLCBjdjIuQ09MT1JfUkdCMkJHUikKICAgICAgICAgICAgICAgICAgICBlbmhhbmNlZCA9IGZhY2VfZW5oYW5jZS5lbmhhbmNlX2ZyYW1lKGZyYW1lX2JnciwgZmFjZV9zY2FsZT00LjAsIGRldGFpbF9zdHJlbmd0aD0wLjYpCiAgICAgICAgICAgICAgICAgICAgZnJhbWVzW2ldID0gY3YyLmN2dENvbG9yKGVuaGFuY2VkLCBjdjIuQ09MT1JfQkdSMlJHQikuYXN0eXBlKG5wLmZsb2F0MzIpIC8gMjU1LjAKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICAgICAgcHJpbnQoZiJbZ3JhZGVdIGZhY2UgZW5oYW5jZW1lbnQgc2tpcHBlZCAoe2Uhcn0pIikKICAgICAgICBlbHNlOgogICAgICAgICAgICBwcmludCgiW2dyYWRlXSBmYWNlIGVuaGFuY2VtZW50IHNraXBwZWQgKEtJVF9OT19GQUNFX0VOSEFOQ0U9MSkiKQogICAgICAgIAogICAgICAgICMgV3JpdGUgZnJhbWVzIGJhY2sKICAgICAgICBmb3IgaSBpbiByYW5nZShuKToKICAgICAgICAgICAgb3V0ID0gKGZyYW1lc1tpXSAqIDI1NSkuYXN0eXBlKG5wLnVpbnQ4KQogICAgICAgICAgICBvdXQgPSBjdjIuY3Z0Q29sb3Iob3V0LCBjdjIuQ09MT1JfUkdCMkJHUikKICAgICAgICAgICAgY3YyLmltd3JpdGUoZnJhbWVzX2luW2ldLCBvdXQsIFtjdjIuSU1XUklURV9QTkdfQ09NUFJFU1NJT04sIDFdKQogICAgICAgIAogICAgICAgICMgUmUtZW5jb2RlCiAgICAgICAgc3VicHJvY2Vzcy5ydW4oWyJmZm1wZWciLCAiLXkiLCAiLWxvZ2xldmVsIiwgImVycm9yIiwKICAgICAgICAgICAgICAgICAgICAgICAiLWZyYW1lcmF0ZSIsIGYie2ZwczouNGZ9IiwKICAgICAgICAgICAgICAgICAgICAgICAiLWkiLCBvcy5wYXRoLmpvaW4odG1wLCAiZl8lMDZkLnBuZyIpLAogICAgICAgICAgICAgICAgICAgICAgICItYzp2IiwgImxpYngyNjQiLCAiLXByZXNldCIsICJzbG93IiwgIi1jcmYiLCAiMTAiLAogICAgICAgICAgICAgICAgICAgICAgICItcGl4X2ZtdCIsICJ5dXY0MjBwIiwKICAgICAgICAgICAgICAgICAgICAgICAiLWNvbG9yc3BhY2UiLCAiYnQ3MDkiLCAiLWNvbG9yX3ByaW1hcmllcyIsICJidDcwOSIsICItY29sb3JfdHJjIiwgImJ0NzA5IiwKICAgICAgICAgICAgICAgICAgICAgICBvdXRfcGF0aF0sIGNoZWNrPVRydWUpCiAgICAgICAgcHJpbnQoZiJbZ3JhZGVdIC0+IHtvdXRfcGF0aH0gKHtwcmVzZXR9LCB7c3RyZW5ndGg6LjFmfSkiKQogICAgZmluYWxseToKICAgICAgICBzaHV0aWwucm10cmVlKHRtcCwgaWdub3JlX2Vycm9ycz1UcnVlKQoKCmRlZiBfZ2V0X2ZwcyhwYXRoKToKICAgIHRyeToKICAgICAgICByID0gc3VicHJvY2Vzcy5ydW4oWyJmZnByb2JlIiwgIi12IiwgImVycm9yIiwgIi1zZWxlY3Rfc3RyZWFtcyIsICJ2OjAiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAiLXNob3dfZW50cmllcyIsICJzdHJlYW09cl9mcmFtZV9yYXRlIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgIi1vZiIsICJkZWZhdWx0PW53PTE6bms9MSIsIHBhdGhdLAogICAgICAgICAgICAgICAgICAgICAgICAgIGNhcHR1cmVfb3V0cHV0PVRydWUsIHRleHQ9VHJ1ZSkKICAgICAgICBhLCBiID0gci5zdGRvdXQuc3RyaXAoKS5zcGxpdCgiLyIpCiAgICAgICAgcmV0dXJuIGZsb2F0KGEpIC8gZmxvYXQoYikKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIDI0LjAKCgpkZWYgX2NvcHkoc3JjLCBkc3QpOgogICAgc2h1dGlsLmNvcHkyKHNyYywgZHN0KQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBpZiBsZW4oc3lzLmFyZ3YpIDwgMzoKICAgICAgICAjIENhbGxlZCBmcm9tIHBvc3Rwcm9jZXNzOiBncmFkZV9tYXRjaC5weSBJTiBPVVQgVEFSR0VUIFtTVFJFTkdUSF0gW0dSQUlOXQogICAgICAgIGluX3BhdGgsIG91dF9wYXRoID0gc3lzLmFyZ3ZbMV0sIHN5cy5hcmd2WzJdCiAgICAgICAgdGFyZ2V0X2pzb24gPSBzeXMuYXJndlszXSBpZiBsZW4oc3lzLmFyZ3YpID4gMyBlbHNlICIiCiAgICAgICAgc3RyZW5ndGggPSBmbG9hdChzeXMuYXJndls0XSkgaWYgbGVuKHN5cy5hcmd2KSA+IDQgZWxzZSAwLjgKICAgICAgICBncmFpbl9zdHIgPSBmbG9hdChzeXMuYXJndls1XSkgaWYgbGVuKHN5cy5hcmd2KSA+IDUgZWxzZSAzLjUKICAgICAgICAKICAgICAgICBpZiBub3Qgb3MucGF0aC5leGlzdHMoaW5fcGF0aCk6CiAgICAgICAgICAgIF9jb3B5KGluX3BhdGgsIG91dF9wYXRoKQogICAgICAgICAgICBzeXMuZXhpdCgwKQogICAgICAgIAogICAgICAgICMgUmVhZCB0YXJnZXQgcHJlc2V0IGZyb20gSlNPTiBpZiBhdmFpbGFibGUKICAgICAgICBwcmVzZXQgPSAia29kYWNocm9tZSIKICAgICAgICBpZiB0YXJnZXRfanNvbiBhbmQgb3MucGF0aC5leGlzdHModGFyZ2V0X2pzb24pOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICB0ID0ganNvbi5sb2FkKG9wZW4odGFyZ2V0X2pzb24pKQogICAgICAgICAgICAgICAgcHJlc2V0ID0gdC5nZXQoImZpbG1fcHJlc2V0IiwgImtvZGFjaHJvbWUiKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgIAogICAgICAgIGdyYWRlX3ZpZGVvKGluX3BhdGgsIG91dF9wYXRoLCBwcmVzZXQsIHN0cmVuZ3RoLCBncmFpbl9zdHIpCiAgICBlbHNlOgogICAgICAgICMgRGlyZWN0IGNhbGwKICAgICAgICBncmFkZV92aWRlbyhzeXMuYXJndlsxXSwgc3lzLmFyZ3ZbMl0sIAogICAgICAgICAgICAgICAgICAgIHN5cy5hcmd2WzNdIGlmIGxlbihzeXMuYXJndikgPiAzIGVsc2UgImtvZGFjaHJvbWUiLAogICAgICAgICAgICAgICAgICAgIGZsb2F0KHN5cy5hcmd2WzRdKSBpZiBsZW4oc3lzLmFyZ3YpID4gNCBlbHNlIDAuOCwKICAgICAgICAgICAgICAgICAgICBmbG9hdChzeXMuYXJndls1XSkgaWYgbGVuKHN5cy5hcmd2KSA+IDUgZWxzZSAzLjUpCg==",
    "face_enhance.py": "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJmYWNlX2VuaGFuY2UucHkg4oCUIEZhY2Utc3BlY2lmaWMgc3VwZXItcmVzb2x1dGlvbiArIGRldGFpbCBmdXNpb24uCiAKQXJjaGl0ZWN0dXJlOgogIElucHV0IGZyYW1lIOKGkiBkZXRlY3QgZmFjZXMgKE9wZW5DViBETk4gLyBIYWFyKSDihpIgY3JvcCBlYWNoIGZhY2UgcmVnaW9uCiAg4oaSIExhbmN6b3MgNHggdXBzY2FsZSDihpIgc2hhcnBlbiDihpIgY29tcG9zaXRlIGJhY2sgYXQgb3JpZ2luYWwgcG9zaXRpb24KICDihpIgRGV0YWlsIGVuaGFuY2VtZW50IG9uIGZ1bGwgZnJhbWUgKGVkZ2UtcHJlc2VydmluZykKIApLZXkgaW5zaWdodDogRmFjZXMgYXJlIH41JSBvZiBmcmFtZSBhcmVhIGJ1dCA4MCUgb2YgcGVyY2VpdmVkIHF1YWxpdHkuCkJ5IHVwc2NhbGluZyBmYWNlcyBhdCA0eCAodnMgMnggZm9yIGJhY2tncm91bmQpLCB3ZSBtYXRjaCBUaXRhbiBmYWNlIGNsYXJpdHkKd2l0aG91dCA0eCB0aGUgVlJBTSBjb3N0LiBBbGwgQ1BVLCBubyBHUFUsIG5vIG1vZGVsIGRvd25sb2Fkcy4KIiIiCmltcG9ydCBjdjIKaW1wb3J0IG51bXB5IGFzIG5wCmltcG9ydCBvcwoKCmRlZiBlbmhhbmNlX2ZyYW1lKGZyYW1lLCBmYWNlX3NjYWxlPTQuMCwgZGV0YWlsX3N0cmVuZ3RoPTAuNik6CiAgICAiIiJBcHBseSBmYWNlLXNwZWNpZmljIHN1cGVyLXJlc29sdXRpb24gKyBkZXRhaWwgZW5oYW5jZW1lbnQuCiAgICAKICAgIEFyZ3M6CiAgICAgICAgZnJhbWU6IFJHQiB1aW50OCBbSCxXLDNdIG9yIEJHUiB1aW50OCBbSCxXLDNdCiAgICAgICAgZmFjZV9zY2FsZTogRmFjZSB1cHNjYWxlIG11bHRpcGxpZXIgKDR4ID0gZmFjZXMgYXQgNHggbmF0aXZlIHJlcykKICAgICAgICBkZXRhaWxfc3RyZW5ndGg6IEVkZ2UtcHJlc2VydmluZyBkZXRhaWwgZW5oYW5jZW1lbnQgKDAtMSkKICAgIAogICAgUmV0dXJuczoKICAgICAgICBFbmhhbmNlZCBmcmFtZSAoc2FtZSBkdHlwZSwgc2FtZSBzaGFwZSkKICAgICIiIgogICAgaCwgdyA9IGZyYW1lLnNoYXBlWzoyXQogICAgaXNfYmdyID0gZnJhbWUuc2hhcGVbMl0gPT0gMyAgIyBPcGVuQ1YgZGVmYXVsdAogICAgCiAgICAjIERldGVjdCBmYWNlcwogICAgZmFjZV9yZWN0cyA9IF9kZXRlY3RfZmFjZXMoZnJhbWUpCiAgICAKICAgIGlmIGxlbihmYWNlX3JlY3RzKSA9PSAwOgogICAgICAgICMgTm8gZmFjZXM6IGp1c3QgZGV0YWlsIGVuaGFuY2VtZW50CiAgICAgICAgcmV0dXJuIF9kZXRhaWxfZW5oYW5jZShmcmFtZSwgZGV0YWlsX3N0cmVuZ3RoKQogICAgCiAgICByZXN1bHQgPSBmcmFtZS5jb3B5KCkKICAgIAogICAgZm9yICh4LCB5LCBmdywgZmgpIGluIGZhY2VfcmVjdHM6CiAgICAgICAgIyBBZGQgbWFyZ2luICg1MCUgZXhwYW5zaW9uIGZvciBjb250ZXh0KQogICAgICAgIG1hcmdpbl94ID0gaW50KGZ3ICogMC4zKQogICAgICAgIG1hcmdpbl95ID0gaW50KGZoICogMC40KQogICAgICAgIHgxID0gbWF4KDAsIHggLSBtYXJnaW5feCkKICAgICAgICB5MSA9IG1heCgwLCB5IC0gbWFyZ2luX3kpCiAgICAgICAgeDIgPSBtaW4odywgeCArIGZ3ICsgbWFyZ2luX3gpCiAgICAgICAgeTIgPSBtaW4oaCwgeSArIGZoICsgbWFyZ2luX3kpCiAgICAgICAgCiAgICAgICAgIyBFeHRyYWN0IGZhY2UgcmVnaW9uCiAgICAgICAgZmFjZV9yb2kgPSBmcmFtZVt5MTp5MiwgeDE6eDJdCiAgICAgICAgaWYgZmFjZV9yb2kuc2l6ZSA9PSAwOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIAogICAgICAgICMgNHggTGFuY3pvcyB1cHNjYWxlIGZvciBmYWNlIG9ubHkKICAgICAgICBmYWNlX2gsIGZhY2VfdyA9IGZhY2Vfcm9pLnNoYXBlWzoyXQogICAgICAgIHVwX2ggPSBpbnQoZmFjZV9oICogZmFjZV9zY2FsZSkKICAgICAgICB1cF93ID0gaW50KGZhY2VfdyAqIGZhY2Vfc2NhbGUpCiAgICAgICAgZmFjZV91cCA9IGN2Mi5yZXNpemUoZmFjZV9yb2ksICh1cF93LCB1cF9oKSwgaW50ZXJwb2xhdGlvbj1jdjIuSU5URVJfTEFOQ1pPUzQpCiAgICAgICAgCiAgICAgICAgIyBTaGFycGVuIHRoZSB1cHNjYWxlZCBmYWNlCiAgICAgICAgZmFjZV91cCA9IF9zbWFydF9zaGFycGVuKGZhY2VfdXAsIHN0cmVuZ3RoPTAuNikKICAgICAgICAKICAgICAgICAjIERvd25zY2FsZSBiYWNrIHRvIG9yaWdpbmFsIHNpemUgd2l0aCBlbmhhbmNlZCBkZXRhaWwKICAgICAgICBmYWNlX2VuaGFuY2VkID0gY3YyLnJlc2l6ZShmYWNlX3VwLCAoZmFjZV93LCBmYWNlX2gpLCBpbnRlcnBvbGF0aW9uPWN2Mi5JTlRFUl9MQU5DWk9TNCkKICAgICAgICAKICAgICAgICAjIEJsZW5kIHdpdGggb3JpZ2luYWwgKGZlYXRoZXIgZWRnZXMgdG8gYXZvaWQgaGFyZCBjdXQpCiAgICAgICAgYmxlbmRfbWFzayA9IF9mZWF0aGVyX21hc2soZmFjZV93LCBmYWNlX2gsIGZlYXRoZXJfcHg9aW50KG1pbihmYWNlX3csIGZhY2VfaCkgKiAwLjE1KSkKICAgICAgICBmb3IgYyBpbiByYW5nZSgzKToKICAgICAgICAgICAgcmVzdWx0W3kxOnkyLCB4MTp4MiwgY10gPSAoZmFjZV9lbmhhbmNlZFs6LDosY10gKiBibGVuZF9tYXNrICsgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJlc3VsdFt5MTp5MiwgeDE6eDIsIGNdICogKDEgLSBibGVuZF9tYXNrKSkKICAgIAogICAgIyBGdWxsLWZyYW1lIGRldGFpbCBlbmhhbmNlbWVudAogICAgcmVzdWx0ID0gX2RldGFpbF9lbmhhbmNlKHJlc3VsdCwgZGV0YWlsX3N0cmVuZ3RoKQogICAgCiAgICByZXR1cm4gbnAuY2xpcChyZXN1bHQsIDAsIDI1NSkuYXN0eXBlKG5wLnVpbnQ4KQoKCmRlZiBfZGV0ZWN0X2ZhY2VzKGZyYW1lKToKICAgICIiIkRldGVjdCBmYWNlcyB1c2luZyBPcGVuQ1YgSGFhciBjYXNjYWRlIChubyBtb2RlbCBkb3dubG9hZHMsIGluc3RhbnQpLgogICAgRmFsbHMgYmFjayB0byBza2luLWNvbG9yIGRldGVjdGlvbiBpZiBIYWFyIGZhaWxzLiIiIgogICAgZ3JheSA9IGN2Mi5jdnRDb2xvcihmcmFtZSwgY3YyLkNPTE9SX0JHUjJHUkFZKSBpZiBmcmFtZS5zaGFwZVsyXSA9PSAzIGVsc2UgZnJhbWUKICAgIAogICAgIyBUcnkgSGFhciBjYXNjYWRlIGZpcnN0IChidWlsdCBpbnRvIE9wZW5DViwgbm8gZG93bmxvYWQpCiAgICBjYXNjYWRlX3BhdGggPSBjdjIuZGF0YS5oYWFyY2FzY2FkZXMgKyAnaGFhcmNhc2NhZGVfZnJvbnRhbGZhY2VfZGVmYXVsdC54bWwnCiAgICBpZiBvcy5wYXRoLmV4aXN0cyhjYXNjYWRlX3BhdGgpOgogICAgICAgIGNhc2NhZGUgPSBjdjIuQ2FzY2FkZUNsYXNzaWZpZXIoY2FzY2FkZV9wYXRoKQogICAgICAgIGZhY2VzID0gY2FzY2FkZS5kZXRlY3RNdWx0aVNjYWxlKGdyYXksIHNjYWxlRmFjdG9yPTEuMSwgbWluTmVpZ2hib3JzPTUsIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtaW5TaXplPSgzMCwgMzApKQogICAgICAgIGlmIGxlbihmYWNlcykgPiAwOgogICAgICAgICAgICByZXR1cm4gW3R1cGxlKGludCh2KSBmb3IgdiBpbiBmKSBmb3IgZiBpbiBmYWNlc10KICAgIAogICAgIyBGYWxsYmFjazogc2tpbi1jb2xvciBiYXNlZCBkZXRlY3Rpb24KICAgIHJldHVybiBfc2tpbl9mYWNlX2RldGVjdGlvbihmcmFtZSkKCgpkZWYgX3NraW5fZmFjZV9kZXRlY3Rpb24oZnJhbWUpOgogICAgIiIiU2ltcGxlIHNraW4tY29sb3IgYmFzZWQgZmFjZSBkZXRlY3Rpb24gZmFsbGJhY2suIiIiCiAgICBpZiBmcmFtZS5zaGFwZVsyXSAhPSAzOgogICAgICAgIHJldHVybiBbXQogICAgaHN2ID0gY3YyLmN2dENvbG9yKGZyYW1lLCBjdjIuQ09MT1JfQkdSMkhTVikKICAgICMgU2tpbiBjb2xvciByYW5nZSAoYnJvYWQpCiAgICBtYXNrID0gY3YyLmluUmFuZ2UoaHN2LCAoMCwgMjAsIDcwKSwgKDIwLCAxNTAsIDI1NSkpCiAgICBtYXNrID0gY3YyLmVyb2RlKG1hc2ssIE5vbmUsIGl0ZXJhdGlvbnM9MikKICAgIG1hc2sgPSBjdjIuZGlsYXRlKG1hc2ssIE5vbmUsIGl0ZXJhdGlvbnM9NCkKICAgIGNvbnRvdXJzLCBfID0gY3YyLmZpbmRDb250b3VycyhtYXNrLCBjdjIuUkVUUl9FWFRFUk5BTCwgY3YyLkNIQUlOX0FQUFJPWF9TSU1QTEUpCiAgICBmYWNlcyA9IFtdCiAgICBmb3IgY250IGluIGNvbnRvdXJzOgogICAgICAgIGFyZWEgPSBjdjIuY29udG91ckFyZWEoY250KQogICAgICAgIGlmIGFyZWEgPCA1MDA6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgeCwgeSwgdywgaCA9IGN2Mi5ib3VuZGluZ1JlY3QoY250KQogICAgICAgICMgRmFjZSBhc3BlY3QgcmF0aW8gfjAuNy0xLjIKICAgICAgICBpZiAwLjUgPCB3L2ggPCAxLjU6CiAgICAgICAgICAgIGZhY2VzLmFwcGVuZCgoeCwgeSwgdywgaCkpCiAgICByZXR1cm4gZmFjZXMKCgpkZWYgX3NtYXJ0X3NoYXJwZW4oaW1nLCBzdHJlbmd0aD0wLjYpOgogICAgIiIiVW5zaGFycCBtYXNrIHdpdGggZWRnZS1hd2FyZSBsaW1pdGluZyAobm8gcmluZ2luZykuIiIiCiAgICBibHVyID0gY3YyLkdhdXNzaWFuQmx1cihpbWcsICgwLCAwKSwgMy4wKQogICAgc2hhcnBlbmVkID0gY3YyLmFkZFdlaWdodGVkKGltZywgMS4wICsgc3RyZW5ndGgsIGJsdXIsIC1zdHJlbmd0aCwgMCkKICAgIHJldHVybiBzaGFycGVuZWQKCgpkZWYgX2RldGFpbF9lbmhhbmNlKGltZywgc3RyZW5ndGg9MC42KToKICAgICIiIkVkZ2UtcHJlc2VydmluZyBkZXRhaWwgZW5oYW5jZW1lbnQgdXNpbmcgT3BlbkNWLiIiIgogICAgIyBBcHBseSBiaWxhdGVyYWwgZmlsdGVyIGZvciBlZGdlLWF3YXJlIHNtb290aGluZwogICAgc21vb3RoID0gY3YyLmJpbGF0ZXJhbEZpbHRlcihpbWcsIDksIDc1LCA3NSkKICAgICMgRGV0YWlsID0gb3JpZ2luYWwgLSBzbW9vdGgKICAgIGRldGFpbCA9IGN2Mi5zdWJ0cmFjdChpbWcuYXN0eXBlKG5wLmludDE2KSwgc21vb3RoLmFzdHlwZShucC5pbnQxNikpCiAgICAjIEJvb3N0IGRldGFpbAogICAgZW5oYW5jZWQgPSBjdjIuYWRkKGltZy5hc3R5cGUobnAuaW50MTYpLCAoZGV0YWlsICogc3RyZW5ndGgpLmFzdHlwZShucC5pbnQxNikpCiAgICByZXR1cm4gbnAuY2xpcChlbmhhbmNlZCwgMCwgMjU1KS5hc3R5cGUobnAudWludDgpCgoKZGVmIF9mZWF0aGVyX21hc2sodywgaCwgZmVhdGhlcl9weD0yMCk6CiAgICAiIiJDcmVhdGUgYSBmZWF0aGVyZWQgYmxlbmRpbmcgbWFzayAoMSBpbnNpZGUsIDAgYXQgZWRnZXMsIHNtb290aCB0cmFuc2l0aW9uKS4iIiIKICAgIG1hc2sgPSBucC5vbmVzKChoLCB3KSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIGlmIGZlYXRoZXJfcHggPD0gMDoKICAgICAgICByZXR1cm4gbWFzawogICAgIyBEaXN0YW5jZSBmcm9tIGVkZ2UKICAgIGRpc3QgPSBucC5vbmVzKChoLCB3KSwgZHR5cGU9bnAuZmxvYXQzMikKICAgIGZvciB5IGluIHJhbmdlKGgpOgogICAgICAgIGZvciB4IGluIHJhbmdlKHcpOgogICAgICAgICAgICBkID0gbWluKHgsIHksIHcgLSAxIC0geCwgaCAtIDEgLSB5KQogICAgICAgICAgICBkaXN0W3ksIHhdID0gZAogICAgIyBTaWdtb2lkIGZlYXRoZXIKICAgIG1hc2sgPSAxLjAgLyAoMS4wICsgbnAuZXhwKC0oZGlzdCAtIGZlYXRoZXJfcHggKiAwLjUpIC8gKGZlYXRoZXJfcHggKiAwLjIpKSkKICAgIHJldHVybiBtYXNrCgoKZGVmIGVuaGFuY2VfdmlkZW8oaW5fcGF0aCwgb3V0X3BhdGgsIGZhY2Vfc2NhbGU9NC4wLCBkZXRhaWxfc3RyZW5ndGg9MC42LCAKICAgICAgICAgICAgICAgICAgdGVtcG9yYWxfc21vb3RoPVRydWUpOgogICAgIiIiUHJvY2VzcyBhbGwgZnJhbWVzIG9mIGEgdmlkZW8gd2l0aCBmYWNlIGVuaGFuY2VtZW50LgogICAgCiAgICBVc2VzIGZmbXBlZyBmb3IgSS9PLCBPcGVuQ1YgZm9yIHByb2Nlc3NpbmcuIEFsbCBDUFUuCiAgICAiIiIKICAgIGltcG9ydCB0ZW1wZmlsZSwgc3VicHJvY2Vzcywgc2h1dGlsLCBnbG9iCiAgICB0bXAgPSB0ZW1wZmlsZS5ta2R0ZW1wKCkKICAgIHRyeToKICAgICAgICAjIEV4dHJhY3QgZnJhbWVzCiAgICAgICAgc3VicHJvY2Vzcy5ydW4oWyJmZm1wZWciLCAiLXkiLCAiLWxvZ2xldmVsIiwgImVycm9yIiwgIi1pIiwgaW5fcGF0aCwKICAgICAgICAgICAgICAgICAgICAgICBvcy5wYXRoLmpvaW4odG1wLCAiZl8lMDZkLnBuZyIpXSwgY2hlY2s9VHJ1ZSkKICAgICAgICBmcmFtZXNfaW4gPSBzb3J0ZWQoZ2xvYi5nbG9iKG9zLnBhdGguam9pbih0bXAsICJmXyoucG5nIikpKQogICAgICAgIGlmIG5vdCBmcmFtZXNfaW46CiAgICAgICAgICAgIHNodXRpbC5jb3B5Mihpbl9wYXRoLCBvdXRfcGF0aCkKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgCiAgICAgICAgIyBHZXQgRlBTCiAgICAgICAgciA9IHN1YnByb2Nlc3MucnVuKFsiZmZwcm9iZSIsICItdiIsICJlcnJvciIsICItc2VsZWN0X3N0cmVhbXMiLCAidjowIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAiLXNob3dfZW50cmllcyIsICJzdHJlYW09cl9mcmFtZV9yYXRlIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAiLW9mIiwgImRlZmF1bHQ9bnc9MTpuaz0xIiwgaW5fcGF0aF0sCiAgICAgICAgICAgICAgICAgICAgICAgICBjYXB0dXJlX291dHB1dD1UcnVlLCB0ZXh0PVRydWUpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBhLCBiID0gci5zdGRvdXQuc3RyaXAoKS5zcGxpdCgiLyIpCiAgICAgICAgICAgIGZwcyA9IGZsb2F0KGEpIC8gZmxvYXQoYikKICAgICAgICBleGNlcHQ6CiAgICAgICAgICAgIGZwcyA9IDI0LjAKICAgICAgICAKICAgICAgICAjIFByb2Nlc3MgZnJhbWVzCiAgICAgICAgZm9yIGksIGZfcGF0aCBpbiBlbnVtZXJhdGUoZnJhbWVzX2luKToKICAgICAgICAgICAgZnJhbWUgPSBjdjIuaW1yZWFkKGZfcGF0aCwgY3YyLklNUkVBRF9DT0xPUikKICAgICAgICAgICAgaWYgZnJhbWUgaXMgTm9uZToKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIAogICAgICAgICAgICAjIEZhY2UgZW5oYW5jZW1lbnQKICAgICAgICAgICAgZW5oYW5jZWQgPSBlbmhhbmNlX2ZyYW1lKGZyYW1lLCBmYWNlX3NjYWxlLCBkZXRhaWxfc3RyZW5ndGgpCiAgICAgICAgICAgIAogICAgICAgICAgICAjIFRlbXBvcmFsIHNtb290aGluZzogYmxlbmQgd2l0aCBwcmV2aW91cyBmcmFtZSdzIGZhY2UgcmVnaW9uCiAgICAgICAgICAgIGlmIHRlbXBvcmFsX3Ntb290aCBhbmQgaSA+IDA6CiAgICAgICAgICAgICAgICBwcmV2ID0gY3YyLmltcmVhZChmcmFtZXNfaW5baS0xXSwgY3YyLklNUkVBRF9DT0xPUikKICAgICAgICAgICAgICAgIGlmIHByZXYgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICAgICAgYWxwaGEgPSAwLjE1ICAjIHN1YnRsZSB0ZW1wb3JhbCBibGVuZAogICAgICAgICAgICAgICAgICAgIGVuaGFuY2VkID0gY3YyLmFkZFdlaWdodGVkKGVuaGFuY2VkLCAxLjAgLSBhbHBoYSwgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcmV2LCBhbHBoYSwgMCkKICAgICAgICAgICAgCiAgICAgICAgICAgIGN2Mi5pbXdyaXRlKGZfcGF0aCwgZW5oYW5jZWQsIFtjdjIuSU1XUklURV9QTkdfQ09NUFJFU1NJT04sIDFdKQogICAgICAgICAgICAKICAgICAgICAgICAgaWYgKGkgKyAxKSAlIDEwMCA9PSAwOgogICAgICAgICAgICAgICAgcHJpbnQoZiJbZmFjZV9lbmhhbmNlXSB7aSsxfS97bGVuKGZyYW1lc19pbil9IGZyYW1lcyIpCiAgICAgICAgCiAgICAgICAgIyBSZS1lbmNvZGUKICAgICAgICBzdWJwcm9jZXNzLnJ1bihbImZmbXBlZyIsICIteSIsICItbG9nbGV2ZWwiLCAiZXJyb3IiLAogICAgICAgICAgICAgICAgICAgICAgICItZnJhbWVyYXRlIiwgZiJ7ZnBzOi40Zn0iLAogICAgICAgICAgICAgICAgICAgICAgICItaSIsIG9zLnBhdGguam9pbih0bXAsICJmXyUwNmQucG5nIiksCiAgICAgICAgICAgICAgICAgICAgICAgIi1jOnYiLCAibGlieDI2NCIsICItcHJlc2V0IiwgInNsb3ciLCAiLWNyZiIsICIxMCIsCiAgICAgICAgICAgICAgICAgICAgICAgIi1waXhfZm10IiwgInl1djQyMHAiLAogICAgICAgICAgICAgICAgICAgICAgICItY29sb3JzcGFjZSIsICJidDcwOSIsICItY29sb3JfcHJpbWFyaWVzIiwgImJ0NzA5IiwgIi1jb2xvcl90cmMiLCAiYnQ3MDkiLAogICAgICAgICAgICAgICAgICAgICAgIG91dF9wYXRoXSwgY2hlY2s9VHJ1ZSkKICAgICAgICBwcmludChmIltmYWNlX2VuaGFuY2VdIERvbmU6IHtsZW4oZnJhbWVzX2luKX0gZnJhbWVzIC0+IHtvdXRfcGF0aH0iKQogICAgZmluYWxseToKICAgICAgICBzaHV0aWwucm10cmVlKHRtcCwgaWdub3JlX2Vycm9ycz1UcnVlKQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBpbXBvcnQgc3lzCiAgICBpZiBsZW4oc3lzLmFyZ3YpIDwgMzoKICAgICAgICBwcmludCgiVXNhZ2U6IGZhY2VfZW5oYW5jZS5weSBJTiBPVVQgW0ZBQ0VfU0NBTEVdIFtERVRBSUxfU1RSRU5HVEhdIikKICAgICAgICBzeXMuZXhpdCgxKQogICAgaW5fcCA9IHN5cy5hcmd2WzFdCiAgICBvdXRfcCA9IHN5cy5hcmd2WzJdCiAgICBmcyA9IGZsb2F0KHN5cy5hcmd2WzNdKSBpZiBsZW4oc3lzLmFyZ3YpID4gMyBlbHNlIDQuMAogICAgZHMgPSBmbG9hdChzeXMuYXJndls0XSkgaWYgbGVuKHN5cy5hcmd2KSA+IDQgZWxzZSAwLjYKICAgIGVuaGFuY2VfdmlkZW8oaW5fcCwgb3V0X3AsIGZzLCBkcykK",
}
for _fn, _b in _FIXED_FILES.items():
    _p = os.path.join(_SRC_DIR, _fn)
    with open(_p, "wb") as _f:
        _f.write(_b64.b64decode(_b))
    print("[patch] wrote corrected %s (%d bytes)" % (_fn, os.path.getsize(_p)))

rm = "/kaggle/working/src/render_motion.py"
_s = open(rm).read()
if "KIT_SHOT_START" not in _s:
    anchor = 'shots = load_shots(root, cfg)["shots"]'
    assert anchor in _s, "anchor not found in render_motion.py"
    inject = anchor + "\n" + \
        '    _st = int(os.environ.get("KIT_SHOT_START", "1")) - 1\n' + \
        '    _ed = int(os.environ.get("KIT_SHOT_END", str(len(shots))))\n' + \
        '    shots = shots[_st:_ed]\n' + \
        '    print("[progressive chunk] Sliced shot list to range %d - %d (%d shots)" % (_st+1, _ed, len(shots)))\n' + \
        '    _sn = int(os.environ.get("KIT_SHARD_COUNT", "1"))\n' + \
        '    _si = int(os.environ.get("KIT_SHARD_INDEX", "0"))\n' + \
        '    if _sn > 1:\n' + \
        '        shots = [s for _k, s in enumerate(shots) if _k % _sn == _si]\n' + \
        '        print("[shard] GPU %d/%d assigned %d shots from this chunk" % (_si, _sn, len(shots)))'
    _s = _s.replace(anchor, inject, 1)
    open(rm, "w").write(_s)
    print("[patch] render_motion.py: progressive scene chunking & sharding enabled")

# ==============================================================================
# STEP 3.5: AUDIO ENGINES SETUP
# ==============================================================================
print("\n--- 3.5 AUDIO SETUP ---")
def _sh(cmd): return subprocess.run(cmd, capture_output=True, text=True)

_hf = _sh([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"])
if _hf.returncode == 0: os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("MG_MODEL", "facebook/musicgen-medium")

try: _sh(["apt-get", "-y", "-qq", "install", "espeak-ng"])
except Exception: pass

_sh([sys.executable, "-m", "pip", "install", "-q", "kokoro>=0.9.2", "soundfile", "misaki[en]"])

# ==============================================================================
# STEP 8: LAUNCH TWO COMFYUI SERVERS (ONE PER T4)
# ==============================================================================
print("\n--- 8. STARTING TWO COMFYUI SERVERS ---")
subprocess.run(['pkill', '-f', 'ComfyUI/main.py'], capture_output=True)
time.sleep(3)

# --- SAFE GPU GUARD -------------------------------------------------------
# Each ComfyUI server pins ~90% of system RAM (~28.9 GB of 32 GB). Launching
# one server per GPU (T4 x2) means two servers on ONE machine -> >32 GB RAM ->
# kernel OOM restart. Default to a SINGLE server; only allow more if BOTH the
# GPU count AND the host RAM can actually hold them (~29 GB pinned per server).
_ALL_GPU_PORTS = [(0, 8188), (1, 8189)]
try:
    import torch as _torch
    _avail = _torch.cuda.device_count()
except Exception:
    _avail = 1
try:
    import psutil as _ps
    _ram_gb = _ps.virtual_memory().total / 1e9
except Exception:
    _ram_gb = 32.0
_want = int(os.environ.get("KIT_NUM_GPUS", "1"))          # default 1 = safe
_safe_by_ram = max(1, int(_ram_gb // 29))                 # ~29 GB pinned/server
_ngpu = max(1, min(_want, _avail, _safe_by_ram, len(_ALL_GPU_PORTS)))
if _want > _ngpu:
    print("[gpu-guard] Requested %d GPU server(s) but capping to %d "
          "(avail=%d, ram=%.0fGB, ~29GB pinned/server). This prevents the "
          "host-RAM OOM caused by two ComfyUI servers on one machine."
          % (_want, _ngpu, _avail, _ram_gb))
GPU_PORTS = _ALL_GPU_PORTS[:_ngpu]
print("[gpu-guard] Launching %d ComfyUI server(s): %s" % (len(GPU_PORTS), GPU_PORTS))
# --------------------------------------------------------------------------
server_procs = []; server_logs = []
for gpu, port in GPU_PORTS:
    outdir = "/kaggle/working/ComfyUI/output_gpu%d" % gpu
    os.makedirs(outdir, exist_ok=True)
    userdir = "/kaggle/working/ComfyUI/user_gpu%d" % gpu
    os.makedirs(userdir, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    lf = open("/kaggle/working/comfyui_gpu%d.log" % gpu, "w")
    p = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port),
         "--output-directory", outdir, "--user-directory", userdir, "--disable-smart-memory"],
        cwd=str(COMFY), stdout=lf, stderr=subprocess.STDOUT, env=env)
    server_procs.append((p, gpu, port)); server_logs.append(lf)
    ready = False
    print("Waiting for ComfyUI on GPU%d / port %d..." % (gpu, port))
    for elapsed in range(1, 240):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/system_stats" % port, timeout=1) as resp:
                if resp.status == 200:
                    ready = True; print("✓ GPU%d online (:%d)" % (gpu, port)); break
        except Exception: time.sleep(1)
    if not ready:
        lf.close()
        raise RuntimeError("ComfyUI on GPU%d / port %d failed to start." % (gpu, port))

comfy_proc = server_procs[0][0]

# ==============================================================================
# STEP 8.5: SCHEMA-RESOLVED WORKFLOW BUILDER (LENGTH 31 FOR T4 STABILITY)
# ==============================================================================
print("\n--- 8.5 RESOLVING LTX-VIDEO SCHEMA ---")
COMFY_API = "http://127.0.0.1:8188"
def _object_info(cls):
    with urllib.request.urlopen(f"{COMFY_API}/object_info/{cls}", timeout=30) as r: data = json.load(r)
    return data[cls]
def _req(info): return info["input"]["required"]
def _opt(info): return info["input"].get("optional", {})
def _outs(info): return info.get("output_name") or info.get("output") or []

# === WAN2.2 14B I2V ENGINE (MoE high/low expert + 4-step lightx2v LoRA, GGUF) ===
KIT_ENGINE = os.environ.get("KIT_ENGINE", "ltxv")
_ENGINE_BUILT = "ltxv"
if KIT_ENGINE == "wan":
    print("\n--- 8.5w RESOLVING WAN2.2 14B I2V SCHEMA ---")
    try:
        _wi2v = _object_info("WanImageToVideo")
        for _n in ("UnetLoaderGGUF", "LoraLoaderModelOnly", "ModelSamplingSD3", "KSamplerAdvanced", "VAELoader", "CLIPLoader", "VAEDecode"):
            _object_info(_n)
        _wreq = _req(_wi2v); _wopt = _opt(_wi2v); _wall = {**_wreq, **_wopt}
        if "clip_vision_output" in _wreq:
            raise RuntimeError("WanImageToVideo requires clip_vision_output; not provisioned")
        _wo = [str(o).lower() for o in _outs(_wi2v)]
        def _wslot(name, dflt):
            return next((i for i, n in enumerate(_wo) if name in n), dflt)
        WPOS, WNEG, WLAT = _wslot("positive", 0), _wslot("negative", 1), _wslot("latent", 2)
        WLEN_KEY = next((k for k in ("length", "num_frames", "frames") if k in _wall), "length")
        WIMG_KEY = next((k for k in ("start_image", "image", "ref_image") if k in _wall), "start_image")
        _wtypes = _req(_object_info("CLIPLoader"))["type"][0]
        WCLIP_TYPE = next((o for o in _wtypes if "wan" in str(o).lower()), _wtypes[0])
        _msd3_keys = {**_req(_object_info("ModelSamplingSD3")), **_opt(_object_info("ModelSamplingSD3"))}
        _ksa_keys = {**_req(_object_info("KSamplerAdvanced")), **_opt(_object_info("KSamplerAdvanced"))}
        WAN_HIGH = os.environ["WAN_HIGH"]; WAN_LOW = os.environ["WAN_LOW"]
        WAN_UMT5 = os.environ["WAN_UMT5"]; WAN_VAE = os.environ["WAN_VAE"]
        WAN_LORA_HIGH = os.environ.get("WAN_LORA_HIGH", ""); WAN_LORA_LOW = os.environ.get("WAN_LORA_LOW", "")
        WAN_SHIFT = float(os.environ.get("KIT_WAN_SHIFT", "5.0"))
        LENGTH = int(os.environ.get("KIT_WAN_LENGTH", "41"))
        WW = int(os.environ.get("KIT_WAN_W", "768")); WH = int(os.environ.get("KIT_WAN_H", "512"))
        graph = {
            "1":  {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_HIGH}},
            "21": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_LOW}},
            "2":  {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_UMT5, "type": WCLIP_TYPE}},
            "3":  {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
            "4":  {"class_type": "CLIPTextEncode", "inputs": {"text": "POSITIVE", "clip": ["2", 0]}},
            "5":  {"class_type": "CLIPTextEncode", "inputs": {"text": "NEGATIVE", "clip": ["2", 0]}},
            "6":  {"class_type": "LoadImage", "inputs": {"image": "man_anchor.png"}},
        }
        _hi_model = ["1", 0]; _lo_model = ["21", 0]
        if WAN_LORA_HIGH:
            graph["20"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": WAN_LORA_HIGH, "strength_model": 1.0}}
            _hi_model = ["20", 0]
        if WAN_LORA_LOW:
            graph["22"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["21", 0], "lora_name": WAN_LORA_LOW, "strength_model": 1.0}}
            _lo_model = ["22", 0]
        _ms_hi = {"model": _hi_model, "shift": WAN_SHIFT}
        _ms_lo = {"model": _lo_model, "shift": WAN_SHIFT}
        graph["7"] = {"class_type": "ModelSamplingSD3", "inputs": {k: v for k, v in _ms_hi.items() if k in _msd3_keys or k == "model"}}
        graph["23"] = {"class_type": "ModelSamplingSD3", "inputs": {k: v for k, v in _ms_lo.items() if k in _msd3_keys or k == "model"}}
        _wi = {"positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0], WIMG_KEY: ["6", 0], "width": WW, "height": WH, WLEN_KEY: LENGTH, "batch_size": 1}
        graph["14"] = {"class_type": "WanImageToVideo", "inputs": {k: v for k, v in _wi.items() if k in _wall or k in ("positive", "negative", "vae")}}
        _TOTAL = int(os.environ.get("KIT_WAN_STEPS", "4")); _BOUND = max(1, _TOTAL // 2)
        _CFG = float(os.environ.get("KIT_WAN_CFG", "1.0"))
        _ks_hi = {"model": ["7", 0], "add_noise": "enable", "noise_seed": 42, "steps": _TOTAL, "cfg": _CFG, "sampler_name": "euler", "scheduler": "simple", "positive": ["14", WPOS], "negative": ["14", WNEG], "latent_image": ["14", WLAT], "start_at_step": 0, "end_at_step": _BOUND, "return_with_leftover_noise": "enable"}
        _ks_lo = {"model": ["23", 0], "add_noise": "disable", "noise_seed": 42, "steps": _TOTAL, "cfg": _CFG, "sampler_name": "euler", "scheduler": "simple", "positive": ["14", WPOS], "negative": ["14", WNEG], "latent_image": ["15", 0], "start_at_step": _BOUND, "end_at_step": _TOTAL, "return_with_leftover_noise": "disable"}
        graph["15"] = {"class_type": "KSamplerAdvanced", "inputs": {k: v for k, v in _ks_hi.items() if k in _ksa_keys}}
        graph["24"] = {"class_type": "KSamplerAdvanced", "inputs": {k: v for k, v in _ks_lo.items() if k in _ksa_keys}}
        graph["16"] = {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["3", 0]}}
        graph["17"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["16", 0], "frame_rate": 16, "loop_count": 0, "filename_prefix": "wan_i2v_take", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 18, "save_output": True, "pingpong": False, "save_image": True}}
        graph["_bind"] = {"positive": ["4", "text"], "negative": ["5", "text"], "width": ["14", "width"], "height": ["14", "height"], "seconds": ["14", WLEN_KEY], "seed": ["15", "noise_seed"], "ref_dirs": ["6", "image"]}  # DESYNC FIX: steps/cfg NOT bound for Wan MoE; both samplers keep build-time _TOTAL split (KIT_WAN_STEPS) + _CFG. Per-shot steps override would clobber only high sampler and desync the 2+2 schedule -> pure noise.
        _ENGINE_BUILT = "wan"
        print("[graph] WAN2.2 14B I2V MoE wired: GGUF high(1)+low(21) | lora H=%s L=%s | %d steps split@%d | cfg %.1f | %dx%d len %d" % (bool(WAN_LORA_HIGH), bool(WAN_LORA_LOW), _TOTAL, _BOUND, _CFG, WW, WH, LENGTH))
    except Exception as _we:
        _ENGINE_BUILT = "ltxv"
        print("[graph] WAN build unavailable (%s); falling back to LTXV engine." % _we)
if _ENGINE_BUILT != "wan":
    i2v = _object_info("LTXVImgToVideo")
    _object_info("CheckpointLoaderSimple"); _object_info("ModelSamplingLTXV")
    _object_info("VAEDecode"); _object_info("KSampler")
    cl = _object_info("CLIPLoader")
    type_opts = _req(cl)["type"][0]
    CLIP_TYPE = next((o for o in type_opts if "ltxv" in o.lower() or "ltx" in o.lower()), type_opts[0])

    outs = [o.lower() for o in _outs(i2v)]
    def _slot(name, dflt): return next((i for i, n in enumerate(outs) if name in n), dflt)
    POS, NEG, LAT = _slot("positive", 0), _slot("negative", 1), _slot("latent", 2)
    i2v_in = {**_req(i2v), **_opt(i2v)}
    LEN_KEY = next((k for k in ("length", "num_frames", "frames") if k in i2v_in), "length")
    IMG_KEY = next((k for k in ("image", "start_image", "ref_image") if k in i2v_in), "image")

    LENGTH = 41
    # LTXV temporal conditioning (frame_rate) -- REQUIRED for coherent motion.
    # Without it LTXV keeps frame 0 sharp but decays later frames to gray mush.
    _HAS_LTXV_COND = False
    try:
        _cond_info = _object_info("LTXVConditioning")
        _cond_in = {**_req(_cond_info), **_opt(_cond_info)}
        FR_KEY = next((k for k in ("frame_rate", "fps", "frames_per_second") if k in _cond_in), "frame_rate")
        _HAS_LTXV_COND = True
        print(f"[schema] LTXVConditioning available; frame_rate key = {FR_KEY}")
    except Exception as _e:
        print(f"[schema] LTXVConditioning NOT available ({_e}); keeping direct conditioning.")
    LTXV_FPS = float(os.environ.get("KIT_LTXV_FPS", "25"))
    _HAS_LTXV_SAMPLER = False
    try:
        _object_info("SamplerCustom"); _object_info("LTXVScheduler"); _object_info("KSamplerSelect")
        _HAS_LTXV_SAMPLER = True
        print("[schema] SamplerCustom + LTXVScheduler + KSamplerSelect available.")
    except Exception as _e:
        print(f"[schema] LTXV custom sampler chain NOT available ({_e}); using KSampler.")
    graph = {
        "1":  {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT_NAME}},
        "2":  {"class_type": "CLIPLoader", "inputs": {"clip_name": T5_NAME, "type": CLIP_TYPE}},
        "4":  {"class_type": "CLIPTextEncode", "inputs": {"text": "POSITIVE", "clip": ["2", 0]}},
        "5":  {"class_type": "CLIPTextEncode", "inputs": {"text": "NEGATIVE", "clip": ["2", 0]}},
        "6":  {"class_type": "LoadImage", "inputs": {"image": "man_anchor.png"}},
        "7":  {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": 1.3, "base_shift": 0.95}},
        "14": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "vae": ["1", 2], IMG_KEY: ["6", 0], "width": 768, "height": 512, LEN_KEY: LENGTH, "strength": 1.0, "batch_size": 1}},
        "15": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["14", POS], "negative": ["14", NEG], "latent_image": ["14", LAT], "steps": int(os.environ.get("KIT_MOTION_STEPS","20")), "cfg": float(os.environ.get("KIT_LTXV_CFG","3.0")), "sampler_name": "euler", "scheduler": "simple", "seed": 42, "denoise": 1.0}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["1", 2]}},
        "17": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["16", 0], "frame_rate": 12, "loop_count": 0, "filename_prefix": "ltxv_i2v_take", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 18, "save_output": True, "pingpong": False, "save_image": True}},
        "_bind": {"positive": ["4", "text"], "negative": ["5", "text"], "width": ["14", "width"], "height": ["14", "height"], "seconds": ["14", LEN_KEY], "steps": ["15", "steps"], "cfg": ["15", "cfg"], "seed": ["15", "seed"], "ref_dirs": ["6", "image"]},
    }
    graph["14"]["inputs"] = {k: v for k, v in graph["14"]["inputs"].items() if k in (set(i2v_in) | {"positive", "negative", "vae"})}
    if _HAS_LTXV_COND:
        graph["8"] = {"class_type": "LTXVConditioning", "inputs": {"positive": ["14", POS], "negative": ["14", NEG], FR_KEY: LTXV_FPS}}
        graph["15"]["inputs"]["positive"] = ["8", 0]
        graph["15"]["inputs"]["negative"] = ["8", 1]
        print(f"[graph] wired LTXVConditioning (node 8) at {LTXV_FPS} fps -> KSampler pos/neg.")
    if _HAS_LTXV_SAMPLER:
        # LTXV is a flow-matching model: KSampler(simple) leaves temporal frames undenoised (gray).
        # Replace with the canonical LTXVScheduler + SamplerCustom chain for real motion.
        _pos_src = ["8", 0] if _HAS_LTXV_COND else ["14", POS]
        _neg_src = ["8", 1] if _HAS_LTXV_COND else ["14", NEG]
        _steps = int(os.environ.get("KIT_MOTION_STEPS", "20"))
        _sched_keys = {**_req(_object_info("LTXVScheduler")), **_opt(_object_info("LTXVScheduler"))}
        _sched_all = {"steps": _steps, "max_shift": 1.3, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["14", LAT]}
        graph["9"] = {"class_type": "LTXVScheduler", "inputs": {k: v for k, v in _sched_all.items() if k in _sched_keys}}
        graph["10"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        _sc_keys = {**_req(_object_info("SamplerCustom")), **_opt(_object_info("SamplerCustom"))}
        _sc_all = {"model": ["7", 0], "add_noise": True, "noise_seed": 42, "cfg": float(os.environ.get("KIT_LTXV_CFG", "3.0")), "positive": _pos_src, "negative": _neg_src, "sampler": ["10", 0], "sigmas": ["9", 0], "latent_image": ["14", LAT]}
        graph["15"] = {"class_type": "SamplerCustom", "inputs": {k: v for k, v in _sc_all.items() if k in _sc_keys}}
        graph["16"]["inputs"]["samples"] = ["15", 0]
        graph["_bind"]["steps"] = ["9", "steps"]
        graph["_bind"]["cfg"] = ["15", "cfg"]
        graph["_bind"]["seed"] = ["15", "noise_seed"]
        print(f"[graph] LTXV sampling chain wired: LTXVScheduler(9)+KSamplerSelect(10)+SamplerCustom(15), steps={_steps}.")
    else:
        print("[graph] LTXV SamplerCustom chain unavailable; kept KSampler(15).")
# STEP 8.6: SPATIAL UPSCALER PASS -- near-HD crispness after VAE decode
try:
    _object_info("UpscaleModelLoader"); _object_info("ImageUpscaleWithModel")
    _HAS_UPSCALE = True
except Exception as _ue:
    _HAS_UPSCALE = False
    print("[graph] upscale nodes unavailable (%s); keeping native res." % _ue)
os.environ.setdefault("KIT_TEST_NO_UPSCALE", "1")  # TEST DEFAULT: upscale OFF -> render stays 768x512 so the CPU postprocess (hqdn3d denoise + grade_match) fits in the ~3GB RAM left after ComfyUI pins ~28.9GB. Set "0" for HD 1536x1024 only once RAM headroom is solved.
if _HAS_UPSCALE and UPSCALE_MODEL_NAME and os.environ.get("KIT_TEST_NO_UPSCALE") != "1":
    graph["18"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL_NAME}}
    graph["19"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["18", 0], "image": ["16", 0]}}
    graph["17"]["inputs"]["images"] = ["19", 0]
    print("[graph] spatial upscaler wired: UpscaleModelLoader(18)+ImageUpscaleWithModel(19) via %s -> VideoCombine." % UPSCALE_MODEL_NAME)
else:
    print("[graph] spatial upscaler NOT wired (model/nodes missing); VideoCombine uses native VAEDecode output.")
wf_dir = Path("/kaggle/working/workflows"); wf_dir.mkdir(exist_ok=True)
for name in ("flf2v", "phantom", "s2v"): (wf_dir / f"{name}.json").write_text(json.dumps(graph, indent=2))

# ==============================================================================
# STEP 9: FORGE + DUAL-GPU CHUNK RENDER + AUTOMATED EXPORT & DISK CLEANUP
# ==============================================================================
# --- RESUME: pull clips rendered in previous sessions back into the work dir ---
# (finished shots then get SKIPPED, so re-running continues where you left off)
import glob as _glob
_wkdir = '/kaggle/working/confrontation_work'
os.makedirs(_wkdir, exist_ok=True)
_ids = {s['id'] for s in json.load(open(os.path.join(KIT_ROOT, 'shots_confrontation.json')))['shots']}
_restored = 0
if os.path.isdir('/kaggle/input'):
    for _f in _glob.glob('/kaggle/input/**/final.mp4', recursive=True):
        _sid = os.path.basename(os.path.dirname(_f))
        _dst = os.path.join(_wkdir, _sid, 'final.mp4')
        if _sid in _ids and os.path.getsize(_f) > 1024 and not os.path.exists(_dst):
            os.makedirs(os.path.dirname(_dst), exist_ok=True)
            shutil.copy2(_f, _dst); _restored += 1
print('[resume] restored %d finished clip(s) from previous sessions.' % _restored)

print(f"\n--- 9. EXECUTING REAL GPU CHUNK: SHOTS {os.environ['KIT_SHOT_START']} TO {os.environ['KIT_SHOT_END']} ---")
# --- FORCE CLEAN this chunk so stale mock/placeholder clips can NEVER be silently reused ---
# render_motion.py skips any shot whose confrontation_work/<sid>/final.mp4 already exists, and
# /kaggle/working survives a kernel Restart. Without this, a prior mock run's placeholders get
# re-zipped and no prompt ever reaches ComfyUI. We clear ONLY the current chunk's shot dirs.
if os.environ.get('KIT_FORCE_RERENDER', '1') == '1':
    import shutil as _sh
    _wk = '/kaggle/working/confrontation_work'
    _allshots = json.load(open(os.path.join(KIT_ROOT, 'shots_confrontation.json')))['shots']
    _a = int(os.environ['KIT_SHOT_START']); _b = int(os.environ['KIT_SHOT_END'])
    _targets = _allshots[_a - 1:_b]
    _cleared = 0
    for _s in _targets:
        _d = os.path.join(_wk, _s['id'])
        if os.path.isdir(_d):
            _sh.rmtree(_d, ignore_errors=True); _cleared += 1
            print(f"[force] cleared stale render dir: {_s['id']}")
    print(f"[force] re-render armed for shots {_a}..{_b} ({len(_targets)} shot(s), {_cleared} cleared); no cached final.mp4 will be reused.")

SRC = os.path.join(KIT_ROOT, "src")
def _stage(title, *cmd, env=None, fatal=True, timeout=None):
    print("\n===== %s =====" % title, flush=True)
    r = subprocess.run(list(cmd), cwd=KIT_ROOT, env=env or {**os.environ})
    if r.returncode != 0 and fatal: raise RuntimeError(f"Stage '{title}' failed rc={r.returncode}")
    return r.returncode

# ===== MERGE: NO-COMPROMISE VERIFY GATE (oneclick playbook) =====
# Abort BEFORE any GPU spend unless the proven fixes are all in place.
def _verify_or_die():
    import json as _vj
    problems = []
    _wf = "/kaggle/working/workflows/flf2v.json"
    try:
        _g = _vj.load(open(_wf))
        _cts = {n.get("class_type") for n in _g.values() if isinstance(n, dict)}
        if not ({"WanImageToVideo", "UnetLoaderGGUF"} & _cts):
            problems.append("flf2v.json is not the inline WAN I2V graph (missing WanImageToVideo/UnetLoaderGGUF)")
        _blob = _vj.dumps(_g).lower()
        for _bad in ("wan2.2-flf2v", "wan2.2-s2v", "wan2.2-phantom"):
            if _bad in _blob:
                problems.append("flf2v.json references un-downloaded static model '%s'" % _bad)
    except Exception as _e:
        problems.append("cannot read %s (%s)" % (_wf, _e))
    try:
        if "KIT_SHOT_START" not in open(os.path.join(SRC, "render_motion.py")).read():
            problems.append("render_motion.py missing KIT_SHOT_START range patch")
    except Exception as _e:
        problems.append("cannot read render_motion.py (%s)" % _e)
    for _need in (os.environ.get("KIT_CONFIG", "config_confrontation.yaml"), "shots_confrontation.json"):
        if not os.path.exists(os.path.join(KIT_ROOT, _need)):
            problems.append("missing %s" % _need)
    if problems:
        raise RuntimeError("VERIFY GATE FAILED (no GPU spent):\n  - " + "\n  - ".join(problems))
    print("[merge] VERIFY GATE PASSED: inline WAN graph + range patch + story assets present; shots %s-%s." % (os.environ.get("KIT_SHOT_START"), os.environ.get("KIT_SHOT_END")))
_verify_or_die()

# ===== MERGE: KAGGLE-COMPATIBLE AUTO-FORGE (SDXL ensure) =====
# forge_character.py runs charforge.json (an SDXL text->image graph). On Kaggle
# the SDXL checkpoint is never downloaded, so forge 400s. In auto mode we fetch
# it under the EXACT name charforge.json expects so the existing forge stage
# (below) invents jack/julie. ComfyUI rescans models/checkpoints on dir-mtime
# change, so a running server picks it up.
if GODSHAND_FORGE == "auto":
    _ckpt_name = os.environ.get("KIT_FORGE_CKPT", "sd_xl_base_1.0.safetensors")
    _ckpt_dir  = COMFY / "models" / "checkpoints"; _ckpt_dir.mkdir(parents=True, exist_ok=True)
    _ckpt = _ckpt_dir / _ckpt_name
    if not (_ckpt.exists() and _ckpt.stat().st_size > 1_000_000_000):
        from huggingface_hub import hf_hub_download as _hf
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1"); os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("[merge] auto-forge: downloading SDXL base (~6.9GB, first run only)...", flush=True)
        _lp = Path(_hf(os.environ.get("KIT_FORGE_REPO", "stabilityai/stable-diffusion-xl-base-1.0"),
                       filename=os.environ.get("KIT_FORGE_REPO_FILE", "sd_xl_base_1.0.safetensors"),
                       local_dir=str(_ckpt_dir), token=os.environ.get("HF_TOKEN") or None))
        if _lp.resolve() != _ckpt.resolve():
            shutil.move(str(_lp), str(_ckpt))
        for _c in _ckpt_dir.rglob(".cache"):
            shutil.rmtree(_c, ignore_errors=True)
    print("[merge] auto-forge: SDXL checkpoint ready -> %s" % _ckpt)
    _cg = os.path.join(KIT_ROOT, "workflows", "charforge.json")
    if not os.path.exists(_cg):
        subprocess.run([sys.executable, os.path.join(SRC, "make_workflows.py")], cwd=KIT_ROOT, check=False)
    try:
        _cgj = json.load(open(_cg)); _chg = False
        for _n in _cgj.values():
            if isinstance(_n, dict) and _n.get("class_type") == "CheckpointLoaderSimple" and _n.get("inputs", {}).get("ckpt_name") != _ckpt_name:
                _n["inputs"]["ckpt_name"] = _ckpt_name; _chg = True
        if _chg:
            json.dump(_cgj, open(_cg, "w"), indent=2); print("[merge] auto-forge: charforge.json ckpt_name -> %s" % _ckpt_name)
        else:
            print("[merge] auto-forge: charforge.json already targets %s" % _ckpt_name)
    except Exception as _e:
        print("[merge] auto-forge: charforge.json align skipped (%s)" % _e)
else:
    print("[merge] identity via REAL anchors -> forge_character.py will skip bound leads.")

try:
    _stage("preflight (free)", sys.executable, os.path.join(SRC, "preflight.py"), fatal=False)
    _stage("forge character identities", sys.executable, os.path.join(SRC, "forge_character.py"))

    print("\n--- 9b. PARALLEL RENDERING ACROSS BOTH T4 GPUs ---")
    render_procs = []
    for idx, (gpu, port) in enumerate(GPU_PORTS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "COMFY_PORT": str(port),
               "COMFY_OUTPUT_DIR": "/kaggle/working/ComfyUI/output_gpu%d" % gpu,
               "KIT_SHARD_INDEX": str(idx), "KIT_SHARD_COUNT": str(len(GPU_PORTS))} 
        lf = open("/kaggle/working/render_shard%d.log" % idx, "w")
        p = subprocess.Popen([sys.executable, os.path.join(SRC, "render_motion.py"), "--max-hours", os.environ.get("KIT_MAX_HOURS", "11")], cwd=KIT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        render_procs.append((p, lf, idx, gpu))

    while any(p.poll() is None for p, _, _, _ in render_procs):
        time.sleep(15) # Checked less frequently in real GPU mode
        for p, _, idx, gpu in render_procs:
            try: _last = open("/kaggle/working/render_shard%d.log" % idx).read().splitlines()[-1:]
            except Exception: _last = []
            _state = "running" if p.poll() is None else ("done rc=%d" % p.returncode)
            print("  [shard %d / GPU%d] %s  %s" % (idx, gpu, _state, _last[0] if _last else ""))

    for p, lf, idx, gpu in render_procs:
        lf.close()
        if p.returncode != 0: raise RuntimeError(f"Render shard {idx} failed rc={p.returncode}")
    print("[ok] Both shards finished rendering this scene chunk.")

    # --- CUMULATIVE EXPORT + PROGRESS STATUS (hands-off) ---
    print('==================================================')
    print('  STEP 10: PACKAGING PROGRESS & CHECKING COMPLETION ')
    print('==================================================')
    _wkdir = '/kaggle/working/confrontation_work'
    _allshots = json.load(open(os.path.join(KIT_ROOT, 'shots_confrontation.json')))['shots']
    def _fin(_s): return os.path.join(_wkdir, _s['id'], 'final.mp4')
    _done = [s for s in _allshots if os.path.exists(_fin(s)) and os.path.getsize(_fin(s)) > 1024]
    _done_ids = {s['id'] for s in _done}
    _remaining = [s['id'] for s in _allshots if s['id'] not in _done_ids]
    _total = len(_allshots); _ndone = len(_done); _complete = (_ndone == _total)
    export_dir = '/kaggle/working/_progress_export'
    shutil.rmtree(export_dir, ignore_errors=True); os.makedirs(export_dir, exist_ok=True)
    for s in _done:
        _cd = os.path.join(export_dir, 'clips', s['id']); os.makedirs(_cd, exist_ok=True)
        shutil.copy2(_fin(s), os.path.join(_cd, 'final.mp4'))
    if _done:
        _lst = os.path.join(export_dir, '_preview.txt')
        with open(_lst, 'w') as _fh:
            for s in _done:
                _fh.write('file ' + repr(_fin(s)) + '\n')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',_lst,'-c:v','libx264','-pix_fmt','yuv420p',os.path.join(export_dir,'progress_preview.mp4')], check=False)
    if _complete:
        print('[auto] all %d shots done -> building final master (stitch.py)...' % _total)
        _r = subprocess.run([sys.executable, os.path.join(SRC, 'stitch.py'), '--root', KIT_ROOT], cwd=KIT_ROOT)
        _outd = os.path.join(KIT_ROOT, 'output')
        if _r.returncode == 0 and os.path.isdir(_outd):
            for _mp in _glob.glob(os.path.join(_outd, '*.mp4')):
                shutil.copy2(_mp, export_dir)
    _msg = 'Rendered: %d / %d shots\n\n' % (_ndone, _total)
    if _complete:
        _msg += 'ALL DONE. The final movie is the master .mp4 in this zip. Nothing else to run.\n'
    else:
        _msg += 'NOT finished. To continue next session:\n'
        _msg += '  1) Add THIS run Output as an Input (Add Input -> Your Work -> Notebook Output)\n'
        _msg += '  2) Run All. It resumes automatically.\n\n'
        _msg += 'Remaining (%d): %s\n' % (len(_remaining), ', '.join(_remaining))
    open(os.path.join(export_dir, 'STATUS.txt'), 'w').write(_msg)
    zip_name = '/kaggle/working/Beholder_progress'
    if os.path.exists(zip_name + '.zip'): os.remove(zip_name + '.zip')
    shutil.make_archive(zip_name, 'zip', export_dir)
    print('[PACK] Beholder_progress.zip  (%d/%d shots)' % (_ndone, _total))
    if _complete:
        print('ALL DONE -> download Beholder_progress.zip; it holds the final movie.')
    else:
        print('MORE TO GO -> just RE-RUN next session (add this Output as an Input first).')
        print('Remaining shots: %d' % len(_remaining))
    print('Purging raw ComfyUI render caches to free scratch disk...')
    for gpu, _ in GPU_PORTS:
        shutil.rmtree(f'/kaggle/working/ComfyUI/output_gpu{gpu}', ignore_errors=True)
    print('Scratch disk clean (finished clips kept for resume).')

finally:
    for _p, _gpu, _port in server_procs:
        try: _p.terminate()
        except Exception: pass
    for _lf in server_logs:
        try: _lf.close()
        except Exception: pass
    gc.collect(); torch.cuda.empty_cache()