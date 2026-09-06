import json
import os
BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(BASE, 'rainman-wan-MERGED-TUNED.ipynb')

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

code = ''.join(notebook['cells'][1]['source'])

# Fix 1: Update KIT_FAST_POST from '1' to '0' (in the merged section)
code = code.replace(
    "os.environ['KIT_FAST_POST']   = '1'           # skip 60fps CPU interpolation per shot;",
    "os.environ['KIT_FAST_POST']   = '0'           # RESTORED: was '1' and silently skipped color grade + face-SR;"
)

# Fix 2: Add missing tuned config variables after KIT_FAST_POST
# Find the location after KIT_FAST_POST comment
import re
# The pattern: after the KIT_FAST_POST line and its comment, before KIT_SHOT_START
old_pattern = """os.environ['KIT_FAST_POST']   = '0'           # RESTORED: was '1' and silently skipped color grade + face-SR;
                                               # use upscale_hd_kaggle.ipynb on CPU after download
os.environ['KIT_SHOT_START']"""

new_pattern = """os.environ['KIT_FAST_POST']   = '0'           # RESTORED: was '1' and silently skipped color grade + face-SR;
                                               # use upscale_hd_kaggle.ipynb on CPU after download

# === TUNED FINAL ROUND: direct answer to the field observations + the stills-first blueprint ===
os.environ['KIT_FLF2V_ENDFRAME'] = '1'    # OBS #2 (no real motion): chain each still into the next -> real movement, not a frozen pose
os.environ['KIT_MOTION_DENOISE'] = '0.60' # BLUEPRINT (let WAN exercise full freedom): more generative motion (cfg default 0.55)
os.environ['KIT_FACE_ENHANCE']   = '1'    # OBS #1 (take_0 sharper than final): cheap face super-res finishing pass
# interpolate_fps:16 in config already skips the costly per-shot interpolation via the NEED_INTERP guard,
# so KIT_FAST_POST is no longer needed for speed -- color grade + face-SR now run and quality is preserved.

os.environ['KIT_SHOT_START']"""

code = code.replace(old_pattern, new_pattern)

# Fix 3: Add pre-render disk check before the render section
# Find "--- 9. EXECUTING REAL GPU CHUNK"
pre_render_idx = code.find("print(f\"\\n--- 9. EXECUTING REAL GPU CHUNK")
if pre_render_idx >= 0:
    pre_render_check = """\n# --- PRE-RENDER DISK CHECK ---
check_disk_space(min_gb=3)
cleanup_kaggle_working()
gpu_memory_cleanup()
check_disk_space(min_gb=3)
"""
    code = code[:pre_render_idx] + pre_render_check + code[pre_render_idx:]

# Fix 4: Add final disk check before the final cleanup
# Find "Purging raw ComfyUI render caches"
final_cleanup_idx = code.find("print('Purging raw ComfyUI render caches to free scratch disk...')")
if final_cleanup_idx >= 0:
    final_check = """\n# --- FINAL DISK CHECK ---
check_disk_space()
cleanup_kaggle_working()
gpu_memory_cleanup()
check_disk_space()
"""
    code = code[:final_cleanup_idx] + final_check + code[final_cleanup_idx:]

# Update the notebook
notebook['cells'][1]['source'] = code.splitlines(keepends=True)

# Write back
with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook patched successfully!")

# Verify
code = ''.join(notebook['cells'][1]['source'])
for key in ["KIT_FLF2V_ENDFRAME", "KIT_MOTION_DENOISE", "KIT_FACE_ENHANCE", "KIT_FAST_POST", "PRE-RENDER DISK CHECK", "FINAL DISK CHECK"]:
    if key in code:
        print(f"  FOUND: {key}")
    else:
        print(f"  MISSING: {key}")