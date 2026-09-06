import json
import os
BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(BASE, 'rainman-wan-MERGED-TUNED.ipynb')

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    data = json.load(f)

code = ''.join(data['cells'][1]['source'])

# The problem: final disk check code is between try block end and finally
# Need to move it inside the try block

# Find the section to fix - from "# --- FINAL DISK CHECK ---" to before "finally:"
old_section = """# --- FINAL DISK CHECK ---
check_disk_space()
cleanup_kaggle_working()
gpu_memory_cleanup()
check_disk_space()
print('Purging raw ComfyUI render caches to free scratch disk...')
    for gpu, _ in GPU_PORTS:
        shutil.rmtree(f'/kaggle/working/ComfyUI/output_gpu{gpu}', ignore_errors=True)
    print('Scratch disk clean (finished clips kept for resume).')

finally:"""

new_section = """    # --- FINAL DISK CHECK (inside try block) ---
    check_disk_space()
    cleanup_kaggle_working()
    gpu_memory_cleanup()
    check_disk_space()
    print('Purging raw ComfyUI render caches to free scratch disk...')
    for gpu, _ in GPU_PORTS:
        shutil.rmtree(f'/kaggle/working/ComfyUI/output_gpu{gpu}', ignore_errors=True)
    print('Scratch disk clean (finished clips kept for resume).')

finally:"""

if old_section in code:
    code = code.replace(old_section, new_section)
    print("✓ Fixed final disk check placement")
else:
    print("✗ Could not find section to fix")
    # Try alternative - maybe indentation is different
    import re
    # Find the pattern more flexibly
    match = re.search(r'# --- FINAL DISK CHECK ---.*?finally:', code, re.DOTALL)
    if match:
        print("Found with regex")
        print(match.group()[:500])

data['cells'][1]['source'] = code.splitlines(keepends=True)

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Notebook saved")