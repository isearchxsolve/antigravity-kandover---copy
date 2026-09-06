import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(BASE, 'rainman-wan-MERGED-TUNED.ipynb')

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

code = ''.join(notebook['cells'][1]['source'])

insert_after = "os.environ['KIT_FAST_POST'] = '0'  # RESTORED: color grade + face-SR now run\n"
idx = code.find(insert_after)
if idx < 0:
    insert_after = "os.environ['KIT_FAST_POST'] = '0'"
    idx = code.find(insert_after)
if idx < 0:
    print("✗ Could not find insertion point (KIT_FAST_POST not found)")

if idx >= 0:
    insert_pos = idx + len(insert_after)
    oom_settings = "\n# --- OOM PROOFING for kaggle_2xt4_max (2xT4 16GB) ---\n"
    oom_settings += "os.environ['CUDA_LAUNCH_BLOCKING']      = '0'          # async kernel launch\n"
    oom_settings += "os.environ['TOKENIZERS_PARALLELISM']    = 'false'      # avoid fork warnings\n"
    oom_settings += "os.environ['PYTORCH_CUDA_ALLOC_CONF']   = 'expandable_segments:True,max_split_size_mb:128'\n"
    new_code = code[:insert_pos] + oom_settings + code[insert_pos:]
    notebook['cells'][1]['source'] = new_code.splitlines(keepends=True)

    with open(NOTEBOOK, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print("✓ Added missing OOM settings")
else:
    print("✗ Could not find insertion point")

code = ''.join(notebook['cells'][1]['source'])
for term in ['CUDA_LAUNCH_BLOCKING', 'TOKENIZERS_PARALLELISM', 'PYTORCH_CUDA_ALLOC_CONF']:
    if term in code:
        print(f"  ✓ {term} now present")
    else:
        print(f"  ✗ {term} still missing")