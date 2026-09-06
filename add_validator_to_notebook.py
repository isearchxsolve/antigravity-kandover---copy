import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(BASE, 'rainman-wan-MERGED-TUNED.ipynb')

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    data = json.load(f)

code = ''.join(data['cells'][1]['source'])

old_preflight = '_stage("preflight (free)", sys.executable, os.path.join(SRC, "preflight.py"), fatal=False)'

new_preflight = '''# --- STAGE 0: RUNTIME INTEGRATION VALIDATOR (prevents VideoHelperSuite/frame extraction bugs) ---
    _stage("runtime integration validator", sys.executable, os.path.join(SRC, "runtime_integration_validator.py"), fatal=True)
    _stage("preflight (free)", sys.executable, os.path.join(SRC, "preflight.py"), fatal=False)'''

if old_preflight in code:
    code = code.replace(old_preflight, new_preflight)
    print("✓ Added runtime integration validator to preflight stage")
else:
    print("✗ Could not find preflight stage to replace")

data['cells'][1]['source'] = code.splitlines(keepends=True)

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("✓ Notebook updated")