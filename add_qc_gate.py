import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(BASE, 'rainman-wan-MERGED-TUNED.ipynb')

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

code = ''.join(notebook['cells'][1]['source'])

old_stitch_section = """    if _complete:
        print('[auto] all %d shots done -> building final master (stitch.py)...' % _total)
        _r = subprocess.run([sys.executable, os.path.join(SRC, 'stitch.py'), '--root', KIT_ROOT], cwd=KIT_ROOT)
        _outd = os.path.join(KIT_ROOT, 'output')"""

new_stitch_section = """    if _complete:
        # --- STAGE 7/14: PER-SHOT QUALITY GATE (qc.py) ---
        print('[auto] running per-shot quality gate (qc.py)...')
        _r = subprocess.run([sys.executable, os.path.join(SRC, 'qc.py'), '--work-dir', _wkdir], cwd=KIT_ROOT)
        if _r.returncode != 0:
            raise RuntimeError('QC GATE FAILED: one or more shots did not meet quality thresholds. Check logs above.')

        print('[auto] all %d shots done -> building final master (stitch.py)...' % _total)
        _r = subprocess.run([sys.executable, os.path.join(SRC, 'stitch.py'), '--root', KIT_ROOT], cwd=KIT_ROOT)
        _outd = os.path.join(KIT_ROOT, 'output')"""

code = code.replace(old_stitch_section, new_stitch_section)

old_post_stitch = """        if _r.returncode == 0 and os.path.isdir(_outd):
            for _mp in _glob.glob(os.path.join(_outd, '*.mp4')):
                shutil.copy2(_mp, export_dir)
    _msg = 'Rendered: %d / %d shots"""

new_post_stitch = """        if _r.returncode == 0 and os.path.isdir(_outd):
            for _mp in _glob.glob(os.path.join(_outd, '*.mp4')):
                shutil.copy2(_mp, export_dir)

            # --- STAGE 14/14: FINAL MASTER QC ---
            _master_files = _glob.glob(os.path.join(_outd, '*.mp4'))
            if _master_files:
                _master = _master_files[0]
                print(f'[auto] running final master QC on {_master}...')
                _r_qc = subprocess.run([sys.executable, os.path.join(SRC, 'qc.py'), '--clips', '--work-dir', _outd], cwd=KIT_ROOT, capture_output=True, text=True)
                if _r_qc.returncode != 0:
                    print(f'[WARN] Master QC issues: {_r_qc.stdout} {_r_qc.stderr}')
                else:
                    print('[auto] master QC passed')
    _msg = 'Rendered: %d / %d shots"""

code = code.replace(old_post_stitch, new_post_stitch)

notebook['cells'][1]['source'] = code.splitlines(keepends=True)

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("✓ Added QC gate calls to MERGED-TUNED notebook")

code = ''.join(notebook['cells'][1]['source'])
for c in ['qc.py', '--work-dir', 'PER-SHOT QUALITY GATE', 'FINAL MASTER QC']:
    if c in code:
        print(f'  ✓ {c}')
    else:
        print(f'  ✗ {c} MISSING')