import json
import os

workspace = r"C:\Users\Admin\Downloads\antigravity-kandover"
notebook_path = os.path.join(workspace, "rainman-wan-MERGED-TUNED.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][1]
lines = cell["source"]

for idx, l in enumerate(lines):
    if "def _dl(repo, fn, sub):" in l:
        print(f"Found _dl at index {idx+1}")
        lines.insert(idx + 1, "            if os.environ.get('COMFY_MOCK') == '1': return fn\n")
        break

source = "".join(lines)
compile(source, "MERGED-TUNED-CELL1", "exec")
print("[OK] Notebook cell 1 compiles 100% cleanly!")

cell["source"] = lines

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Successfully saved rainman-wan-MERGED-TUNED.ipynb with clean _dl mock bypass")
