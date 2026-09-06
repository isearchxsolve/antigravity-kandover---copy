import json

nb_path = r"C:\Users\Admin\Downloads\antigravity-kandover\rainman-wan-MERGED-TUNED.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

source = "".join(nb["cells"][1]["source"])
lines = source.split("\n")

for i, l in enumerate(lines):
    if "WAN_LORA_HIGH" in l:
        print(f"Line {i+1}: {l.strip()}")
