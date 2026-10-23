"""Record a generated v2 demo's deployed asset hashes."""

import argparse, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
files = {}
for f in sorted((a.root / "assets").rglob("*")):
    if f.is_file():
        h = hashlib.sha256()
        with f.open("rb") as s:
            for block in iter(lambda: s.read(1024 * 1024), b""):
                h.update(block)
        files["dist/" + str(f.relative_to(a.root))] = h.hexdigest()
a.output.write_text(
    json.dumps(
        {"format": 2, "learner_version": 2, "motion_split": "test37", "files": files},
        indent=2,
    )
    + "\n"
)
print(len(files), "assets locked")
