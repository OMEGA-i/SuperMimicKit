"""Verify relocated canonical payloads against their existing dataset checksums."""

import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not any(x.is_symlink() for x in a.root.rglob("*")), (
    "Canonical data still contains links"
)
report = {}
for dataset in ("lafan", "grail"):
    root = a.root / dataset
    manifest = json.loads((root / "manifest.json").read_text())
    counts = {}
    for robot, info in manifest["robots"].items():
        for clip in info["clips"]:
            motion = root / clip["directory"] / "motion.npz"
            assert (
                hashlib.sha256(motion.read_bytes()).hexdigest() == clip["motion_sha256"]
            ), str(motion)
            meta = json.loads(motion.with_name("metadata.json").read_text())
            source = Path(meta["source"])
            assert source.is_file() and a.root in source.parents, str(source)
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest() == meta["source_sha256"]
            ), str(source)
            if meta.get("scene_required"):
                assert (root / meta["scene"]).exists(), meta["scene"]
        counts[robot] = len(info["clips"])
    report[dataset] = counts
split = json.loads((a.root / "lafan/manifest.json").read_text())["splits"]
assert len(split["train"]) == 40 and len(split["test"]) == 37
assert not set(split["train"]) & set(split["test"])
report.update(symlinks=0, checksums_match=True, split={"train": 40, "test": 37})
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report))
