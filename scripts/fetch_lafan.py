"""Fetch original Ubisoft BVH archive once; keep its license and provenance."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("destination", type=Path)
a = p.parse_args()
a.destination.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/ubisoft/ubisoft-laforge-animation-dataset/master/"
url = "https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip"
archive = a.destination / "lafan1.zip"
if not archive.exists():
    temp = archive.with_suffix(".tmp")
    urllib.request.urlretrieve(url, temp)
    temp.replace(archive)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        target = (a.destination / info.filename).resolve()
        if not target.is_relative_to(a.destination.resolve()):
            raise ValueError("Unsafe archive path")
        if not target.exists():
            z.extract(info, a.destination)
for name in ["README.md", "license.txt"]:
    try:
        urllib.request.urlretrieve(base + name, a.destination / name)
    except urllib.error.HTTPError:
        pass
(a.destination / "provenance.json").write_text(
    json.dumps(
        {"url": url, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
        indent=2,
    )
)
print(a.destination, len(list(a.destination.rglob("*.bvh"))))
