"""Fetch the official A3 rigid-foot retarget model and its referenced meshes."""

import argparse
import hashlib
import json
import posixpath
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "c9df753d31ef63c930b825b1b6767cabd37c1549"
BASE = f"https://raw.githubusercontent.com/AgibotTech/sonic_for_a3/{REVISION}/"
XML = "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5.xml"
NOTICES = [
    "LICENSE",
    "NOTICE",
    "legal/THIRD-PARTY SOFTWARE NOTICES AND ASSET LICENSES - GEAR-SONIC.txt",
]


def fetch(output):
    if output.exists():
        raise FileExistsError(output)
    with urllib.request.urlopen(BASE + XML, timeout=60) as response:
        raw = response.read()
    tree = ET.fromstring(raw)
    if tree.findall("include"):
        raise ValueError(
            "Unexpected model includes; explicit asset resolution required"
        )
    compiler = tree.find("compiler")
    meshdir = posixpath.join(posixpath.dirname(XML), compiler.get("meshdir", ""))
    paths = {
        posixpath.normpath(posixpath.join(meshdir, m.attrib["file"]))
        for m in tree.findall("./asset/mesh")
    }
    paths.update(NOTICES)
    output.mkdir(parents=True)
    target = output / XML
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)

    def download(name):
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("Asset path escapes output directory")
        with urllib.request.urlopen(
            BASE + urllib.parse.quote(name), timeout=120
        ) as response:
            data = response.read()
        if data.startswith(b"version https://git-lfs.github.com/spec"):
            expected = next(
                line.split("sha256:", 1)[1]
                for line in data.decode().splitlines()
                if line.startswith("oid sha256:")
            )
            media = f"https://media.githubusercontent.com/media/AgibotTech/sonic_for_a3/{REVISION}/"
            with urllib.request.urlopen(
                media + urllib.parse.quote(name), timeout=120
            ) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"LFS checksum mismatch: {name}")
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return name, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(download, sorted(paths)))
    hashes[XML] = hashlib.sha256(raw).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": "AgibotTech/sonic_for_a3",
                "commit": REVISION,
                "xml": XML,
                "sha256": hashes,
                "usage": "rigid-foot kinematic retarget asset; not the 035 passive-foot training model",
            },
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    fetch(p.parse_args().output)
