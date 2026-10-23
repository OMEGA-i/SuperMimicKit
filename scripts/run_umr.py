"""Run the pinned UMR in its own environment; outputs stay per robot/clip."""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UMR = ROOT / "third_party/UMR"


def stage_model(source_xml, directory):
    """Keep UMR-generated XML local and avoid its legacy /assets/ path rewrite."""
    directory.mkdir()
    tree = ET.parse(source_xml)
    root = tree.getroot()
    if root.findall("include"):
        raise ValueError("Flatten MJCF includes before staging a UMR model")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for kind, attribute in [("mesh", "meshdir"), ("texture", "texturedir")]:
        original = (source_xml.parent / compiler.get(attribute, "")).resolve()
        folder = directory / ("meshes" if kind == "mesh" else "textures")
        folder.mkdir()
        compiler.set(attribute, folder.name)
        for index, element in enumerate(root.findall(f"./asset/{kind}")):
            if "file" not in element.attrib:
                continue
            target = (original / element.attrib["file"]).resolve()
            if not target.is_file():
                raise FileNotFoundError(target)
            link = folder / f"{index}{target.suffix}"
            link.symlink_to(target)
            element.set("file", link.name)
    staged = directory / source_xml.name
    tree.write(staged, encoding="unicode")
    return staged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["g1", "h2", "a3"], required=True)
    p.add_argument(
        "--motion", type=Path, required=True, help="LaFAN converted to SMPL-X NPZ"
    )
    p.add_argument("--smplx-model-dir", type=Path, required=True)
    p.add_argument("--xml", type=Path, help="Required for A3")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--slots", type=Path, help="Reuse existing correspondence slots")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.start < 0 or (args.end != -1 and args.end <= args.start):
        p.error("Invalid frame interval")
    if args.slots is not None and not args.slots.is_file():
        p.error("Correspondence slots file is missing")
    if args.max_frames < 0:
        p.error("--max-frames must be nonnegative")
    if not (UMR / "scripts/humanoid_retarget_pipeline.py").is_file():
        p.error("Initialize third_party/UMR with git submodule update --init")
    motion = args.motion.resolve()
    if not motion.is_file() or motion.suffix != ".npz":
        p.error("--motion must be an existing SMPL-X NPZ, not raw BVH")
    model_dir = args.smplx_model_dir.resolve()
    if not any(
        (model_dir / f"SMPLX_NEUTRAL.{ext}").is_file() for ext in ["npz", "pkl"]
    ):
        p.error("--smplx-model-dir must contain the licensed SMPLX_NEUTRAL model")
    cfg = json.loads((ROOT / f"configs/umr/{args.robot}.json").read_text())
    if "upstream_config" in cfg:
        source = UMR / cfg["upstream_config"]
        cfg = json.loads(source.read_text())
        if "xml" in cfg["robot"]:
            cfg["robot"]["xml"] = str((source.parent / cfg["robot"]["xml"]).resolve())
    if "xml" not in cfg["robot"] and args.xml is None:
        p.error("--xml is required for A3")
    if args.xml is not None:
        if args.robot in {"g1", "h2"}:
            p.error("G1/H2 use the official UMR model; --xml is only for A3")
        cfg["robot"]["xml"] = str(args.xml.resolve())
    if not Path(cfg["robot"]["xml"]).is_file():
        p.error("Robot MJCF is missing")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Official robots use the upstream asset and XML policy directly. Only
    # external A3 assets need staging to resolve their supplier mesh paths.
    source_xml = Path(cfg["robot"]["xml"])
    source_xml_sha = hashlib.sha256(source_xml.read_bytes()).hexdigest()
    staged_xml = (
        source_xml if args.robot in {"g1", "h2"}
        else stage_model(source_xml, out / "model")
    )
    cfg["robot"]["xml"] = str(staged_xml)
    cfg["smplx_model_dir"] = str(model_dir)
    cfg["motion"] = {
        "data": str(motion.parent),
        "seq_key": motion.stem,
        "max_frames": args.max_frames,
        "start": args.start,
        "end": args.end,
    }
    cfg["view"] = {"enabled": False}
    correspondence = cfg.setdefault("correspondence", {})
    if args.slots is not None:
        correspondence["slots"] = str(args.slots.resolve())
    correspondence.setdefault("dataset", {}).update(
        out=str(out / "correspondence.npz"),
        smpl_models=[{"type": "smplx", "dir": str(model_dir), "genders": ["neutral"]}],
    )
    correspondence.setdefault("train", {})["out_dir"] = str(out / "correspondence")
    cfg["retarget"] = {"out": str(out / "retarget.npz")}
    config_path = out / "umr_config.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UMR, text=True
    ).strip()
    metadata = {
        "umr_commit": revision,
        "robot": cfg["robot"]["name"],
        "motion": str(motion),
        "motion_sha256": hashlib.sha256(motion.read_bytes()).hexdigest(),
        "source_xml": str(source_xml),
        "source_xml_sha256": source_xml_sha,
        "xml_sha256": hashlib.sha256(staged_xml.read_bytes()).hexdigest(),
        "status": "prepared",
        "visual_qc": "pending",
        "physics_qc": "pending",
    }
    manifest = out / "provenance.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.prepare_only:
        return
    try:
        with (out / "umr.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(UMR / "scripts/humanoid_retarget_pipeline.py"),
                    "--config",
                    str(config_path),
                    "--skip-view",
                    "--stage",
                    "retarget" if args.slots is not None else "all",
                ],
                cwd=UMR,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if not (out / "retarget.npz").is_file():
            raise RuntimeError("UMR exited without a result")
        metadata["status"] = "retarget_process_completed"
        metadata["result_sha256"] = hashlib.sha256(
            (out / "retarget.npz").read_bytes()
        ).hexdigest()
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
