"""Retarget SMPL-X clips using an existing absolute-path robot run config."""

import argparse
import copy
import fcntl
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = json.loads(args.config.read_text())
    # Explicit absolute paths avoid rebasing a supplier's relative asset references.
    for value in [config["robot"]["xml"], config["smplx_model_dir"]]:
        if not Path(value).is_absolute() or not Path(value).exists():
            parser.error("Use an existing run config with absolute XML and model paths")
    corr = config["correspondence"]
    slots = Path(corr["slots"]) if "slots" in corr else Path(corr["train"]["out_dir"]) / "correspondence_slots_final.npz"
    if not slots.is_absolute() or not slots.is_file():
        parser.error(
            "Train correspondence first; an existing absolute slots path is required"
        )
    sources = sorted(args.source.resolve().rglob(args.pattern))
    if not sources:
        parser.error("No matching SMPL-X clips")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    lock = (output / ".runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {
        "status": "running",
        "expected_clips": len(sources),
        "umr_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "config_sha256": digest(args.config),
        "slots_sha256": digest(slots),
        "clips": [],
    }
    previous = output / "dataset.json"
    if args.resume and previous.exists():
        old = json.loads(previous.read_text())
        for key in ("config_sha256", "slots_sha256", "expected_clips"):
            if old[key] != report[key]:
                raise ValueError(f"Resume contract changed: {key}")
        report["previous_umr_commit"] = old["umr_commit"]
        report["clips"] = [c for c in old["clips"] if c["status"] == "completed"]
    completed = {c["directory"]: c for c in report["clips"]}
    if len(completed) != len(report["clips"]):
        raise ValueError("Duplicate completed clips")
    pending = []
    known = set()
    for source in sources:
        relative = source.relative_to(args.source.resolve()).with_suffix("")
        key = str(relative)
        known.add(key)
        if key in completed:
            entry = completed[key]
            if entry["source_sha256"] != digest(source) or entry["result_sha256"] != digest(output / relative / "retarget.npz"):
                raise ValueError(f"Completed clip changed: {relative}")
        else:
            pending.append((source, relative))
    if not set(completed).issubset(known):
        raise ValueError("Completed source missing from input snapshot")
    report["workers"] = args.workers

    def save_report():
        report["clips"].sort(key=lambda c: c["directory"])
        temporary = output / "dataset.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "dataset.json")

    def run_clip(source, relative):
        directory = output / relative
        if directory.exists():
            archive = output / ".interrupted" / (str(relative).replace("/", "_") + "-" + uuid.uuid4().hex[:8])
            archive.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(archive)
        directory.mkdir(parents=True)
        cfg = copy.deepcopy(config)
        cfg["correspondence"]["slots"] = str(slots)
        cfg["motion"] = {
            "data": str(source.parent),
            "seq_key": source.stem,
            "start": 0,
            "end": -1,
            "stride": 1,
            "max_frames": 0,
        }
        # Upstream writes a floating MJCF beside the source XML using the output
        # basename. Give each clip a unique name to avoid concurrent XML writes.
        basename = "retarget_" + hashlib.sha256(str(relative).encode()).hexdigest()[:16]
        working_result = directory / (basename + ".npz")
        cfg.setdefault("retarget", {})["out"] = str(working_result)
        cfg["view"] = {"enabled": False}
        path = directory / "config.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        entry = {
            "source": str(source),
            "source_sha256": digest(source),
            "directory": str(relative),
        }
        with (directory / "umr.log").open("w") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("humanoid_retarget_pipeline.py")),
                    "--config",
                    str(path),
                    "--stage",
                    "retarget",
                    "--skip-view",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result = working_result
        entry["returncode"] = process.returncode
        entry["status"] = (
            "completed" if process.returncode == 0 and result.is_file() else "failed"
        )
        if entry["status"] == "completed":
            entry["result_sha256"] = digest(result)
        if entry["status"] == "completed":
            result.rename(directory / "retarget.npz")
        return entry

    save_report()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_clip, source, relative) for source, relative in pending]
        for future in as_completed(futures):
            entry = future.result()
            report["clips"].append(entry)
            save_report()
            print(f"{len(report['clips'])}/{len(sources)} {entry['directory']}: {entry['status']}", flush=True)
    report["status"] = (
        "completed"
        if all(c["status"] == "completed" for c in report["clips"])
        else "failed"
    )
    save_report()
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
