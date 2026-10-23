"""Summarize stable iteration wall time and overlapping nvidia-smi samples."""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from statistics import mean, median

p = argparse.ArgumentParser()
p.add_argument("--gpu-log", type=Path, required=True)
p.add_argument("--run", action="append", required=True, help="path:GPU-index")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
samples = []
with a.gpu_log.open() as f:
    for row in csv.DictReader(f):
        row = {k.strip(): v.strip() for k, v in row.items()}
        try:
            samples.append(
                (
                    dt.datetime.strptime(
                        row["timestamp"], "%Y/%m/%d %H:%M:%S.%f"
                    ).timestamp(),
                    int(row["index"]),
                    float(row["utilization.gpu [%]"].split()[0]),
                    float(row["power.draw [W]"].split()[0]),
                )
            )
        except (ValueError, KeyError):
            continue
results = []
for spec in a.run:
    path, gpu = spec.rsplit(":", 1)
    path, gpu = Path(path), int(gpu)
    config = json.loads((path / "run.json").read_text())
    lines = [json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines()]
    # First two full updates may include graph specialization; discard through iteration 3.
    steady = [
        x
        for x in lines
        if x["iteration"] >= 4 and x["iteration"] != config.get("profile_iteration")
    ]
    if not steady:
        results.append(
            {"run": path.name, "completed": False, "error": "No stable iterations"}
        )
        continue
    inferred_start = (path / "metrics.jsonl").stat().st_mtime - lines[-1]["elapsed_s"]
    windows = []
    for x in steady:
        end = x.get("timestamp", inferred_start + x["elapsed_s"])
        windows.append(
            (end - sum(x[k] for k in ("collect_s", "world_s", "policy_s")), end)
        )
    util = [
        s
        for s in samples
        if s[1] == gpu and any(lo <= s[0] <= hi for lo, hi in windows)
    ]
    phases = {
        k: mean(x[k] for x in steady) for k in ("collect_s", "world_s", "policy_s")
    }
    total = sum(phases.values())
    results.append(
        {
            "run": path.name,
            "commit": config["git_commit"],
            "precision": config.get(
                "precision", "bf16" if config.get("amp") == "bf16" else "tf32"
            ),
            "completed": (path / "COMPLETED.json").exists(),
            "gpu": gpu,
            "iterations": len(steady),
            "mean_s": total,
            "iterations_per_s": 1 / total,
            **phases,
            "median_s": median(sum(x[k] for k in phases) for x in steady),
            "gpu_util_mean_pct": mean(s[2] for s in util) if util else None,
            "power_mean_W": mean(s[3] for s in util) if util else None,
            "gpu_samples": len(util),
            "peak_actor_grad_norm": max(x.get("actor_grad_norm", 0) for x in lines),
            "final": lines[-1],
        }
    )
a.output.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
