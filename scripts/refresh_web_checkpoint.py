"""Atomically refresh v2 WebGPU inference, preserving the held-out test motions."""

import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import mujoco as mj
import torch
from supermimic.core_v2 import Actor, Layout, WorldModel
from supermimic.kinematics import Kinematics
from scripts.export_web_demo import inference_metadata, package_model
from scripts.export_web_onnx import export_networks, rollout_fixture
from supermimic.mjwarp import build_model


def refresh(checkpoint_path, assets, publish=True):
    torch.set_num_threads(2)
    meta_path = assets / "metadata.json"
    meta = json.loads(meta_path.read_text())
    ckpt = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    if meta.get("learner_version") != 2 or ckpt.get("learner_version") != 2:
        raise ValueError("Legacy checkpoints retired; re-export v2 assets first")
    if (
        ckpt["joint_names"] != meta["joint_names"]
        or ckpt["body_names"] != meta["body_names"]
    ):
        raise ValueError("Robot schema mismatch")
    iteration = int(ckpt["iteration"]) + 1
    training_run = str(checkpoint_path.resolve().parent)
    if meta.get("inference_backend") == "onnxruntime-webgpu-f32" and meta.get("training_run") == training_run and meta.get("completed_iterations", 0) >= iteration:
        print(json.dumps({"robot": meta["robot"], "status": "unchanged", "iterations": iteration}), flush=True)
        return
    layout = Layout(len(meta["joint_names"]), len(meta["body_names"]))
    actor = Actor(layout, ckpt["args"]["small"]).eval()
    actor.load_state_dict(ckpt["actor"])
    asset_hash = ckpt['training_contract']['asset_sha256']
    if asset_hash != meta.get('source_asset_sha256'):
        run = json.loads((checkpoint_path.parent / 'run.json').read_text())
        source_path = run.get('asset_path')
        if not source_path:
            # Older runs predate asset_path metadata; use only the exact live trainer.
            pid = int(checkpoint_path.parent.with_suffix('.pid').read_text())
            proc = Path(f'/proc/{pid}')
            argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            if b'supermimic.train' not in argv or os.fsencode(checkpoint_path.parent) not in argv:
                raise ValueError('Cannot resolve asset from the active training process')
            env = dict(x.split('=', 1) for x in proc.joinpath('environ').read_text().split('\0') if '=' in x)
            source_path = env[ckpt['config']['asset_env']]
        source = Path(source_path)
        if hashlib.sha256(source.read_bytes()).hexdigest() != asset_hash:
            raise ValueError('Demo physical asset does not match checkpoint')
        os.environ[ckpt['config']['asset_env']] = str(source)
        model, _ = build_model(ckpt['config'], ckpt['joint_names'])
        model_prefix = 'models/' + asset_hash
        model_dir = assets / model_prefix
        model_dir.mkdir(parents=True, exist_ok=True)
        packaged = package_model(model, model_dir)
        packaged['xml'] = model_prefix + '/' + packaged['xml']
        packaged['files'] = [model_prefix + '/' + name for name in packaged['files']]
        meta.update(model=packaged, source_asset_sha256=asset_hash)
    else:
        model = mj.MjModel.from_xml_path(str(assets / meta["model"]["xml"]))
    world = WorldModel(
        layout,
        layout.joints + 4,
        Kinematics(model, meta["joint_names"], meta["body_names"]),
        width=ckpt["world"]["dynamics.0.weight"].shape[0],
    ).eval()
    world.load_state_dict(ckpt["world"])
    iteration = int(ckpt["iteration"]) + 1
    stamp = datetime.now(timezone.utc)
    prefix = f"checkpoints/{iteration}-{stamp.strftime('%Y%m%dT%H%M%S%f')}"
    out = assets / prefix
    out.mkdir(parents=True, exist_ok=False)
    networks = export_networks(out, actor, world)
    for net in networks.values():
        net['file'] = prefix + '/' + net['file']
    fixture = json.loads((assets / meta["inference_fixture"]).read_text())
    with torch.no_grad():
        h = torch.tensor(fixture["history"]).reshape(1, 10, layout.obs_dim)
        g = torch.tensor(fixture["goals"]).reshape(1, 11, layout.dim)
        s = torch.tensor(fixture["state"]).reshape(1, layout.dim)
        c = torch.tensor(meta["context"])[None]
        action = actor(h, g)
        n = world(s, action, c)
    fixture.update(action=action.flatten().tolist(), world_next=n.flatten().tolist())
    fixture['rollout'] = rollout_fixture(actor, world, fixture, meta['context'], meta['default_pose'])
    fixture_name = prefix + "/inference_fixture.json"
    (assets / fixture_name).write_text(json.dumps(fixture))
    meta.update(
        training_run=training_run,
        checkpoint_stage="full-training",
        onnx=networks,
        inference_backend="onnxruntime-webgpu-f32",
        weight_storage="float32 ONNX",
        **inference_metadata(actor, world),
        inference_fixture=fixture_name,
        completed_iterations=iteration,
        exported_at=stamp.isoformat(),
    )
    meta.pop("networks", None)
    temporary = assets / "metadata.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2))
    subprocess.run([str(Path(sys.executable).parent / "node"), str(Path(__file__).with_name("check_web_v2.mjs")), str(assets), str(temporary)], check=True)
    if publish:
        os.replace(temporary, meta_path)
    print(
        json.dumps(
            {
                "robot": meta["robot"],
                "iterations": iteration,
                "test_motions": len(meta["motions"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    refresh(a.checkpoint, a.assets, publish=not a.prepare_only)
