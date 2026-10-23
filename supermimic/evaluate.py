"""Deterministic per-clip simulation evaluation; saves actual and reference trajectories."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--backend", choices=("mujoco-warp", "physx"), default="mujoco-warp")
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--backend", default="mujoco-warp")
    if selector.parse_known_args()[0].backend == "physx":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    else:
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--headless", action="store_true")
    args = p.parse_args()
    app = AppLauncher(args) if args.backend == "physx" else None
    try:
        run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if app is not None:
            app.app.close()


def run(args):
    import numpy as np
    import torch

    from .core_v2 import Actor, Layout, VERSION
    from .data import Motions, select_motion_paths
    if args.backend == "physx":
        from .physx import PhysX as Simulator
    else:
        from .mjwarp import MJWarp as Simulator

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    if checkpoint.get("learner_version") != VERSION:
        raise ValueError("Legacy checkpoints are retired; use v2")
    motions = Motions(select_motion_paths(sorted(args.motions.rglob("motion.npz")), "test37"), args.device)
    if motions.names != (checkpoint["joint_names"], checkpoint["body_names"]):
        raise ValueError("Checkpoint asset schema mismatch")
    if checkpoint.get("training_contract", {}).get("backend", "physx") != args.backend:
        raise ValueError("Checkpoint simulation backend mismatch")
    env = Simulator(checkpoint["config"], motions.names, 1, args.device)
    env.layout = Layout(env.layout.joints, env.layout.bodies)
    actor = Actor(env.layout, checkpoint["args"]["small"]).to(args.device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for ci, clip in enumerate(motions.clips):
        ids = torch.tensor([ci], device=args.device)
        time = torch.zeros(1, device=args.device, dtype=torch.long)
        env.reset(torch.zeros_like(time), clip[:1])
        s = env.state()
        prev = torch.zeros(1, env.layout.joints, device=args.device)
        hist = env.layout.obs(s, prev, env.default, False, goal=clip[:1])[:, None].repeat(1, 10, 1)
        actual = [s[0].cpu().numpy()]
        reference = [clip[0].cpu().numpy()]
        rewards = []
        failed = False
        full_length = len(clip) - 1
        limit = min(full_length, args.max_steps) if args.max_steps else full_length
        with torch.no_grad():
            for t in range(limit):
                goals = motions.get(ids, time, 11)
                a = actor(hist, goals)
                s = env.step(a)
                target = clip[t + 1 : t + 2]
                reward = env.layout.reward(s, target, a, prev, env.limits)
                failed = bool(env.layout.done(s, target).item())
                actual.append(s[0].cpu().numpy())
                reference.append(target[0].cpu().numpy())
                rewards.append(float(reward))
                hist = torch.cat(
                    (hist[:, 1:], env.layout.obs(s, a, env.default, False, goal=target)[:, None]), 1
                )
                prev = a
                time += 1
                if failed:
                    break
        out = args.output / f"clip_{ci:04d}"
        out.mkdir(exist_ok=True)
        np.savez_compressed(
            out / "rollout.npz",
            actual=np.array(actual),
            reference=np.array(reference),
            rewards=np.array(rewards),
            fps=50,
            joint_names=np.array(motions.names[0]),
            body_names=np.array(motions.names[1]),
        )
        result = {
            "clip": ci,
            "steps": len(rewards),
            "full_steps": full_length,
            "failed": failed,
            "truncated": len(rewards) < full_length and not failed,
            "normalized_return": sum(rewards) / (0.1 * full_length),
            "max_reward_per_step": 0.1,
        }
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
