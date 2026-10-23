"""Single GPU collect / world fit / differentiable imagination entrypoint."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def learning_rate(iteration, base=2e-4, decay_iterations=250_000, floor=0.03):
    """Paper Table A.5: cosine to 3% at 250k updates, then constant."""
    if iteration < 0 or decay_iterations < 1 or not 0 < floor <= 1:
        raise ValueError("Invalid learning-rate schedule")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("Learning rate must be finite and positive")
    progress = min(iteration / decay_iterations, 1.0)
    return base * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--motions", type=Path, required=True)
    p.add_argument("--motion-split", choices=("train40",), default="train40")
    p.add_argument("--split-manifest", type=Path,
                   default=Path("datasets/lafan/manifest.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=320_000)
    p.add_argument("--replay-steps", type=int, default=1000)
    p.add_argument(
        "--batch-size",
        "--world-batch-size",
        dest="batch_size",
        type=int,
        default=2048,
        help="World-model windows per optimizer step",
    )
    p.add_argument(
        "--policy-batch-size",
        type=int,
        default=4096,
        help="Imagined rollouts per policy update",
    )
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--world-batches", type=int, default=10)
    p.add_argument("--small", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=64)
    p.add_argument("--world-only-iterations", type=int, default=500)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cuda-graphs", action="store_true", help="Use reduce-overhead compiled network CUDA graphs")
    p.add_argument("--no-runtime-tuning", action="store_true", help="Ablate fused Adam and compiled collection/critic")
    p.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="fp32",
                   help="Network compute mode; FP32 storage and geometry in every mode")
    p.add_argument("--legacy-execution", action="store_true", help="Unbatched execution for throughput comparison")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lr-decay-iterations", type=int, default=250_000)
    p.add_argument("--lr-floor", type=float, default=0.03)
    p.add_argument("--profile-iteration", type=int, default=-1,
                   help="Capture one iteration to profile.json; exclude it from throughput averages")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-asset-sha256", help="Explicitly migrate from this old asset hash; discard old replay/runtime")
    p.add_argument("--resume-equivalent-asset", action="store_true", help="Verify path-only asset relocation and retain full runtime/replay")
    p.add_argument("--resume", type=Path, help="Trusted local full training checkpoint")
    p.add_argument("--checkpoint-every", type=int, default=1000)
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
    if (
        min(
            args.num_envs,
            args.iterations,
            args.batch_size,
            args.policy_batch_size,
            args.world_batches,
            args.checkpoint_every,
        )
        < 1
    ):
        p.error("Counts must be positive")
    if args.replay_steps < 8 or not 1 <= args.horizon <= 32:
        p.error("Replay must hold 8 steps; horizon must be in [1,32]")
    learning_rate(0, args.learning_rate, args.lr_decay_iterations, args.lr_floor)
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    launcher = AppLauncher(args) if args.backend == "physx" else None
    try:
        run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        (args.output / "FAILED.txt").write_text(traceback.format_exc())
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


def run(args):
    import torch

    from .core_v2 import Learner, Layout, VERSION
    from .kinematics import Kinematics
    from .data import Motions, Replay, ResetSampler, select_motion_paths
    if args.backend == "physx":
        from .physx import PhysX as Simulator
    else:
        from .mjwarp import MJWarp as Simulator

    if args.compile:
        from .compiler import check_compiler
        check_compiler()
    torch.manual_seed(args.seed)
    from .precision import configure_precision
    precision_settings = configure_precision(args.precision)
    cfg = json.loads(args.config.read_text())
    paths = select_motion_paths(sorted(args.motions.rglob("motion.npz")), args.motion_split, args.split_manifest)
    motions = Motions(paths, args.device)
    env = Simulator(cfg, motions.names, args.num_envs, args.device, randomize=True)
    env.layout = Layout(env.layout.joints, env.layout.bodies)
    if args.backend == "mujoco-warp":
        fk_model = env.cpu_model
    else:
        import mujoco
        fk_model = mujoco.MjModel.from_xml_path(os.environ[cfg["asset_env"]])
    root_body = int(fk_model.jnt_bodyid[0])
    if (fk_model.body_mass[root_body] == 0.001
            and (abs(fk_model.body_inertia[root_body] - 1e-6) < 1e-12).all()):
        raise ValueError("Retarget placeholder root inertia is invalid for training; use the corrected physical MJCF")
    kinematics = Kinematics(fk_model, *motions.names).to(args.device)
    learner = Learner(
        env.layout,
        env.default,
        env.limits,
        env.context.shape[1],
        kinematics,
        args.device,
        args.small,
        lr=args.learning_rate,
        compile=args.compile,
        runtime_tuning=not args.no_runtime_tuning,
        cuda_graphs=args.cuda_graphs,
        optimized=not args.legacy_execution,
        amp="bf16" if args.precision == "bf16" else "none",
    )
    replay = Replay(args.replay_steps)
    sampler = ResetSampler(motions.lengths, tail=args.horizon + 11)
    contract = {
        "backend": args.backend,
        "physics_contract_version": 5,
        "learner_version": VERSION,
        "motion_split": args.motion_split,
        "precision": args.precision,
        "domain_randomization": "A3-default-offset-torso-com-v1",
        "reset_sampler": "D1-1s-50uniform-40current-10historical-EMA0.1-v1",
        "config": cfg,
        "asset_sha256": hashlib.sha256(
            Path(os.environ[cfg["asset_env"]]).read_bytes()
        ).hexdigest(),
        "motions": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        "arguments": {
            k: getattr(args, k)
            for k in (
                "num_envs",
                "replay_steps",
                "batch_size",
                "policy_batch_size",
                "horizon",
                "world_batches",
                "small",
                "warmup_steps",
                "world_only_iterations",
                "learning_rate",
                "lr_decay_iterations",
                "lr_floor",
            )
        },
    }
    checkpoint = None
    asset_migration = False
    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu" if args.resume_asset_sha256 else args.device,
            mmap=bool(args.resume_asset_sha256), weights_only=False
        )
        saved_contract = checkpoint.get("training_contract")
        if saved_contract is not None:
            # Old "none" meant FP32 storage with high/TF32 matmul enabled.
            if saved_contract.get("precision", "none") == "none":
                saved_contract["precision"] = "tf32"
        if args.resume_equivalent_asset:
            if args.resume_asset_sha256 or not saved_contract:
                raise ValueError("Equivalent relocation requires a contract and excludes physics migration")
            from .asset_identity import verify_asset
            verify_asset(os.environ[cfg["asset_env"]], saved_contract["asset_sha256"])
            saved_contract = dict(saved_contract, asset_sha256=contract["asset_sha256"])
        if args.resume_asset_sha256:
            if not saved_contract or saved_contract.get("asset_sha256") != args.resume_asset_sha256:
                raise ValueError("Migration source asset hash mismatch")
            expected = dict(saved_contract, asset_sha256=contract["asset_sha256"])
            if expected != contract:
                raise ValueError("Asset migration permits only the asset hash to change")
            saved_contract = expected
            asset_migration = True
        if (
            saved_contract != contract
            or "runtime" not in checkpoint
        ):
            raise ValueError(
                "Resume requires a full checkpoint with matching assets, motions and training settings"
            )
        if args.iterations <= checkpoint["iteration"] + 1:
            raise ValueError(
                "--iterations must exceed the checkpoint's completed iterations"
            )
        for name in ("actor", "world", "critic"):
            getattr(learner, name).load_state_dict(checkpoint[name])
        for optimizer, saved in zip(
            learner.optimizers, checkpoint["optimizers"], strict=True
        ):
            optimizer.load_state_dict(saved)
        if not asset_migration:
            replay.load_state_dict(checkpoint["replay"])

    ids, times = sampler.sample(args.num_envs)
    start_bins = sampler.bin_ids(ids, times)
    episode_return = torch.zeros(args.num_envs, device=args.device)
    episode_discount = torch.ones_like(episode_return)
    env.reset(
        torch.arange(args.num_envs, device=args.device), motions.get(ids, times)[:, 0]
    )
    state = env.state()
    previous = torch.zeros(args.num_envs, env.layout.joints, device=args.device)
    history = env.layout.obs(state, previous, env.default, context=env.context, goal=motions.get(ids, times)[:, 0])[:, None].repeat(1, 10, 1)
    start_iteration, steps = 0, 0
    if checkpoint is not None and not asset_migration:
        runtime = checkpoint["runtime"]
        ids, times = runtime["ids"], runtime["times"]
        sampler.load_state_dict(checkpoint["reset_sampler"])
        start_bins = runtime["start_bins"]
        episode_return = runtime["episode_return"]
        episode_discount = runtime["episode_discount"]
        previous, history = runtime["previous"], runtime["history"]
        env.restore_randomization(runtime["randomization"])
        env.reset(torch.arange(args.num_envs, device=args.device), runtime["state"], resample_context=False)
        state = env.state()
        start_iteration = checkpoint["iteration"] + 1
        steps = runtime["steps"]
        torch.set_rng_state(runtime["cpu_rng"].cpu())
        if runtime["cuda_rng"] is not None:
            torch.cuda.set_rng_state(runtime["cuda_rng"].cpu(), args.device)
    if checkpoint is not None and asset_migration:
        start_iteration = checkpoint["iteration"] + 1
        steps = checkpoint["runtime"]["steps"]
        sampler.load_state_dict({k: v.to(args.device) if torch.is_tensor(v) else v
                                 for k, v in checkpoint["reset_sampler"].items()})
        del checkpoint
        checkpoint = None
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.jsonl").exists() and not args.resume:
        raise FileExistsError("Use a new output directory")
    import subprocess
    from importlib import metadata

    manifest = {
        **vars(args),
        "config_data": cfg,
        "motion_paths": [str(p) for p in paths],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {
            p: metadata.version(p) for p in (("torch", "isaacsim", "isaaclab", "numpy") if args.backend == "physx" else ("torch", "mujoco", "mujoco-warp", "warp-lang", "numpy"))
        },
        "backend": args.backend,
        "precision_settings": precision_settings,
        "asset_path": str(Path(os.environ[cfg["asset_env"]]).resolve()),
        "context": "friction, joint default-position offsets, torso COM offsets; per episode",
        "randomization_readback": env.verify_context(),
        "status": "experimental bring-up; not paper-matched",
    }
    elapsed_offset = 0.0
    if args.resume and (args.output / "metrics.jsonl").exists():
        with (args.output / "metrics.jsonl").open() as prior:
            for line in prior:
                try: elapsed_offset = max(elapsed_offset, json.loads(line).get("elapsed_s", 0.0))
                except json.JSONDecodeError: pass
        manifest["resume_iteration"] = start_iteration
        manifest["asset_migration"] = asset_migration
        manifest["previous_asset_sha256"] = args.resume_asset_sha256
        (args.output / f"resume_{start_iteration}_{time.time_ns()}.json").write_text(json.dumps(manifest, default=str, indent=2))
    (args.output / "run.json").write_text(json.dumps(manifest, default=str, indent=2))
    start = time.monotonic() - elapsed_offset
    with (args.output / "metrics.jsonl").open("a" if args.resume else "w", buffering=1) as log:
        for iteration in range(start_iteration, args.iterations):
            lr = learning_rate(
                iteration, args.learning_rate, args.lr_decay_iterations, args.lr_floor
            )
            for optimizer in learner.optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr
            profiler = None
            if iteration == args.profile_iteration:
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.CUDA])
                profiler.start()
            torch.cuda.synchronize(args.device)
            phase_start = time.monotonic()
            collect = max(8, args.warmup_steps) if iteration == 0 or (asset_migration and iteration == start_iteration) else (3 if iteration % 2 else 2)
            for _ in range(collect):
                if args.cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    goal = motions.get(ids, times, 11)
                    action = learner.collect_actor(history, goal) + 0.1 * torch.randn_like(
                        previous
                    )
                    action = action.clamp(-20, 20)
                    ns = env.step(action)
                    learner.observe_step(state, action, env.context, ns, history, goal)
                    ng = motions.get(ids, times + 1)[:, 0]
                    terminal, reward, next_obs = learner.collect_metrics(
                        ns, ng, action, previous, env.limits, env.context
                    )
                    if args.cuda_graphs:
                        terminal, reward, next_obs = (x.clone() for x in (terminal, reward, next_obs))
                    boundary = terminal | (
                        times + 1 >= motions.lengths[ids] - args.horizon - 10
                    )
                    episode_return += episode_discount * reward
                    episode_discount *= sampler.gamma
                    replay.add(
                        state=state,
                        next_state=ns,
                        action=action,
                        goal=goal[:, 0],
                        next_goal=ng,
                        terminal=terminal,
                        boundary=boundary,
                        context=env.context,
                        history=history,
                        previous=previous,
                        clip=ids,
                        time=times,
                    )
                    previous = action
                    state = ns
                    times = times + 1
                    history = torch.cat(
                        (
                            history[:, 1:],
                            next_obs[:, None],
                        ),
                        1,
                    )
                    reset = boundary.nonzero().flatten()
                    if len(reset):
                        sampler.update(start_bins[reset], episode_return[reset])
                        ri, rt = sampler.sample(len(reset))
                        start_bins[reset] = sampler.bin_ids(ri, rt)
                        episode_return[reset] = 0
                        episode_discount[reset] = 1
                        ids[reset] = ri
                        times[reset] = rt
                        env.reset(reset, motions.get(ri, rt)[:, 0])
                        state = env.state()
                        previous[reset] = 0
                        history[reset] = env.layout.obs(
                            state[reset], previous[reset], env.default, context=env.context[reset], goal=motions.get(ri, rt)[:, 0]
                        )[:, None].repeat(1, 10, 1)
                steps += 1
            torch.cuda.synchronize(args.device)
            collect_s = time.monotonic() - phase_start
            phase_start = time.monotonic()
            for _ in range(args.world_batches):
                wm_loss = learner.world_update(replay.sample(args.batch_size))
            torch.cuda.synchronize(args.device)
            world_s = time.monotonic() - phase_start
            phase_start = time.monotonic()
            metrics = {"actor_loss": 0., "critic_loss": 0., "world_error8": learner.last_error8}
            # A changed physical asset invalidates old transitions. Refit the
            # retained world model on fresh replay before resuming actor updates.
            policy_start = max(args.world_only_iterations, start_iteration + 500 if asset_migration else 0)
            if iteration >= policy_start:
                batch = replay.sample(args.policy_batch_size)
                goals = motions.get(batch["clip"], batch["time"], args.horizon + 11)
                metrics.update(learner.policy_update(
                    batch["states"][0], batch["history"], batch["previous"],
                    batch["context"], goals, args.horizon,
                ))
            torch.cuda.synchronize(args.device)
            policy_s = time.monotonic() - phase_start
            metrics.update(
                collect_s=collect_s, world_s=world_s, policy_s=policy_s,
                timestamp=time.time(),
                iteration=iteration,
                learning_rate=lr,
                sim_steps=steps,
                elapsed_s=time.monotonic() - start,
                world_loss=wm_loss,
                sim_reward=float(reward.mean()),
                terminal_fraction=float(terminal.float().mean()),
            )
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(metrics)
            if profiler is not None:
                profiler.stop()
                profiler.export_chrome_trace(str(args.output / "profile.json"))
                (args.output / "profile.txt").write_text(profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=40))
            log.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if (iteration + 1) % 10000 == 0:
                from .checkpoints import save_evaluation_checkpoint
                save_evaluation_checkpoint({
                    "learner_version": VERSION,
                    "training_contract": contract,
                    "actor": learner.actor.state_dict(),
                    "world": learner.world.state_dict(),
                    "iteration": iteration,
                    "config": cfg,
                    "joint_names": motions.names[0],
                    "body_names": motions.names[1],
                    "args": vars(args),
                }, args.output / f"ckpt_{iteration + 1}.pt", include_world=True)
            if (
                iteration + 1
            ) % args.checkpoint_every == 0 or iteration + 1 == args.iterations:
                temp = args.output / "checkpoint.tmp"
                torch.save(
                    {
                        "learner_version": VERSION,
                        "training_contract": contract,
                        "replay": replay.state_dict(),
                        "reset_sampler": sampler.state_dict(),
                        "runtime": {
                            "randomization": env.randomization_state(),
                            "start_bins": start_bins,
                            "episode_return": episode_return,
                            "episode_discount": episode_discount,
                            "ids": ids,
                            "times": times,
                            "state": state,
                            "previous": previous,
                            "history": history,
                            "steps": steps,
                            "cpu_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state(args.device)
                            if str(args.device).startswith("cuda")
                            else None,
                        },
                        "resume_semantics": "Physical state restored; solver contact caches are rebuilt, not bitwise continuation",
                        "actor": learner.actor.state_dict(),
                        "world": learner.world.state_dict(),
                        "critic": learner.critic.state_dict(),
                        "optimizers": [o.state_dict() for o in learner.optimizers],
                        "iteration": iteration,
                        "config": cfg,
                        "joint_names": motions.names[0],
                        "body_names": motions.names[1],
                        "layout": vars(env.layout),
                        "args": vars(args),
                    },
                    temp,
                )
                temp.replace(args.output / "last.pt")
    (args.output / "COMPLETED.json").write_text(
        json.dumps(
            {
                "iterations": args.iterations,
                "sim_steps": steps,
                "convergence_validated": False,
                "randomization_readback": env.verify_context(),
            }
        )
    )


if __name__ == "__main__":
    main()
