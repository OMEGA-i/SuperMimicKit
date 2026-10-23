"""Compare imported PhysX link poses against named motion FK, before stepping."""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check-inertia", action="store_true")
    p.add_argument("--body-alias", action="append", default=[], metavar="PHYSX=MJCF")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if not args.motion.is_file() or not args.xml.is_file():
        p.error("Both --motion and --xml must be existing files")
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = AppLauncher(args)
    try:
        import numpy as np
        import torch

        from supermimic.data import Motions
        from supermimic.physx import PhysX, tensor

        motions = Motions([args.motion], args.device)
        # Diagnostic only: gains are irrelevant to the pre-step FK measurement.
        cfg = {
            "asset_env": "SUPERMIMIC_PARITY_XML",
            "asset_format": "mjcf",
            "default_pose": {},
            "actuators": [
                {"pattern": ".*", "armature": 0.01, "effort": 1.0, "velocity": 100.0}
            ],
        }
        os.environ[cfg["asset_env"]] = str(args.xml.resolve())
        env = PhysX(cfg, motions.names, 1, args.device)
        rows = []
        for index in np.linspace(0, len(motions.data) - 1, 5, dtype=int):
            target = motions.data[index : index + 1]
            env.reset(torch.zeros(1, device=args.device, dtype=torch.long), target)
            actual = env.state()
            _, q, _, bodies = env.layout.parts(actual)
            _, tq, _, tb = env.layout.parts(target)
            pos = (bodies[..., :3] - tb[..., :3]).norm(dim=-1)
            dot = (bodies[..., 3:7] * tb[..., 3:7]).sum(-1).abs().clamp(0, 1)
            angle = 2 * torch.acos(dot)
            rows.append(
                {
                    "frame": int(index),
                    "max_joint_error_rad": float((q - tq).abs().max()),
                    "max_body_position_error_m": float(pos.max()),
                    "max_body_orientation_error_rad": float(angle.max()),
                    "body_position_errors_m": pos[0].tolist(),
                }
            )
        result = {
            "xml": str(args.xml),
            "motion": str(args.motion),
            "joint_names": motions.names[0],
            "body_names": motions.names[1],
            "samples": rows,
            "dynamic_tracking": "not_tested",
            "passed": all(
                r["max_joint_error_rad"] < 1e-6
                and r["max_body_position_error_m"] < 0.001
                and r["max_body_orientation_error_rad"] < 0.005
                for r in rows
            ),
        }
        if args.check_inertia:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(args.xml))
            mass = tensor(env.robot.data.body_mass)[0].cpu().numpy()
            inertia = (
                tensor(env.robot.data.body_inertia)[0].cpu().numpy().reshape(-1, 3, 3)
            )
            com = tensor(env.robot.data.body_com_pos_b)[0].cpu().numpy()
            aliases = dict(value.split("=", 1) for value in args.body_alias)
            unknown = set(aliases) - set(env.robot.body_names)
            if unknown:
                raise ValueError(f"Unknown PhysX body aliases: {unknown}")
            mapped = [aliases.get(n, n) for n in env.robot.body_names]
            if len(set(mapped)) != len(mapped):
                raise ValueError("Body aliases must be one-to-one")
            result["body_aliases"] = aliases
            physics = []
            for index, name in enumerate(env.robot.body_names):
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, aliases.get(name, name)
                )
                if bid < 0:
                    raise ValueError(f"PhysX body missing from MJCF: {name}")
                eigenvalues = np.linalg.eigvalsh(inertia[index])
                expected = np.sort(model.body_inertia[bid])
                physics.append(
                    {
                        "body": name,
                        "mass_kg": float(mass[index]),
                        "reference_mass_kg": float(model.body_mass[bid]),
                        "mass_error_kg": float(abs(mass[index] - model.body_mass[bid])),
                        "com_error_m": float(
                            np.linalg.norm(com[index] - model.body_ipos[bid])
                        ),
                        "principal_inertia": eigenvalues.tolist(),
                        "reference_principal_inertia": expected.tolist(),
                        "passed": bool(
                            np.isclose(
                                mass[index], model.body_mass[bid], rtol=1e-4, atol=1e-6
                            )
                            and np.allclose(eigenvalues, expected, rtol=1e-3, atol=1e-7)
                            and np.linalg.norm(com[index] - model.body_ipos[bid]) < 1e-5
                        ),
                    }
                )
            result["mass_com_principal_inertia"] = physics
            result["inertia_orientation"] = "not_tested"
            result["collision_parity"] = "not_tested"
            result["passed"] = result["passed"] and all(x["passed"] for x in physics)
        (args.output / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            raise RuntimeError("PhysX FK parity failed; inspect parity.json")
    except BaseException:
        import traceback

        (args.output / "FAILED.txt").write_text(traceback.format_exc())
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
