"""Read-only CPU FK check against exported motions and compiled MuJoCo."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from supermimic.kinematics import Kinematics
from supermimic.core_v2 import Layout, WorldModel
from supermimic.mjwarp import build_model
from supermimic.data import select_motion_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--motions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    paths = select_motion_paths(sorted(a.motions.rglob("motion.npz")), "train40")
    states = []
    for path in paths:
        with np.load(path) as d:
            names = (d["joint_names"].tolist(), d["body_names"].tolist())
            data = d["state"]
            states.append(data[np.linspace(0, len(data) - 1, 4).astype(int)])
    s = torch.tensor(np.concatenate(states))
    cfg = json.loads(a.config.read_text())
    m, _ = build_model(cfg, names[0])
    fk = Kinematics(m, *names)
    l = Layout(*[len(n) for n in names])
    root, q, v, b = l.parts(s)
    pred = fk(root, q, v)
    target = b.clone()
    target[..., 3:7] *= torch.where(
        (target[..., 3:7] * pred[..., 3:7]).sum(-1, keepdim=True) < 0, -1, 1
    )
    err = (target - pred).abs().amax((0, 1))
    torch.testing.assert_close(pred, target, atol=5e-4, rtol=5e-4)
    q = q.clone().requires_grad_()
    f = lambda x: fk(root[:1], x, v[:1])[0, -1, :3].square().sum()
    f(q[:1]).backward()
    assert torch.isfinite(q.grad).all()
    w = WorldModel(l, len(names[0]) + 4, fk, width=16)
    ctx = torch.zeros(len(s), len(names[0]) + 4)
    ctx[:, 0] = 1
    with torch.no_grad():
        wp = w(s, torch.zeros_like(v), ctx)
    torch.testing.assert_close(
        wp[:, : 13 + 2 * l.joints], s[:, : 13 + 2 * l.joints], atol=1e-5, rtol=1e-5
    )
    result = {
        "robot": cfg["robot"],
        "sampled_frames": len(s),
        "train_clips": 40,
        "test_clips": len(
            select_motion_paths(sorted(a.motions.rglob("motion.npz")), "test37")
        ),
        "max_fk_position_error": float(err[:3].max()),
        "max_fk_quaternion_error": float(err[3:7].max()),
        "max_fk_linear_velocity_error": float(err[7:10].max()),
        "max_fk_angular_velocity_error": float(err[10:13].max()),
        "gradient_finite": True,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
