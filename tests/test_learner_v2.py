import numpy as np
import pytest
import torch
import mujoco as mj
from supermimic.kinematics import Kinematics, qrot, qconj, qmul, qexp
from supermimic.core_v2 import Layout, WorldModel, Learner

XML = """<mujoco><worldbody><body name="root"><freejoint/><geom size=".1" mass="2"/>
<body name="fixed" pos=".2 .1 .1" quat=".9238795 0 0 .3826834"><geom size=".1"/>
<body name="link" pos=".1 .2 -.3" quat=".9659258 .258819 0 0"><joint name="j" axis="0 1 0" pos=".03 -.02 .04" ref="12"/><geom size=".1"/>
<body name="tip" pos=".2 0 .1"><geom size=".1"/></body></body></body></body></worldbody></mujoco>"""


def fixture():
    model = mj.MjModel.from_xml_string(XML)
    fk = Kinematics(model, ["j"], ["root", "link", "tip"])
    l = Layout(1, 3)
    s = torch.zeros(3, l.dim)
    s[:, 2] = 1
    s[:, 3:7] = qexp(
        torch.tensor([[0.2, 0.1, 0.5], [0.1, -0.2, -0.3], [0.0, 0.0, 0.0]])
    )
    s[:, 7:13] = torch.arange(18).reshape(3, 6) * 0.01
    s[:, 13] = torch.tensor([0.1, 0.2, -0.2])
    s[:, 14] = 0.3
    s[:, 15:] = fk(s[:, :13], s[:, 13:14], s[:, 14:15]).flatten(1)
    return model, fk, l, s


def test_fk_fixed_links_rotations_anchor_reference_and_velocities():
    model, fk, l, s = fixture()
    data = mj.MjData(model)
    for row in s:
        data.qpos[:7] = row[:7].numpy()
        data.qpos[7] = row[13]
        data.qvel[:3] = row[7:10]
        data.qvel[3:6] = qrot(qconj(row[3:7]), row[10:13])
        data.qvel[6] = row[14]
        mj.mj_forward(model, data)
        bodies = l.parts(row)[-1]
        for i, name in enumerate(["root", "link", "tip"]):
            bid = model.body(name).id
            jp = np.zeros((3, model.nv))
            jr = jp.copy()
            mj.mj_jacBody(model, data, jp, jr, bid)
            target = torch.tensor(
                np.r_[data.xpos[bid], data.xquat[bid], jp @ data.qvel, jr @ data.qvel],
                dtype=torch.float32,
            )
            if (target[3:7] * bodies[i, 3:7]).sum() < 0:
                target[3:7] *= -1
            torch.testing.assert_close(bodies[i], target, atol=2e-6, rtol=2e-6)


def test_fk_gradient_matches_finite_difference():
    _, fk, _, s = fixture()
    q = s[:1, 13:14].clone().requires_grad_()
    f = lambda x: fk(s[:1, :13], x, s[:1, 14:15])[0, -1, 0]
    y = f(q)
    y.backward()
    eps = 1e-3
    fd = (f(q.detach() + eps) - f(q.detach() - eps)) / (2 * eps)
    torch.testing.assert_close(q.grad.squeeze(), fd, atol=2e-5, rtol=2e-3)


def test_aligned_body_reward_and_soft_limits():
    _, _, l, s = fixture()
    s = s[:1]
    a = torch.zeros(1, 1)
    bounds = torch.tensor([[-1.0, 1.0]])
    moved = s.clone()
    moved[:, 0] += 1
    moved[:, 15:].reshape(1, 3, 13)[:, :, 0] += 1
    r = l.reward(moved, s, a, a, bounds)
    baseline = l.reward(s, s, a, a, bounds)
    expected = baseline - 0.01 * (1 - np.exp(-1 / 0.3**2))
    torch.testing.assert_close(r, expected)
    near = s.clone()
    near[:, 13] = 0.95
    torch.testing.assert_close(
        l.reward(near, near, a, a, bounds),
        l.reward(s, s, a, a, bounds) - 0.002,
        atol=1e-6,
        rtol=1e-6,
    )


def test_world_always_has_consistent_pelvis_and_finite_gradients():
    _, fk, l, s = fixture()
    w = WorldModel(l, 5, fk, width=16)
    a = torch.randn(3, 1, requires_grad=True)
    c = torch.zeros(3, 5)
    c[:, 0] = 1
    torch.nn.init.normal_(w.dynamics[-1].weight, std=0.01)
    for _ in range(4):
        s = w(s, a, c)
    torch.testing.assert_close(s[:, :13], s[:, 15:28], atol=0, rtol=0)
    s[:, 15:].sum().backward()
    assert torch.isfinite(a.grad).all()
    assert a.grad.abs().sum() > 0


def test_goal_translation_invariance_and_relative_orientation():
    _, _, l, s = fixture()
    seq = s[:, None].repeat(1, 11, 1)
    other = seq.clone()
    other[..., :2] += 10
    other[..., 15:].reshape(3, 11, 3, 13)[..., :2] += 10
    torch.testing.assert_close(l.goals(seq), l.goals(other))
    changed = s.clone()
    changed[:, 3:7] = qmul(
        qexp(torch.tensor([[0.0, 0.0, 0.7]]).expand(3, -1)), s[:, 3:7]
    )
    a = torch.zeros(3, 1)
    d = torch.zeros(1)
    x = l.obs(s, a, d, noise=False, goal=s)
    y = l.obs(changed, a, d, noise=False, goal=s)
    assert (x[:, -6:] - y[:, -6:]).abs().max() > 0.1


def test_v2_training_backward_and_checkpoint_roundtrip(tmp_path):
    _, fk, l, s = fixture()
    a = Learner(l, torch.zeros(1), torch.tensor([[-1.0, 1.0]]), 5, fk, small=True)
    c = torch.zeros(3, 5)
    c[:, 0] = 1
    prev = torch.zeros(3, 1)
    goals = s[:, None].repeat(1, 15, 1)
    hist = l.obs(s, prev, a.default, goal=s)[:, None].repeat(1, 10, 1)
    a.observe(s, prev, c, s, hist, goals[:, :11])
    batch = {
        "states": s[None].repeat(9, 1, 1),
        "actions": prev[None].repeat(8, 1, 1),
        "context": c,
        "boundary": torch.zeros(8, 3, dtype=torch.bool),
    }
    assert np.isfinite(a.world_update(batch))
    before = {k: v.clone() for k, v in a.world.state_dict().items()}
    metrics = a.policy_update(s, hist, prev, c, goals, 4)
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.equal(v, a.world.state_dict()[k]) for k, v in before.items())
    torch.save(a.actor.state_dict(), tmp_path / "actor.pt")
    a.actor.load_state_dict(torch.load(tmp_path / "actor.pt", weights_only=True))


def test_split_exact_disjoint_and_complete(tmp_path):
    from pathlib import Path
    import json
    from supermimic.data import select_motion_paths

    manifest = json.loads(Path("configs/lafan40.json").read_text())
    paths = [Path(n) / "motion.npz" for n in manifest["motions"] + manifest["test37"]]
    fixture_path = tmp_path / "manifest.json"
    fixture_path.write_text(
        json.dumps(
            {
                "splits": {
                    "status": "assigned",
                    "train": manifest["motions"],
                    "test": manifest["test37"],
                }
            }
        )
    )
    train = select_motion_paths(paths, "train40", fixture_path)
    test = select_motion_paths(paths, "test37", fixture_path)
    assert len(train) == 40 and len(test) == 37 and not set(train) & set(test)
    assert set(train + test) == set(paths)
    with pytest.raises(ValueError):
        select_motion_paths(paths[1:], "all77", fixture_path)


def test_batched_execution_preserves_losses_and_parameter_updates():
    import copy

    _, fk, l, s = fixture()
    fast = Learner(l, torch.zeros(1), torch.tensor([[-1.0, 1.0]]), 5, fk, small=True)
    slow = copy.deepcopy(fast)
    slow.optimized = False
    # Nonzero output weights exercise dynamics and FK gradient propagation.
    torch.nn.init.normal_(fast.world.dynamics[-1].weight, std=0.001)
    slow.world.load_state_dict(fast.world.state_dict())
    c = torch.zeros(3, 5)
    c[:, 0] = 1
    prev = torch.zeros(3, 1)
    goals = s[:, None].repeat(1, 15, 1)
    hist = l.obs(s, prev, fast.default, noise=False, goal=s)[:, None].repeat(1, 10, 1)
    batch = {
        "states": s[None].repeat(9, 1, 1),
        "actions": prev[None].repeat(8, 1, 1),
        "context": c,
        "boundary": torch.zeros(8, 3, dtype=torch.bool),
    }
    batch["boundary"][2, 1] = True
    a, b = fast.world_update(batch), slow.world_update(batch)
    assert a == pytest.approx(b, rel=2e-5, abs=1e-7)
    for p, q in zip(fast.world.parameters(), slow.world.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
    # Start from identical post-world-update parameters; compare policy gradients.
    slow.world.load_state_dict(fast.world.state_dict())
    torch.manual_seed(123)
    a = fast.policy_update(s, hist, prev, c, goals, 4)
    torch.manual_seed(123)
    b = slow.policy_update(s, hist, prev, c, goals, 4)
    for key in a:
        assert a[key] == pytest.approx(b[key], rel=2e-4, abs=2e-6), key
    for p, q in zip(fast.actor.parameters(), slow.actor.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-4)


def test_compiler_preflight_fails_with_actionable_message(monkeypatch):
    from supermimic.compiler import check_compiler

    monkeypatch.setenv("CC", "/missing/compiler")
    monkeypatch.setenv("CXX", "/missing/compiler++")
    with pytest.raises(RuntimeError, match="build-essential"):
        check_compiler()


def test_precision_is_opt_in():
    _, fk, l, _ = fixture()
    learner = Learner(l, torch.zeros(1), torch.tensor([[-1.0, 1.0]]), 5, fk, small=True)
    assert learner.world.amp == learner.actor.amp == learner.critic.amp == "none"


def test_cross_components_match_values_and_gradients():
    from supermimic.kinematics import cross3

    a = torch.randn(8, 1, 3, requires_grad=True)
    b = torch.randn(8, 4, 3, requires_grad=True)
    weight = torch.randn(8, 4, 3)
    x, y = cross3(a, b), torch.cross(a, b, dim=-1)
    torch.testing.assert_close(x, y)
    gx = torch.autograd.grad((x * weight).sum(), (a, b))
    gy = torch.autograd.grad((y * weight).sum(), (a, b))
    for u, v in zip(gx, gy):
        torch.testing.assert_close(u, v)


def test_running_norm_freezes_without_host_scalar_branch():
    from supermimic.core_v2 import RunningNorm

    n = RunningNorm(3)
    n.update(torch.randn(8, 3))
    n.count.fill_(2e8)
    mean, var = n.mean.clone(), n.var.clone()
    n.update(torch.randn(8, 3) + 10)
    torch.testing.assert_close(n.mean, mean)
    torch.testing.assert_close(n.var, var)
