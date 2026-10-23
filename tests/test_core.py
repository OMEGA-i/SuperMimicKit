import numpy as np
import torch

from supermimic.core import Layout, Learner, lambda_returns
from supermimic.data import Replay
from supermimic.retarget import resample


def states(layout, n):
    x = torch.randn(n, layout.dim) * 0.01
    x[:, 3] = 1
    x[:, 13 + 2 * layout.joints :].view(n, layout.bodies, 13)[:, :, 3] = 1
    return layout.project(x)


def test_bootstrap_and_terminal():
    r = torch.tensor([[1.0], [2.0]])
    v = torch.tensor([[8.0], [9.0], [10.0]])
    assert torch.allclose(
        lambda_returns(r, torch.ones_like(r), v, 1, 1), torch.tensor([[13.0], [12.0]])
    )
    assert torch.allclose(lambda_returns(r, torch.zeros_like(r), v, 1, 1), r)


def test_policy_gradient_through_frozen_world():
    torch.manual_seed(10)
    l = Layout(2, 1)
    agent = Learner(l, torch.zeros(2), torch.tensor([[-1.0, 1.0]] * 2), 1, small=True)
    torch.nn.init.normal_(agent.world.dynamics[-1].weight, std=0.02)
    s = states(l, 4)
    prev = torch.zeros(4, 2)
    hist = l.obs(s, prev, torch.zeros(2), False)[:, None].repeat(1, 10, 1)
    goals = s[:, None].repeat(1, 14, 1).clone()
    goals[:, :, 0] += 0.1
    before = {k: v.clone() for k, v in agent.world.state_dict().items()}
    actor_before = [p.clone() for p in agent.actor.parameters()]
    result = agent.policy_update(s, hist, prev, torch.ones(4, 1), goals, horizon=4)
    assert all(np.isfinite(v) for v in result.values())
    assert any(
        not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())
    )
    assert all(torch.equal(v, agent.world.state_dict()[k]) for k, v in before.items())


def test_replay_ring_chronology(tmp_path):
    replay = Replay(4)
    for t in range(7):
        s = torch.tensor([t])
        replay.add(
            state=s[:, None],
            next_state=(s + 1)[:, None],
            action=s[:, None],
            goal=s[:, None],
            next_goal=(s + 1)[:, None],
            terminal=s.bool(),
            boundary=s.bool(),
            context=s[:, None],
            history=s[:, None],
            previous=s[:, None],
            clip=s,
            time=s,
        )
    torch.save(replay.state_dict(), tmp_path / "replay.pt")
    restored = Replay(4)
    restored.load_state_dict(torch.load(tmp_path / "replay.pt", weights_only=True))
    rng = torch.get_rng_state()
    b = replay.sample(100, 3)
    torch.set_rng_state(rng)
    resumed = restored.sample(100, 3)
    assert all(torch.equal(b[k], resumed[k]) for k in b)

    assert (b["states"][1:] - b["states"][:-1] == 1).all()
    assert b["states"].min() >= 3 and b["states"].max() <= 7


def test_terminal_transition_included_reset_excluded():
    torch.manual_seed(3)
    l = Layout(1, 1)
    a = Learner(l, torch.zeros(1), torch.tensor([[-1.0, 1.0]]), 1, small=True)
    b = {
        "states": states(l, 6).reshape(3, 2, -1),
        "actions": torch.zeros(2, 2, 1),
        "context": torch.ones(2, 1),
        "goals": states(l, 6).reshape(3, 2, -1),
        "terminal": torch.ones(2, 2, dtype=torch.bool),
        "boundary": torch.ones(2, 2, dtype=torch.bool),
    }
    c = {k: v.clone() for k, v in b.items()}
    c["states"][2] += 100
    clone = Learner(l, torch.zeros(1), torch.tensor([[-1.0, 1.0]]), 1, small=True)
    clone.world.load_state_dict(a.world.state_dict())
    assert abs(a.world_update(b) - clone.world_update(c)) < 1e-6


def test_resample_quaternion_shortest_arc():
    q = np.zeros((31, 8))
    q[:, 3] = 1
    q[15:, 3] = -1
    q[:, 0] = np.linspace(0, 1, 31)
    result = resample(q, 30)
    assert len(result) == 51
    assert np.allclose(np.linalg.norm(result[:, 3:7], axis=-1), 1)
    assert np.isclose(result[-1, 0], 1)


def test_physx_quaternion_boundary():
    from supermimic.physx import from_physx

    x = torch.zeros(2, 13)
    x[:, 6] = 1
    y = from_physx(x)
    assert torch.equal(y[:, 3:7], torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1))
    assert torch.equal(x[:, 6], torch.ones(2))


def test_world_input_gradient_with_frozen_weights():
    from supermimic.core import frozen

    l = Layout(2, 1)
    agent = Learner(l, torch.zeros(2), torch.tensor([[-1.0, 1.0]] * 2), 1, small=True)
    torch.nn.init.normal_(agent.world.dynamics[-1].weight, std=0.05)
    action = torch.zeros(3, 2, requires_grad=True)
    with frozen(agent.world):
        predicted = agent.world(states(l, 3), action, torch.ones(3, 1))
        predicted[:, :3].sum().backward()
    assert action.grad.abs().sum() > 0
    assert all(p.grad is None for p in agent.world.parameters())


def test_state_loss_quaternion_double_cover():
    l = Layout(2, 1)
    s = states(l, 4)
    target = s.clone()
    target[:, 3:7] *= -1
    target[:, 13 + 4 :].view(4, 1, 13)[:, :, 3:7] *= -1
    assert torch.equal(l.state_loss(s, target), torch.zeros(4))


def test_calibration_is_consistent_in_observation_and_action_coordinates():
    layout = Layout(2, 1)
    state = states(layout, 3)
    default = torch.tensor([0.2, -0.1])
    previous = torch.zeros(3, 2)
    context = torch.zeros(3, 6)
    context[:, 0] = 0.7
    context[:, 1:3] = torch.tensor([[0.01, -0.01], [-0.005, 0.004], [0.002, 0.003]])
    nominal = layout.obs(state, previous, default, noise=False)
    calibrated = layout.obs(state, previous, default, noise=False, context=context)
    torch.testing.assert_close(calibrated[:, :2], nominal[:, :2] - context[:, 1:3])
    torch.testing.assert_close(calibrated[:, 2:], nominal[:, 2:])
    # A physical position equal to the calibrated action origin reads zero.
    state[:, 13:15] = default + context[:, 1:3]
    observed = layout.obs(state.requires_grad_(), previous, default, noise=False, context=context)
    torch.testing.assert_close(observed[:, :2], torch.zeros(3, 2))
    observed[:, :2].sum().backward()
    torch.testing.assert_close(state.grad[:, 13:15], torch.ones(3, 2))
