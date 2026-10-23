"""Differentiable learner. No simulator imports; quaternions are scalar first."""

import itertools
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from .kinematics import cross3


def mlp(dims):
    layers = []
    for i, (a, b) in enumerate(itertools.pairwise(dims)):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def normalize(q):
    return F.normalize(q, dim=-1, eps=1e-8)


def rotate_inverse(q, v):
    q = normalize(q)
    xyz = -q[..., 1:]
    t = 2 * cross3(xyz, v)
    return v + q[..., :1] * t + cross3(xyz, t)


def rotation_error(q, r):
    # Squared geodesic angle; clamp keeps acos derivative finite at identity.
    dot = (normalize(q) * normalize(r)).sum(-1).abs().clamp(max=1 - 1e-7)
    return (2 * torch.acos(dot)).square()


@dataclass
class Layout:
    joints: int
    bodies: int

    @property
    def dim(self):
        return 13 + 2 * self.joints + 13 * self.bodies

    @property
    def obs_dim(self):
        return 3 * self.joints + 6

    def parts(self, s):
        d = self.joints
        return (
            s[..., :13],
            s[..., 13 : 13 + d],
            s[..., 13 + d : 13 + 2 * d],
            s[..., 13 + 2 * d :].reshape(*s.shape[:-1], self.bodies, 13),
        )

    def project(self, s):
        r, q, v, b = self.parts(s)
        r = torch.cat((r[..., :3], normalize(r[..., 3:7]), r[..., 7:]), -1)
        b = torch.cat((b[..., :3], normalize(b[..., 3:7]), b[..., 7:]), -1)
        return torch.cat((r, q, v, b.flatten(-2)), -1)

    def state_loss(self, predicted, target):
        # q and -q encode identical rotations. Align targets before Euclidean loss.
        r, _q, _v, b = self.parts(predicted)
        tr, tq, tv, tb = self.parts(target)
        rq = tr[..., 3:7] * torch.where(
            (r[..., 3:7] * tr[..., 3:7]).sum(-1, keepdim=True) < 0, -1.0, 1.0
        )
        bq = tb[..., 3:7] * torch.where(
            (b[..., 3:7] * tb[..., 3:7]).sum(-1, keepdim=True) < 0, -1.0, 1.0
        )
        aligned = torch.cat(
            (
                torch.cat((tr[..., :3], rq, tr[..., 7:]), -1),
                tq,
                tv,
                torch.cat((tb[..., :3], bq, tb[..., 7:]), -1).flatten(-2),
            ),
            -1,
        )
        return (predicted - aligned).square().mean(-1)

    def obs(self, s, previous_action, default, noise=True, context=None):
        r, q, v, _ = self.parts(s)
        gravity = torch.zeros_like(r[..., :3])
        gravity[..., 2] = -1
        angular = rotate_inverse(r[..., 3:7], r[..., 10:13])
        gravity = rotate_inverse(r[..., 3:7], gravity)
        if context is not None and context.shape[-1] > 1:
            default = default + context[..., 1 : 1 + self.joints]
        o = torch.cat((q - default, v, angular, gravity, previous_action), -1)
        if noise:
            scale = o.new_tensor(
                [0.01] * self.joints
                + [0.5] * self.joints
                + [0.2] * 3
                + [0.05] * 3
                + [0] * self.joints
            )
            o = o + (2 * torch.rand_like(o) - 1) * scale
        return o

    def reward(self, s, goal, action, previous_action, limits):
        r, q, _, b = self.parts(s)
        gr, _, _, gb = self.parts(goal)
        reward = torch.zeros_like(q[..., 0])
        for part, sigma in (
            (slice(0, 3), 0.3),
            (slice(7, 10), 1.0),
            (slice(10, 13), math.pi),
        ):
            error = (b[..., part] - gb[..., part]).square().sum(-1).mean(-1)
            reward = reward + torch.exp(-error / sigma**2)
        reward = reward + torch.exp(
            -rotation_error(b[..., 3:7], gb[..., 3:7]).mean(-1) / 0.4**2
        )
        reward = reward + 0.5 * torch.exp(
            -(r[..., :3] - gr[..., :3]).square().sum(-1) / 0.3**2
        )
        reward = reward + 0.5 * torch.exp(
            -rotation_error(r[..., 3:7], gr[..., 3:7]) / 0.4**2
        )
        reward = reward - 0.1 * (action - previous_action).square().sum(-1)
        reward = reward - 2 * (F.relu(q - limits[:, 1]) + F.relu(limits[:, 0] - q)).sum(
            -1
        )
        return 0.02 * reward

    def done(self, s, goal):
        r, _, _, _ = self.parts(s)
        gr, _, _, _ = self.parts(goal)
        gravity = torch.zeros_like(r[..., :3])
        gravity[..., 2] = -1
        tilt = (
            rotate_inverse(r[..., 3:7], gravity)[..., 2]
            - rotate_inverse(gr[..., 3:7], gravity)[..., 2]
        ).abs()
        return ((r[..., 2] - gr[..., 2]).abs() > 0.25) | (tilt > 0.8)


class WorldModel(nn.Module):
    def __init__(self, layout, context_dim, width=2048):
        super().__init__()
        self.layout = layout
        self.dynamics = mlp(
            [layout.dim + layout.joints + context_dim, width, width, width, layout.dim]
        )
        self.termination = mlp([2 * layout.dim, 256, 128, 1])
        nn.init.zeros_(self.dynamics[-1].weight)
        nn.init.zeros_(self.dynamics[-1].bias)

    def forward(self, state, action, context):
        return self.layout.project(
            state + self.dynamics(torch.cat((state, action, context), -1))
        )

    def logits(self, state, goal):
        return self.termination(torch.cat((state, goal), -1)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, layout, small=False):
        super().__init__()
        widths = [64, 64] if small else [2048, 1024, 512, 256]
        self.encoder = mlp([10 * layout.dim, *widths, 32])
        self.decoder = mlp([10 * layout.obs_dim + 32, *widths, layout.joints])

    def forward(self, history, goals):
        z = torch.tanh(self.encoder(goals.flatten(-2)))
        return self.decoder(torch.cat((history.flatten(-2), z), -1))


@contextmanager
def frozen(*modules):
    parameters = [p for m in modules for p in m.parameters()]
    old = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(parameters, old):
            p.requires_grad_(flag)


def lambda_returns(rewards, continuation, values, gamma=0.99, lam=0.95):
    result = []
    target = values[-1]
    for t in reversed(range(len(rewards))):
        target = rewards[t] + gamma * continuation[t] * (
            (1 - lam) * values[t + 1] + lam * target
        )
        result.append(target)
    return torch.stack(result[::-1]).detach()


class Learner:
    def __init__(
        self, layout, default, limits, context_dim, device="cpu", small=False, lr=2e-4
    ):
        self.layout, self.default, self.limits = layout, default, limits
        self.world = WorldModel(layout, context_dim, 64 if small else 2048).to(device)
        self.actor = Actor(layout, small).to(device)
        self.critic = mlp([layout.dim * 11 + context_dim, 256, 128, 1]).to(device)
        self.optimizers = [
            torch.optim.Adam(self.world.parameters(), lr=lr),
            torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.95)),
            torch.optim.Adam(self.critic.parameters(), lr=lr),
        ]

    def value(self, s, goals, c):
        return self.critic(torch.cat((s, goals.flatten(-2), c), -1)).squeeze(-1)

    def world_update(self, batch):
        s = batch["states"][0]
        loss = s.new_zeros(())
        alive = torch.ones(s.shape[0], device=s.device)
        for t, a in enumerate(batch["actions"]):
            s = self.world(s, a, batch["context"])
            error = self.layout.state_loss(s, batch["states"][t + 1])
            bce = F.binary_cross_entropy_with_logits(
                self.world.logits(s, batch["goals"][t + 1]),
                batch["terminal"][t].float(),
                reduction="none",
            )
            loss = loss + ((error + bce) * alive).sum() / alive.sum().clamp_min(1)
            # Include terminal transition, exclude reset and all later transitions.
            alive = alive * (~batch["boundary"][t]).float()
        loss = loss / len(batch["actions"])
        opt = self.optimizers[0]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.world.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        return float(loss.detach())

    def policy_update(self, s, history, previous, context, goals, horizon=32):
        # goals: [batch, horizon+10, state_dim], seeded at reference time t.
        rewards = []
        continuations = []
        states = []
        windows = []
        values = []
        survival = s.new_ones(s.shape[0])
        objective = s.new_zeros(s.shape[0])
        with frozen(self.world, self.critic):
            for t in range(horizon):
                g = goals[:, t : t + 10]
                states.append(s.detach())
                windows.append(g.detach())
                values.append(self.value(s, g, context))
                a = self.actor(history, g)
                ns = self.world(s, a, context)
                r = self.layout.reward(ns, goals[:, t + 1], a, previous, self.limits)
                c = 1 - torch.sigmoid(self.world.logits(ns, goals[:, t + 1]))
                objective = objective + 0.99**t * survival * r
                survival = survival * c
                history = torch.cat(
                    (history[:, 1:], self.layout.obs(ns, a, self.default, context=context)[:, None]), 1
                )
                s, previous = ns, a
                rewards.append(r)
                continuations.append(c)
            values.append(self.value(s, goals[:, horizon : horizon + 10], context))
            objective = objective + 0.99**horizon * survival * values[-1]
            loss = -objective.mean()
            opt = self.optimizers[1]
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), 1.0, error_if_nonfinite=True
            )
            for p in self.actor.parameters():
                if p.grad is not None:
                    p.grad.clamp_(-0.1, 0.1)
            opt.step()
        targets = lambda_returns(
            torch.stack(rewards).detach(),
            torch.stack(continuations).detach(),
            torch.stack(values).detach(),
        )
        ss = torch.stack(states).flatten(0, 1)
        gg = torch.stack(windows).flatten(0, 1)
        cc = context.repeat(horizon, 1)
        tt = targets.flatten()
        critic_loss = 0.0
        for _ in range(5):
            for idx in torch.randperm(len(ss), device=ss.device).tensor_split(4):
                if len(idx) == 0:
                    continue
                vl = F.mse_loss(self.value(ss[idx], gg[idx], cc[idx]), tt[idx])
                opt = self.optimizers[2]
                opt.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                critic_loss = float(vl.detach())
        return {"actor_loss": float(loss.detach()), "critic_loss": critic_loss}
