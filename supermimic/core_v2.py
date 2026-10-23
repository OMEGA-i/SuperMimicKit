"""Corrected learner: local minimal dynamics, exact FK, aligned rewards and norms.

Legacy core.py stays frozen to load old checkpoints and the existing web demo.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .core import Layout as LegacyLayout, frozen, lambda_returns, mlp, rotation_error
from .kinematics import qmul, qconj, qrot, qexp, qlog, yawq, rot6

VERSION = 2


def run_mlp(net, x, amp):
    # Geometry, normalizers and losses always remain FP32.
    if amp == "bf16" and x.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(x).float()
    return net(x)


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        n = torch.where(self.count < 2e8, float(len(x)), 0.0)
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        total = self.count + n
        delta = m - self.mean
        self.var.copy_(
            (self.var * self.count + v * n + delta.square() * self.count * n / total)
            / total
        )
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def std(self):
        return (self.var + self.eps**2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std()


class Layout(LegacyLayout):
    def parts(self, s):
        d = self.joints
        # Minimal states are used inside dynamics; full states at public boundaries.
        body = s[..., 13 + 2 * d :]
        return (
            s[..., :13],
            s[..., 13 : 13 + d],
            s[..., 13 + d : 13 + 2 * d],
            body.reshape(*s.shape[:-1], body.shape[-1] // 13, 13),
        )

    @property
    def obs_dim(self):
        return 3 * self.joints + 12

    @property
    def inv_dim(self):
        return 13 + 2 * self.joints

    @property
    def goal_dim(self):
        return 10 * (15 + 2 * self.joints)

    def invariant(self, s):
        r, q, v, _ = self.parts(s)
        inv = qconj(r[..., 3:7])
        h = yawq(r[..., 3:7])
        return torch.cat(
            (
                r[..., 2:3],
                rot6(qmul(qconj(h), r[..., 3:7])),
                qrot(inv, r[..., 7:10]),
                qrot(inv, r[..., 10:13]),
                q,
                v,
            ),
            -1,
        )

    def goals(self, seq):
        # Sequence contains current reference plus ten future frames.
        r, q, v, _ = self.parts(seq)
        now = r[..., 0, :]
        future = r[..., 1:11, :]
        h = qconj(yawq(now[..., 3:7]))[..., None, :]
        inv = qconj(future[..., 3:7])
        return torch.cat(
            (
                qrot(h, future[..., :3] - now[..., None, :3]),
                rot6(qmul(h.expand_as(future[..., 3:7]), future[..., 3:7])),
                qrot(inv, future[..., 7:10]),
                qrot(inv, future[..., 10:13]),
                q[..., 1:11, :],
                0.1 * v[..., 1:11, :],
            ),
            -1,
        ).flatten(-2)

    def obs(self, s, previous_action, default, noise=True, context=None, goal=None):
        if goal is None:
            raise ValueError("v2 observation requires current reference")
        o = super().obs(s, previous_action, default, noise=False, context=context)
        r = self.parts(s)[0]
        gr = self.parts(goal)[0]
        rel = rot6(qmul(qconj(r[..., 3:7]), gr[..., 3:7]))
        o = torch.cat((o, rel), -1)
        if noise:
            scales = o.new_tensor(
                [0.01] * self.joints
                + [0.5] * self.joints
                + [0.2] * 3
                + [0.05] * 3
                + [0] * self.joints
                + [0.05] * 6
            )
            o = o + (2 * torch.rand_like(o) - 1) * scales
        return o

    def reward(self, s, goal, action, previous_action, limits):
        r, q, _, b = self.parts(s)
        gr, _, _, gb = self.parts(goal)
        delta = yawq(qmul(r[..., 3:7], qconj(gr[..., 3:7])))
        anchor = torch.cat((r[..., :2], gr[..., 2:3]), -1)
        bp = anchor[..., None, :] + qrot(
            delta[..., None, :], gb[..., :3] - gr[..., None, :3]
        )
        bq = qmul(delta[..., None, :].expand_as(gb[..., 3:7]), gb[..., 3:7])
        reward = torch.exp(-(b[..., :3] - bp).square().sum(-1).mean(-1) / 0.3**2)
        reward = reward + torch.exp(-rotation_error(b[..., 3:7], bq).mean(-1) / 0.4**2)
        for part, sigma in [(slice(7, 10), 1.0), (slice(10, 13), 3.14)]:
            reward = reward + torch.exp(
                -(b[..., part] - gb[..., part]).square().sum(-1).mean(-1) / sigma**2
            )
        reward = reward + 0.5 * torch.exp(
            -(r[..., :3] - gr[..., :3]).square().sum(-1) / 0.3**2
        )
        reward = reward + 0.5 * torch.exp(
            -rotation_error(r[..., 3:7], gr[..., 3:7]) / 0.4**2
        )
        # 90% soft joint bounds, with unlimited joints excluded without inf-inf.
        finite = torch.isfinite(limits).all(-1)
        safe = torch.where(finite[:, None], limits, torch.zeros_like(limits))
        mid = safe.mean(-1)
        half = (safe[:, 1] - safe[:, 0]) * 0.45
        penalty = (F.relu(q - (mid + half)) + F.relu((mid - half) - q)) * finite
        return 0.02 * (
            reward
            - 0.1 * (action - previous_action).square().sum(-1)
            - 2 * penalty.sum(-1)
        )

    def continuation(self, s, goal):
        r = self.parts(s)[0]
        gr = self.parts(goal)[0]
        ez = (r[..., 2] - gr[..., 2]).abs()
        g = torch.zeros_like(r[..., :3])
        g[..., 2] = -1
        tilt = (
            qrot(qconj(r[..., 3:7]), g)[..., 2] - qrot(qconj(gr[..., 3:7]), g)[..., 2]
        ).abs()
        return (1 - torch.sigmoid((ez - 0.25) / 0.02)) * (
            1 - torch.sigmoid((tilt - 0.8) / 0.04)
        )


class WorldModel(nn.Module):
    def __init__(self, layout, context_dim, kinematics, width=2048, amp="none"):
        super().__init__()
        self.layout = layout
        self.amp = amp
        self.kinematics = kinematics
        ind = layout.inv_dim + layout.joints + context_dim
        outd = 12 + 2 * layout.joints
        layers = []
        d = ind
        for _ in range(3):
            layers.extend([nn.Linear(d, width), nn.LayerNorm(width), nn.SiLU()])
            d = width
        layers.append(nn.Linear(d, outd))
        self.dynamics = nn.Sequential(*layers)
        nn.init.zeros_(self.dynamics[-1].weight)
        nn.init.zeros_(self.dynamics[-1].bias)
        self.in_norm = RunningNorm(ind)
        self.out_norm = RunningNorm(outd, eps=1e-4)
        self.register_buffer(
            "error_floor",
            torch.tensor(
                [0.003] * 3
                + [0.005] * 3
                + [0.02] * 3
                + [0.05] * 3
                + [0.003] * layout.joints
                + [0.05] * layout.joints
            ),
        )

    def context_features(self, c):
        return torch.cat(
            (
                (c[..., :1] - 0.75) / 0.45,
                c[..., 1:-3] / 0.01,
                c[..., -3:] / c.new_tensor([0.025, 0.05, 0.05]),
            ),
            -1,
        )

    def inputs(self, s, a, c):
        return torch.cat((self.layout.invariant(s), a, self.context_features(c)), -1)

    def delta(self, s, target):
        r, q, v, _ = self.layout.parts(s)
        n, nq, nv, _ = self.layout.parts(target)
        inv = qconj(r[..., 3:7])
        return torch.cat(
            (
                qrot(qconj(yawq(r[..., 3:7])), n[..., :3] - r[..., :3]),
                qlog(qmul(inv, n[..., 3:7])),
                qrot(inv, n[..., 7:10] - r[..., 7:10]),
                qrot(inv, n[..., 10:13] - r[..., 10:13]),
                nq - q,
                nv - v,
            ),
            -1,
        )

    @torch.no_grad()
    def update_norms(self, s, a, c, ns):
        self.in_norm.update(self.inputs(s, a, c))
        self.out_norm.update(self.delta(s, ns))

    def minimal(self, s, a, c):
        r, q, v, _ = self.layout.parts(s)
        d = (
            run_mlp(self.dynamics, self.in_norm(self.inputs(s, a, c)), self.amp)
            * self.out_norm.std()
            + self.out_norm.mean
        )
        dp, dr, dv, dw, dq, dqd = d.split(
            [3, 3, 3, 3, self.layout.joints, self.layout.joints], -1
        )
        quat = F.normalize(qmul(r[..., 3:7], qexp(dr)), dim=-1)
        root = torch.cat(
            (
                r[..., :3] + qrot(yawq(r[..., 3:7]), dp),
                quat,
                r[..., 7:10] + qrot(r[..., 3:7], dv),
                r[..., 10:13] + qrot(r[..., 3:7], dw),
            ),
            -1,
        )
        q = q + dq
        v = v + dqd
        return torch.cat((root, q, v), -1)

    def full(self, s):
        root, q, v, _ = self.layout.parts(s)
        body = self.kinematics(root, q, v)
        return torch.cat((root, q, v, body.flatten(-2)), -1)

    def forward(self, s, a, c):
        return self.full(self.minimal(s, a, c))

    def error(self, predicted, target):
        return (
            (
                self.delta(target, predicted)
                / torch.maximum(self.out_norm.std(), self.error_floor)
            )
            .square()
            .mean(-1)
        )


class Actor(nn.Module):
    def __init__(self, layout, small=False, amp="none"):
        super().__init__()
        self.layout = layout
        self.amp = amp
        widths = [64, 64] if small else [2048, 1024, 512, 256]
        self.encoder = mlp([layout.goal_dim, *widths, 32])
        self.decoder = mlp([10 * layout.obs_dim + 32, *widths, layout.joints])
        self.obs_norm = RunningNorm(layout.obs_dim)
        self.goal_norm = RunningNorm(layout.goal_dim)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.01)
            self.decoder[-1].bias.zero_()

    @torch.no_grad()
    def update_norms(self, obs, goals):
        self.obs_norm.update(obs)
        self.goal_norm.update(self.layout.goals(goals))

    def forward(self, history, goals):
        z = torch.tanh(
            run_mlp(self.encoder, self.goal_norm(self.layout.goals(goals)), self.amp)
        )
        return run_mlp(
            self.decoder,
            torch.cat((self.obs_norm(history).flatten(-2), z), -1),
            self.amp,
        )


class Critic(nn.Module):
    def __init__(self, dim, amp="none"):
        super().__init__()
        self.amp = amp
        self.norm = RunningNorm(dim)
        self.net = mlp([dim, 256, 128, 1])

    def forward(self, x):
        return run_mlp(self.net, self.norm(x), self.amp).squeeze(-1)


class Learner:
    def __init__(
        self,
        layout,
        default,
        limits,
        context_dim,
        kinematics,
        device="cpu",
        small=False,
        lr=2e-4,
        compile=False,
        optimized=True,
        amp="none",
        runtime_tuning=True,
        cuda_graphs=False,
    ):
        self.layout, self.default, self.limits = layout, default, limits
        self.world = WorldModel(
            layout, context_dim, kinematics, 64 if small else 2048, amp=amp
        ).to(device)
        self.actor = Actor(layout, small, amp=amp).to(device)
        self.critic = Critic(
            layout.inv_dim + layout.goal_dim + context_dim + 9, amp=amp
        ).to(device)
        self.cuda_graphs = cuda_graphs

        def compile_fn(fn):
            return torch.compile(
                fn, mode="reduce-overhead" if cuda_graphs else "default"
            )

        fused = runtime_tuning and str(device).startswith("cuda")
        self.optimizers = [
            torch.optim.Adam(self.world.parameters(), lr=lr, fused=fused),
            torch.optim.Adam(
                self.actor.parameters(), lr=lr, betas=(0.9, 0.95), fused=fused
            ),
            torch.optim.Adam(self.critic.parameters(), lr=lr, fused=fused),
        ]
        self.collect_actor = (
            compile_fn(self.actor.forward)
            if compile and runtime_tuning
            else self.actor.forward
        )
        self.observe_step = (
            compile_fn(self.observe) if compile and runtime_tuning else self.observe
        )
        self.critic_loss_step = (
            compile_fn(self._critic_loss)
            if compile and runtime_tuning
            else self._critic_loss
        )
        self.collect_metrics = (
            compile_fn(self._collect_metrics)
            if compile and runtime_tuning
            else self._collect_metrics
        )
        self.optimized = optimized
        self.world_loss = compile_fn(self._world_loss) if compile else self._world_loss
        self.imag_step = compile_fn(self._imag_step) if compile else self._imag_step
        self.reward_batch = (
            compile_fn(self._reward_batch) if compile else self._reward_batch
        )
        self.features = (
            compile_fn(self.critic_features) if compile else self.critic_features
        )
        self.world_step = (
            compile_fn(self.world.forward) if compile else self.world.forward
        )
        self.reward = compile_fn(layout.reward) if compile else layout.reward
        self.obs = compile_fn(layout.obs) if compile else layout.obs

    def critic_features(self, s, goals, c):
        r = self.layout.parts(s)[0]
        g = self.layout.parts(goals[:, 0])[0]
        err = qrot(qconj(yawq(r[..., 3:7])), g[..., :3] - r[..., :3])
        rel = rot6(qmul(qconj(r[..., 3:7]), g[..., 3:7]))
        return torch.cat(
            (
                self.layout.invariant(s),
                self.layout.goals(goals),
                self.world.context_features(c),
                err,
                rel,
            ),
            -1,
        )

    def _collect_metrics(self, state, goal, action, previous, limits, context):
        return (
            self.layout.done(state, goal),
            self.layout.reward(state, goal, action, previous, limits),
            self.layout.obs(state, action, self.default, context=context, goal=goal),
        )

    def _critic_loss(self, x, target):
        return F.mse_loss(self.critic(x), target)

    def value(self, s, g, c):
        return self.critic(self.critic_features(s, g, c))

    @torch.no_grad()
    def observe(self, s, a, c, ns, history, goals):
        self.world.update_norms(s, a, c, ns)
        self.actor.update_norms(history[:, -1], goals)

    def world_update(self, batch):
        if self.cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        if self.optimized:
            loss, last = self.world_loss(
                batch["states"], batch["actions"], batch["context"], batch["boundary"]
            )
        else:
            s = batch["states"][0]
            alive = torch.ones(len(s), device=s.device)
            loss = s.new_zeros(())
            last = s.new_zeros(())
            for t, a in enumerate(batch["actions"]):
                s = self.world_step(s, a, batch["context"])
                e = self.world.error(s, batch["states"][t + 1])
                last = (e * alive).sum() / alive.sum().clamp_min(1)
                loss = loss + (e * alive).sum() / len(s)
                alive = alive * (~batch["boundary"][t]).float()
        opt = self.optimizers[0]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.world.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        self.last_error8 = float(last.detach())
        return float(loss.detach() / len(batch["actions"]))

    def _world_loss(self, states, actions, context, boundary):
        s = states[0, :, : self.layout.inv_dim]
        preds = []
        for a in actions:
            s = self.world.minimal(s, a, context)
            preds.append(s)
        e = self.world.error(torch.stack(preds), states[1:])
        alive = torch.cat(
            (torch.ones_like(e[:1]), (~boundary[:-1]).float().cumprod(0)), 0
        )
        return (e * alive).sum() / len(s), (e[-1] * alive[-1]).sum() / alive[
            -1
        ].sum().clamp_min(1)

    def _imag_step(self, s, history, context, goals):
        a = self.actor(history, goals[:, :11])
        ns = self.world.minimal(s, a, context)
        obs = self.layout.obs(ns, a, self.default, context=context, goal=goals[:, 1])
        return ns, a, torch.cat((history[:, 1:], obs[:, None]), 1)

    def _reward_batch(self, s, goals, actions, previous):
        full = self.world.full(s)
        return (
            self.layout.reward(full, goals, actions, previous, self.limits),
            self.layout.continuation(s, goals),
        )

    def policy_update(self, s, history, previous, context, goals, horizon=32):
        if self.cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        rewards = []
        continuations = []
        features = []
        survival = s.new_ones(len(s))
        objective = s.new_zeros(len(s))
        with frozen(self.world, self.critic):
            if self.optimized:
                states = [s[..., : self.layout.inv_dim]]
                actions = [previous]
                for t in range(horizon):
                    s, a, history = self.imag_step(
                        states[-1], history, context, goals[:, t : t + 11]
                    )
                    states.append(s)
                    actions.append(a)
                rs, cs = self.reward_batch(
                    torch.stack(states[1:]),
                    goals[:, 1 : horizon + 1].transpose(0, 1),
                    torch.stack(actions[1:]),
                    torch.stack(actions[:-1]),
                )
                alive = cs.cumprod(0)
                before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
                discounts = (0.99 ** torch.arange(horizon, device=s.device))[:, None]
                objective = (discounts * before * rs).sum(0)
                survival = alive[-1]
                rewards, continuations = list(rs.unbind(0)), list(cs.unbind(0))
                with torch.no_grad():
                    windows = (
                        goals.unfold(1, 11, 1)
                        .movedim(-1, -2)[:, :horizon]
                        .transpose(0, 1)
                    )
                    f = self.features(
                        torch.stack(states[:-1]).detach().flatten(0, 1),
                        windows.flatten(0, 1),
                        context.expand(horizon, -1, -1).flatten(0, 1),
                    )
                    features = list(f.reshape(horizon, len(s), -1).unbind(0))
            else:
                for t in range(horizon):
                    g = goals[:, t : t + 11]
                    features.append(self.critic_features(s, g, context).detach())
                    a = self.actor(history, g)
                    ns = self.world_step(s, a, context)
                    r = self.reward(ns, goals[:, t + 1], a, previous, self.limits)
                    c = self.layout.continuation(ns, goals[:, t + 1])
                    objective = objective + 0.99**t * survival * r
                    survival = survival * c
                    history = torch.cat(
                        (
                            history[:, 1:],
                            self.obs(
                                ns,
                                a,
                                self.default,
                                context=context,
                                goal=goals[:, t + 1],
                            )[:, None],
                        ),
                        1,
                    )
                    s, previous = ns, a
                    rewards.append(r)
                    continuations.append(c)
            final_features = self.critic_features(
                s, goals[:, horizon : horizon + 11], context
            )
            objective = objective + 0.99**horizon * survival * self.critic(
                final_features
            )
            loss = -objective.mean()
            opt = self.optimizers[1]
            opt.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(
                self.actor.parameters(), 1.0, error_if_nonfinite=True
            )
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            opt.step()
        with torch.no_grad():
            x = torch.stack(features + [final_features.detach()])
            self.critic.norm.update(x)
            values = self.critic(x)
            targets = lambda_returns(
                torch.stack(rewards).detach(),
                torch.stack(continuations).detach(),
                values,
            )
        xx = x[:-1].flatten(0, 1)
        tt = targets.flatten()
        critic_loss = 0.0
        for _ in range(5):
            for idx in torch.randperm(len(xx), device=xx.device).tensor_split(4):
                if len(idx) == 0:
                    continue
                vl = self.critic_loss_step(xx[idx], tt[idx])
                opt = self.optimizers[2]
                opt.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                critic_loss = vl.detach()
        return {
            "actor_loss": float(loss.detach()),
            "critic_loss": float(critic_loss),
            "imag_survival": float(survival.detach().mean()),
            "actor_grad_norm": float(gn),
            "imag_reward": float(torch.stack(rewards).detach().mean()),
        }
