"""Single motion-tracking core aligned with SuperMimic (arXiv:2610.28480).

CHECKPOINT_SCHEMA is a serialized feature contract, not a selectable learner.
Old checkpoint semantics are deliberately unsupported.
"""
import itertools
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .kinematics import cross3, qmul, qconj, qrot, qexp, qlog, yawq, rot6

CHECKPOINT_SCHEMA = 6
PAPER_REFERENCE = "https://arxiv.org/abs/2610.28480v1"


def validate_checkpoint(checkpoint):
    if checkpoint.get("learner_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint feature schema is unsupported by the current core; start a fresh run.")



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
        bootstrap = (1 - lam) * values[t + 1] + lam * target
        # IEEE 0 * NaN is NaN: stop a terminal tail before multiplication.
        bootstrap = torch.where(continuation[t] > 0, bootstrap, torch.zeros_like(bootstrap))
        target = rewards[t] + gamma * continuation[t] * bootstrap
        result.append(target)
    return torch.stack(result[::-1]).detach()

def matrix_rot6(q):
    # Flatten R[:, :2] in row-major order, not column by column.
    columns = rot6(q)
    return columns.reshape(*columns.shape[:-1], 2, 3).transpose(-1, -2).flatten(-2)

def rotation_error(q, r):
    # trace(Rq.T @ Rr) = 4 * <q,r>**2 - 1 for unit quaternions.
    dot = (F.normalize(q, dim=-1) * F.normalize(r, dim=-1)).sum(-1)
    return (2 * dot.square() - 1).clamp(-1 + 1e-6, 1 - 1e-6).acos().square()

def tracked_bodies(config, recorded_names):
    if config['robot'] == 'unitree_g1_29dof':
        return ['pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
                'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',
                'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
                'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link']
    # Other embodiments retain their explicit named-body contract.
    return list(recorded_names)

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
        self.distributed = False

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        n = torch.where(self.count < 2e8, float(len(x)), 0.0)
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        if self.distributed:
            from .distributed import global_moments
            global_n, m, v = global_moments(x)
            n = torch.where(self.count < 2e8, global_n, 0.0)
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

@dataclass
class Layout:
    joints: int
    bodies: int
    rotation_error = staticmethod(rotation_error)

    @property
    def dim(self):
        return 13 + 2 * self.joints + 13 * self.bodies

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
        return 3 * self.joints + 6

    @property
    def inv_dim(self):
        return 13 + 2 * self.joints

    @property
    def goal_dim(self):
        return 10 * (15 + 2 * self.joints)

    def invariant(self, s):
        r, q, qd, _ = self.parts(s)
        inv = qconj(r[..., 3:7])
        return torch.cat((r[..., 2:3], matrix_rot6(qmul(qconj(yawq(r[..., 3:7])), r[..., 3:7])),
                          qrot(inv, r[..., 7:10]), qrot(inv, r[..., 10:13]), q, qd), -1)

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
                matrix_rot6(qmul(h.expand_as(future[..., 3:7]), future[..., 3:7])),
                qrot(inv, future[..., 7:10]),
                qrot(inv, future[..., 10:13]),
                q[..., 1:11, :],
                0.1 * v[..., 1:11, :],
            ),
            -1,
        ).flatten(-2)

    def obs(self, s, previous_action, default, noise=True, context=None, goal=None):
        r, q, v, _ = self.parts(s)
        gravity = torch.zeros_like(r[..., :3])
        gravity[..., 2] = -1
        angular = rotate_inverse(r[..., 3:7], r[..., 10:13])
        gravity = rotate_inverse(r[..., 3:7], gravity)
        if context is not None and context.shape[-1] > 1:
            default = default + context[..., 1 : 1 + self.joints]
        o = torch.cat((q - default, v, angular, gravity, previous_action), -1)
        return self.add_noise(o) if noise else o

    def add_noise(self, observation):
        """Table A.1 sensor noise; executed previous-action channels stay exact."""
        scales = observation.new_tensor(
            [0.01] * self.joints + [0.5] * self.joints + [0.2] * 3
            + [0.05] * 3 + [0] * self.joints
        )
        return observation + (2 * torch.rand_like(observation) - 1) * scales

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
        reward = reward + torch.exp(-self.rotation_error(b[..., 3:7], bq).mean(-1) / 0.4**2)
        for part, sigma in [(slice(7, 10), 1.0), (slice(10, 13), 3.14)]:
            reward = reward + torch.exp(
                -(b[..., part] - gb[..., part]).square().sum(-1).mean(-1) / sigma**2
            )
        reward = reward + 0.5 * torch.exp(
            -(r[..., :3] - gr[..., :3]).square().sum(-1) / 0.3**2
        )
        reward = reward + 0.5 * torch.exp(
            -self.rotation_error(r[..., 3:7], gr[..., 3:7]) / 0.4**2
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

    def context_features(self, c):
        return torch.cat(((c[..., :1]-.75)/.45, c[..., 1:-3]/.01, c[..., -3:]/.05), -1)

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
        # Eq. 1: squared norm of the minimal simulator state, not a loss whose
        # per-coordinate weights drift with the running delta normalizer.
        # Align the quaternion sign: q and -q denote the same physical state.
        predicted = predicted[..., :self.layout.inv_dim]
        target = target[..., :self.layout.inv_dim]
        sign = torch.where((predicted[..., 3:7] * target[..., 3:7]).sum(-1, keepdim=True) < 0, -1., 1.)
        target = torch.cat((target[..., :3], target[..., 3:7] * sign, target[..., 7:]), -1)
        return (predicted - target).square().sum(-1)

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
        z = torch.tanh(run_mlp(self.encoder, self.goal_norm(self.layout.goals(goals)), self.amp))
        return run_mlp(self.decoder, torch.cat((z, self.obs_norm(history).flatten(-2)), -1), self.amp)

