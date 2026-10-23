"""Differentiable named-body FK from the exact compiled robot model.

Supports fixed bodies, one floating root, and hinge joints with nonzero anchors,
reference angles and body rotations. Link-origin velocities match mj_jacBody.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def cross3(a, b):
    # Explicit components avoid an Inductor linalg_cross backward indexing failure
    # seen in the fused FP32 imagination step at batch=4096.
    ax, ay, az = a.unbind(-1)
    bx, by, bz = b.unbind(-1)
    return torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), -1)


def qmul(a, b):
    av, bv = a[..., 1:], b[..., 1:]
    return torch.cat(
        (
            a[..., :1] * b[..., :1] - (av * bv).sum(-1, keepdim=True),
            a[..., :1] * bv + b[..., :1] * av + cross3(av, bv),
        ),
        -1,
    )


def qconj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def qrot(q, v):
    xyz, v = torch.broadcast_tensors(q[..., 1:], v)
    t = 2 * cross3(xyz, v)
    return v + q[..., :1] * t + cross3(xyz, t)


def qexp(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat(
        (torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * torch.pi)) * v), -1
    )


def qlog(q):
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    n = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    # atan2 regularized only at zero; finite derivative and identity value.
    scale = 2 * torch.atan2(n, q[..., :1].clamp_min(1e-8)) / n.clamp_min(1e-7)
    return q[..., 1:] * torch.where(n > 1e-7, scale, 2 * torch.ones_like(scale))


def yawq(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)


def rot6(q):
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    return qrot(q[..., None, :], eye[:2]).flatten(-2)


class Kinematics(nn.Module):
    def __init__(self, model, joint_names, body_names):
        super().__init__()
        import mujoco as mj

        free = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("FK requires exactly one floating root")
        root = int(model.jnt_bodyid[free[0]])
        joint_map = {
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n): i
            for i, n in enumerate(joint_names)
        }
        self.parents = []
        self.joints = []
        bids = []
        pos = []
        quat = []
        axes = []
        anchors = []
        refs = []
        mapping = {}
        for b in range(root, model.nbody):
            if b != root and int(model.body_parentid[b]) not in mapping:
                raise ValueError("Bodies outside floating-root tree are unsupported")
            mapping[b] = len(bids)
            bids.append(b)
            self.parents.append(
                -1 if b == root else mapping[int(model.body_parentid[b])]
            )
            n = int(model.body_jntnum[b])
            j = int(model.body_jntadr[b])
            if b == root:
                if n != 1:
                    raise ValueError("Unexpected root joints")
                idx = -1
            elif n == 0:
                idx = -1
            elif (
                n == 1
                and model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE
                and j in joint_map
            ):
                idx = joint_map[j]
            else:
                raise ValueError(f"Unsupported joints on {model.body(b).name}")
            self.joints.append(idx)
            pos.append(model.body_pos[b])
            quat.append(model.body_quat[b])
            axes.append(model.jnt_axis[j] if idx >= 0 else np.array([1.0, 0, 0]))
            anchors.append(model.jnt_pos[j] if idx >= 0 else np.zeros(3))
            refs.append(model.qpos0[model.jnt_qposadr[j]] if idx >= 0 else 0.0)
        self.tracked = [
            mapping[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)] for n in body_names
        ]
        if sorted(j for j in self.joints if j >= 0) != list(range(len(joint_names))):
            raise ValueError("FK joint coverage mismatch")
        for name, value in [
            ("offset", pos),
            ("rotation", quat),
            ("axis", axes),
            ("anchor", anchors),
            ("reference", refs),
        ]:
            self.register_buffer(
                name, torch.tensor(np.asarray(value), dtype=torch.float32)
            )

    def forward(self, root, q, qd):
        P, Q, V, W = [], [], [], []
        for i, (parent, j) in enumerate(zip(self.parents, self.joints)):
            if parent < 0:
                p, r, v, w = (
                    root[..., :3],
                    root[..., 3:7],
                    root[..., 7:10],
                    root[..., 10:13],
                )
            else:
                r0 = qmul(Q[parent], self.rotation[i].expand_as(Q[parent]))
                p0 = P[parent] + qrot(Q[parent], self.offset[i])
                v0 = V[parent] + cross3(W[parent], p0 - P[parent])
                if j >= 0:
                    r = qmul(
                        r0, qexp(self.axis[i] * (q[..., j : j + 1] - self.reference[i]))
                    )
                    a0 = qrot(r0, self.anchor[i])
                    a1 = qrot(r, self.anchor[i])
                    p = p0 + a0 - a1
                    joint_w = qrot(r0, self.axis[i]) * qd[..., j : j + 1]
                    w = W[parent] + joint_w
                    v = v0 + cross3(W[parent], a0) - cross3(w, a1)
                else:
                    p, r, v, w = p0, r0, v0, W[parent]
            P.append(p)
            Q.append(r)
            V.append(v)
            W.append(w)
        return torch.stack(
            [torch.cat((P[i], Q[i], V[i], W[i]), -1) for i in self.tracked], -2
        )
