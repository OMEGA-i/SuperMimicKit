"""Read LaFAN BVH channels into global Z-up poses without a retarget dependency."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_bvh(path):
    lines = Path(path).read_text().splitlines()
    names, parents, offsets, channels = [], [], [], []
    stack = []
    pending = None
    cursor = 0
    for index, line in enumerate(lines):
        words = line.strip().split()
        if not words:
            continue
        if words[0] in ("ROOT", "JOINT"):
            pending = len(names)
            names.append(words[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            channels.append(None)
        elif words[:2] == ["End", "Site"]:
            pending = -1
        elif words[0] == "{":
            stack.append(pending)
        elif words[0] == "}":
            stack.pop()
        elif words[0] == "OFFSET" and stack[-1] >= 0:
            offsets[stack[-1]] = list(map(float, words[1:]))
        elif words[0] == "CHANNELS":
            count = int(words[1])
            channels[stack[-1]] = (cursor, words[2:])
            if count != len(words[2:]):
                raise ValueError("Malformed BVH channels")
            cursor += count
        elif words[0] == "MOTION":
            break
    else:
        raise ValueError("BVH has no MOTION section")
    count = int(lines[index + 1].split(":")[1])
    dt = float(lines[index + 2].split(":")[1])
    values = np.loadtxt(lines[index + 3 :], ndmin=2)
    if values.shape != (count, cursor) or not np.isfinite(values).all() or dt <= 0:
        raise ValueError("Invalid BVH motion data")
    positions = np.empty((count, len(names), 3))
    rotations = []
    basis = Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    for j, (start, labels) in enumerate(channels):
        local_pos = np.broadcast_to(offsets[j], (count, 3)).copy()
        angles = []
        axes = ""
        for k, label in enumerate(labels):
            axis = label[0].lower()
            if label.endswith("position"):
                local_pos[:, "xyz".index(axis)] = values[:, start + k]
            elif label.endswith("rotation"):
                axes += axis.upper()
                angles.append(values[:, start + k])
            else:
                raise ValueError(f"Unknown BVH channel {label}")
        local_rot = Rotation.from_euler(axes, np.stack(angles, axis=-1), degrees=True)
        parent = parents[j]
        if parent < 0:
            rotations.append(basis * local_rot)
            positions[:, j] = basis.apply(local_pos) / 100
        else:
            rotations.append(rotations[parent] * local_rot)
            positions[:, j] = (
                positions[:, parent] + rotations[parent].apply(local_pos) / 100
            )
    quaternions = np.stack([r.as_quat(scalar_first=True) for r in rotations], axis=1)
    return names, np.asarray(parents), positions, quaternions, 1 / dt
