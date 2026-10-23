"""Motion and replay contracts. Replay stores pre-reset next states explicitly."""

import numpy as np
import torch


class Motions:
    def __init__(self, paths, device):
        self.clips = []
        self.names = []
        for path in paths:
            with np.load(path, allow_pickle=False) as d:
                if float(d["fps"]) != 50:
                    raise ValueError("Expected resampled 50 Hz motions")
                names = (d["joint_names"].tolist(), d["body_names"].tolist())
                if self.clips and names != self.names:
                    raise ValueError("Motion schema mismatch")
                self.names = names
                state = torch.tensor(d["state"], device=device, dtype=torch.float32)
                if len(state) < 42 or not torch.isfinite(state).all():
                    raise ValueError(f"Invalid clip: {path}")
                self.clips.append(state)
        if not self.clips:
            raise ValueError("No motion NPZ files")
        self.device = device
        self.lengths = torch.tensor([len(x) for x in self.clips], device=device)
        self.data = torch.cat(self.clips)
        self.offsets = torch.cat(
            (self.lengths.new_zeros(1), self.lengths.cumsum(0)[:-1])
        )

    def sample(self, n):
        # Uniform motion/time baseline; return-based sampling is a documented deviation.
        ids = torch.randint(len(self.clips), (n,), device=self.device)
        times = (torch.rand(n, device=self.device) * (self.lengths[ids] - 42)).long()
        return ids, times

    def get(self, ids, times, count=1):
        ts = times[:, None] + torch.arange(count, device=self.device)
        ts = torch.minimum(ts, self.lengths[ids, None] - 1)
        return self.data[self.offsets[ids, None] + ts]


class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = {}
        self.cursor = 0
        self.size = 0

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "cursor": self.cursor,
            "size": self.size,
            "storage": self.storage,
        }

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity mismatch")
        if (
            not 0 <= state["cursor"] < self.capacity
            or not 0 <= state["size"] <= self.capacity
        ):
            raise ValueError("Invalid replay cursor/size")
        if any(v.shape[0] != self.capacity for v in state["storage"].values()):
            raise ValueError("Replay storage shape mismatch")
        self.cursor, self.size = state["cursor"], state["size"]
        self.storage = state["storage"]

    def add(self, **transition):
        if not self.storage:
            self.storage = {
                k: v.new_empty((self.capacity, *v.shape)) for k, v in transition.items()
            }
        for key, value in transition.items():
            self.storage[key][self.cursor].copy_(value.detach())
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, steps=8):
        if self.size < steps:
            raise ValueError("Replay warmup incomplete")
        any_tensor = next(iter(self.storage.values()))
        device = any_tensor.device
        env = torch.randint(any_tensor.shape[1], (batch,), device=device)
        logical = torch.randint(self.size - steps + 1, (batch,), device=device)
        start = self.cursor if self.size == self.capacity else 0
        indices = (
            logical[None] + torch.arange(steps, device=device)[:, None] + start
        ) % self.capacity
        first_only = {"context", "history", "previous", "clip", "time", "state", "goal"}
        out = {k: v[indices[:1] if k in first_only else indices, env]
               for k, v in self.storage.items()}
        # Never use reset states as next-state targets.
        return {
            "states": torch.cat((out["state"][:1], out["next_state"]), 0),
            "actions": out["action"],
            "goals": torch.cat((out["goal"][:1], out["next_goal"]), 0),
            "terminal": out["terminal"],
            "boundary": out["boundary"],
            "context": out["context"][0],
            "history": out["history"][0],
            "previous": out["previous"][0],
            "clip": out["clip"][0],
            "time": out["time"][0],
        }


class ResetSampler:
    """D.1 mixture over 1 s bins, estimated from episodes starting in each bin.

    The paper does not specify the running estimator: use EMA 0.1 for returns
    and cumulative softmax weights for historical difficulty. Unseen returns
    start at zero. Returns include failed transitions, without bootstrapping.
    """

    def __init__(self, lengths, tail=42, gamma=0.99):
        self.gamma = gamma
        self.valid = (lengths - tail).clamp_min(1)
        counts = (self.valid + 49) // 50
        self.offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        self.clips = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), counts)
        self.starts = (torch.arange(len(self.clips), device=lengths.device) - self.offsets[self.clips]) * 50
        self.returns = torch.zeros(len(self.clips), device=lengths.device)
        self.past = torch.zeros_like(self.returns)
        self.updates = 0

    def probabilities(self):
        current = torch.softmax(-25 * self.returns / (0.1 / (1 - self.gamma)), 0)
        history = self.past / self.past.sum() if self.updates else torch.full_like(current, 1 / len(current))
        return 0.5 / len(current) + 0.4 * current + 0.1 * history

    def sample(self, n):
        bins = torch.multinomial(self.probabilities(), n, replacement=True)
        ids = self.clips[bins]
        width = (self.valid[ids] - self.starts[bins]).clamp_max(50)
        times = self.starts[bins] + (torch.rand(n, device=ids.device) * width).long()
        return ids, times

    def bin_ids(self, ids, times):
        return self.offsets[ids] + times // 50

    def update(self, bins, returns):
        if not len(bins):
            return
        if not torch.isfinite(returns).all():
            raise FloatingPointError("Nonfinite episode return")
        sums = torch.zeros_like(self.returns).scatter_add_(0, bins, returns)
        counts = torch.zeros_like(self.returns).scatter_add_(0, bins, torch.ones_like(returns))
        seen = counts > 0
        self.returns[seen] = self.returns[seen] * 0.9 + 0.1 * sums[seen] / counts[seen]
        self.past += torch.softmax(-25 * self.returns / (0.1 / (1 - self.gamma)), 0)
        self.updates += 1

    def state_dict(self):
        return {"returns": self.returns, "past": self.past, "updates": self.updates}

    def load_state_dict(self, state):
        for key in ("returns", "past"):
            value = state[key]
            if value.shape != self.returns.shape or not torch.isfinite(value).all():
                raise ValueError("Reset sampler state mismatch")
            setattr(self, key, value)
        self.updates = state["updates"]


def select_motion_paths(paths, split='train40', manifest_path=None):
    """Exact filename split shared by all embodiments; no retarget-dependent split."""
    import json
    from pathlib import Path
    manifest_path = Path(manifest_path) if manifest_path is not None else Path('datasets/lafan/manifest.json')
    manifest = json.loads(manifest_path.read_text())['splits']
    if manifest.get('status') != 'assigned':
        raise ValueError('Dataset splits are not assigned')
    if len(set(manifest['train'])) != 40 or len(set(manifest['test'])) != 37 or set(manifest['train']) & set(manifest['test']):
        raise ValueError('Expected disjoint train40/test37 dataset splits')
    if split == 'all77':
        wanted = set(manifest['train'] + manifest['test'])
    elif split == 'train40':
        wanted = set(manifest['train'])
    elif split == 'test37':
        wanted = set(manifest['test'])
    else:
        raise ValueError(f'Unknown motion split: {split}')
    found = {p.parent.name: p for p in paths}
    if len(found) != len(paths):
        raise ValueError('Duplicate motion names')
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'Missing {split} motions: {sorted(missing)}')
    return [found[name] for name in sorted(wanted)]
