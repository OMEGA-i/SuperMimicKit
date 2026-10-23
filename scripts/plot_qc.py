"""Metric-scale trajectory QC from saved state, not a claim of rendered robot geometry."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("input", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
d = np.load(a.input, allow_pickle=False)
n = len(d["joint_names"])
names = d["body_names"].tolist()
actual = d["actual"] if "actual" in d else d["state"]
reference = d.get("reference", None)
pairs = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (7, 11),
    (11, 12),
    (12, 13),
]
if len(names) != 14:
    raise ValueError("Plot topology currently supports the G1 tracked-body schema")
frames = np.linspace(0, len(actual) - 1, 5).astype(int)
fig, axes = plt.subplots(2, 5, figsize=(15, 7), sharey=True)
for row, coordinate in enumerate([0, 1]):
    for col, t in enumerate(frames):
        ax = axes[row, col]
        center = actual[t, coordinate]
        for state, color, label in [
            (reference, "#ef9b32", "reference"),
            (actual, "#176fc1", "actual" if reference is not None else "reference"),
        ]:
            if state is None:
                continue
            pts = state[t, 13 + 2 * n :].reshape(len(names), 13)[:, :3]
            for i, j in pairs:
                ax.plot(
                    pts[[i, j], coordinate] - center, pts[[i, j], 2], color=color, lw=2
                )
            ax.scatter(
                pts[:, coordinate] - center, pts[:, 2], s=10, c=color, label=label
            )
        ax.set_ylim(0, 2.5)
        ax.set_xlim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.set_yticks(np.arange(0, 2.51, 0.5))
        ax.grid(alpha=0.3)
        ax.axhline(0, c="black")
        ax.set_title(
            f"t={t / float(d['fps']):.2f}s / {'world X–Z' if row == 0 else 'world Y–Z'}"
        )
        ax.set_xlabel("relative distance (m)")
axes[0, 0].set_ylabel("height (m)")
axes[1, 0].set_ylabel("height (m)")
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Saved tracked-body trajectories • fixed 2.5 m framing • 0.5 m grid")
fig.tight_layout(h_pad=3.0)
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=140)
