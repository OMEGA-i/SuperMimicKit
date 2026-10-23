"""Render saved PhysX/reference keypoint trajectories to a synchronized QC video."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("rollout", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
data = np.load(a.rollout, allow_pickle=False)
n = len(data["joint_names"])
names = data["body_names"].tolist()
if len(names) != 14:
    raise ValueError("Expected the G1 tracked body schema")
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
actual = data["actual"]
ref = data["reference"]
fps = float(data["fps"])
fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
with imageio.get_writer(
    a.output, fps=fps / 2, codec="libx264", macro_block_size=2
) as writer:
    for t in range(0, len(actual), 2):
        for ax, coordinate in zip(axes, [0, 1]):
            ax.clear()
            center = ref[t, coordinate]
            for state, color, label in [
                (ref, "#ef9b32", "reference"),
                (actual, "#176fc1", "PhysX"),
            ]:
                pos = state[t, 13 + 2 * n :].reshape(14, 13)[:, :3]
                for i, j in pairs:
                    ax.plot(
                        pos[[i, j], coordinate] - center,
                        pos[[i, j], 2],
                        color=color,
                        lw=2,
                    )
                ax.scatter(
                    pos[:, coordinate] - center, pos[:, 2], c=color, s=12, label=label
                )
            ax.set(
                xlim=(-1.25, 1.25),
                ylim=(0, 2.5),
                xlabel="reference-relative distance (m)",
                ylabel="height (m)",
            )
            ax.set_yticks(np.arange(0, 2.51, 0.5))
            ax.grid(alpha=0.3)
            ax.set_aspect("equal")
            ax.legend(loc="upper right")
            ax.set_title("world X–Z" if coordinate == 0 else "world Y–Z")
        fig.suptitle(
            f"Saved PhysX rollout / reference — {t / fps:.2f} s\nTracked-body diagnostic; not a robot mesh render"
        )
        fig.tight_layout()
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
