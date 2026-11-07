# Course weights and test results — 2026-11-07

[Download the 16 weight bundles](https://github.com/OMEGA-i/SuperMimicKit/releases/tag/courses-20261107).

G1, H2, A3 and SIH2300; four independent courses each. These are experimental
validation-selected models, not claims of convergence or paper-equivalent performance.
The release is limited to the four embodiments listed above.

## Training design

| Course | Training |
|---|---|
| a | LaFAN, 50k iterations |
| b | BONES-SEED, 50k iterations |
| c | Initialize from a at 25k, then BONES-SEED to total 50k |
| d | Initialize from b at 25k, then LaFAN to total 50k |

4,096 environments, FP32, seed 0, MuJoCo Warp; checkpoint and full train/validation
evaluation every 5k. Dataset manifests use train/val/test 8:1:1 with integer rounding.
The destination dataset's validation mean selects the checkpoint, not test performance.
Each bundle records actual saved robot configuration and scalar training options.
Portable configurations are also checked into [configs](configs); resolve their relative
dataset paths against your own data root. The c/d initialization source is the corresponding
a/b checkpoint at 25k, not the validation-best weights published here.
G1 b was deliberately stopped early (selected 30k); SIH2300 d was interrupted at 45k
(selected 35k). The remaining 14 courses reached 50k. A completed course can still regress;
selected iteration is not necessarily the last iteration.

## Test results

Cells contain mean normalized return · failed clips / total. All three datasets use
the same selected checkpoint per course. See [machine-readable results](test-results.csv).

| Robot | Course | Selected iteration | LaFAN | BONES-SEED | GRAIL |
|---|---|---:|---|---|---|
| G1 | a | 45000 | 0.30827 · 5/8 | 0.77154 · 158/13023 | 0.27936 · 1468/2079 |
| G1 | b | 30000 | 0.25527 · 5/8 | 0.86718 · 106/13023 | 0.33976 · 1485/2079 |
| G1 | c | 45000 | 0.27179 · 6/8 | 0.87214 · 27/13023 | 0.34721 · 1437/2079 |
| G1 | d | 50000 | 0.31967 · 5/8 | 0.81217 · 120/13023 | 0.31356 · 1460/2079 |
| H2 | a | 20000 | 0.26007 · 6/8 | 0.60266 · 367/13023 | 0.24337 · 1660/2079 |
| H2 | b | 40000 | 0.10436 · 7/8 | 0.70661 · 96/13023 | 0.24831 · 1545/2079 |
| H2 | c | 35000 | 0.21991 · 6/8 | 0.77638 · 77/13023 | 0.28388 · 1564/2079 |
| H2 | d | 50000 | 0.24946 · 5/8 | 0.58343 · 205/13023 | 0.23913 · 1619/2079 |
| A3 | a | 15000 | 0.27106 · 5/8 | 0.72038 · 224/13023 | 0.24855 · 1560/2079 |
| A3 | b | 25000 | 0.04931 · 8/8 | 0.85662 · 113/13023 | 0.25889 · 1576/2079 |
| A3 | c | 30000 | 0.15858 · 7/8 | 0.82509 · 65/13023 | 0.27897 · 1546/2079 |
| A3 | d | 40000 | 0.28243 · 5/8 | 0.74782 · 148/13023 | 0.26296 · 1496/2079 |
| SIH2300 | a | 20000 | 0.27165 · 5/8 | 0.67802 · 215/13023 | 0.25529 · 1592/2079 |
| SIH2300 | b | 15000 | 0.02174 · 8/8 | 0.75176 · 286/13023 | 0.19002 · 1640/2079 |
| SIH2300 | c | 35000 | 0.05255 · 8/8 | 0.85054 · 46/13023 | 0.27228 · 1575/2079 |
| SIH2300 | d | 35000 | 0.02055 · 8/8 | 0.18142 · 5257/13023 | 0.11266 · 1673/2079 |

Evaluation starts at reference frame 0 with noise and domain randomization disabled.
Return is the arithmetic mean of per-clip reward sum divided by 0.1 times the full
transition count; frames after early termination contribute zero. Failure means early
termination, not a threshold on mean return. Detailed protocol and per-motion results
are included in each archive. GRAIL uses paired scenes and recorded 50 Hz object
trajectories: this is **kinematic-object evaluation**, not free-object manipulation.

## Loading and scope

Bundles contain `weights.pt` (actor and world-model state dictionaries including
normalizers), robot/training configuration, metadata/checksums and per-motion test
results. No replay, optimizer, raw training checkpoint, motion data or robot meshes
are distributed. Load with `torch.load(..., weights_only=True)`.

The repository's older preview uses a different feature schema. Use the accompanying
`runtime` inference snapshot, not the legacy training core. With Python 3.11+, PyTorch,
NumPy and MuJoCo installed, download `inference-runtime.zip` (or use the `runtime` directory here), add its parent to `sys.path`, and call:

```python
from runtime.load import load
actor, world, metadata = load("g1-a_lafan_50k", "path/to/matching/robot.xml")
# actor(history, goals): history [B,10,layout.obs_dim], goals [B,11,layout.dim].
# Feature construction and state layout are defined in runtime/core.py.
```

The matching XMLs are in `assets/robots/{g1,h2,a3,sih2300}/robot.xml` (run `git lfs pull` for mesh dependencies). The loader verifies `training_contract.asset_sha256`.
Weights are not a drop-in replacement for the historical browser preview. Configuration
paths are portable basenames and must be mapped to your licensed local datasets/models.
This release provides inference code and recorded experiment configuration, not a
complete new training-engine port. No ONNX/browser/real-hardware validation is claimed.
Upstream dataset and robot licenses still apply; obtain motion data from their owners.
