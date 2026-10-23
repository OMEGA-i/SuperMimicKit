# SuperMimicKit

**An unofficial reproduction and multi-embodiment toolkit for SuperMimic.**

PyTorch training, MuJoCo Warp simulation, UMR motion retargeting, independent
train/validation metrics, and a browser demo using ONNX Runtime Web / WebGPU.
The public configurations support **Unitree G1, Unitree H2 and Agibot A3**.
This is an independent implementation, not the authors' official release.

- [Original SuperMimic project](https://supermimic.github.io/)
- [Reference reproduction: supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
- [UMR dependency](https://github.com/OMEGA-i/UMR)
- [UMR original project](https://github.com/hanyang9/UMR)

## Installation

Linux with an NVIDIA GPU is the training target. Create a separate environment
for training and UMR; their dependency versions differ.

```bash
git clone --recurse-submodules https://github.com/OMEGA-i/SuperMimicKit.git
cd SuperMimicKit
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[mjwarp,qc,test,tracking,web-refresh]'
python3 scripts/apply_umr_overlay.py
bash scripts/bootstrap_umr.sh
```

The UMR submodule is a public pinned snapshot. The checked-in integration overlay
adds A3 support and fixes used by this toolkit; see
[its provenance and setup notes](third_party/umr_overlay/README.md).
An Isaac Sim / PhysX backend is also present, but needs a separately installed
Isaac Lab environment. MuJoCo Warp is the primary training path.

## Robot assets

Use one self-contained directory per embodiment:

```text
assets/robots/
├── g1/{robot.xml,meshes,README.md,provenance.json}
├── h2/{robot.xml,meshes,README.md,provenance.json}
└── a3/{robot.xml,meshes,README.md,provenance.json}
```

Only source documentation is included here. Obtain model files from their
respective upstream sources and honor their licenses. G1/H2 assets are provided
by UMR; A3 can be fetched with `python scripts/fetch_a3_assets.py --help`.
`package_robot_assets.py` packages an already prepared floating-base MJCF,
rewrites mesh paths and verifies model equivalence. Do not substitute a model
with different joint order or inertia when resuming a checkpoint.

Set `SUPERMIMIC_G1_MJCF`, `SUPERMIMIC_H2_MJCF` or `SUPERMIMIC_A3_MJCF`
to the corresponding absolute `robot.xml` path. Launcher scripts derive these
from `SUPERMIMIC_ROBOT_ASSETS`, which defaults to `$PWD/assets/robots`.
See the per-robot README files and [control sources](docs/control_sources.json).

## Data and retargeting

Datasets and licensed SMPL-X / SOMA assets are not distributed in this repository.
Use the relevant dataset's access and license requirements.

```text
datasets/
├── lafan/{source,smplx,umr,motions,splits,manifest.json}
├── grail/{source,smplx,umr,motions,scenes,manifest.json}
└── bones-seed/{source,umr,motions,manifest.json}
```

LaFAN follows source → SMPL-X → UMR → named 50 Hz motion arrays. BONES-SEED
uses BVH → in-memory SOMA parsing / 50 Hz resampling → UMR → motions.
GRAIL motion and scene outputs must remain paired. The data adapters and their
limitations are described in the UMR overlay README.

Useful entrypoints:

```bash
.venv/bin/python scripts/fetch_lafan.py --help
.venv-umr/bin/python scripts/lafan_to_smplx.py --help
.venv-umr/bin/python scripts/run_umr.py --help
.venv/bin/python -m supermimic.retarget --help
```

The LaFAN split is explicit in `configs/lafan40.json`: 40 training clips and
37 held-out clips. Preserve these names in `datasets/lafan/manifest.json`.
This split is a reproduction convention; it has not been proven identical to
the original authors' exact training clip list.

## Training and evaluation

Run from the project root. For example, with G1 assets and motions prepared:

```bash
export SUPERMIMIC_G1_MJCF="$PWD/assets/robots/g1/robot.xml"
.venv/bin/python -m supermimic.train \
  --config configs/g1.json --motions datasets/lafan/motions/g1 \
  --split-manifest datasets/lafan/manifest.json --motion-split train40 \
  --backend mujoco-warp --device cuda:0 --num-envs 4096 \
  --precision tf32 --compile --cuda-graphs \
  --iterations 320000 --checkpoint-every 500 --output outputs/g1_lafan
```

`fp32` is the default; `tf32` and optional `bf16` are explicit settings. Use one
training process per embodiment / GPU. Resume a trusted full checkpoint with
`--resume outputs/g1_lafan/last.pt`, retaining the same training contract.
Do not equate an iteration count or short numerical test with convergence.

```bash
.venv/bin/python -m scripts.evaluate_full_motions \
  --checkpoint outputs/g1_lafan/last.pt \
  --motions datasets/lafan/motions/g1 \
  --manifest datasets/lafan/manifest.json --motion-split all77 \
  --device cuda:0 --output outputs/g1_lafan/evaluation
```

Evaluation starts each clip at frame zero, disables noise/randomization, and
ends on the first terminal event. Per-clip normalized return is
`sum(reward) / (0.1 * (frames - 1))`; frames after failure contribute zero.
Report arithmetic means separately for train40 and val37. A failed motion is
one that triggers early termination. Training's per-step `sim_reward` is a
different statistic. See [metric limitations](docs/reproduction_scope.md).

`scripts/swanlab_tracking.py` supports independent train/eval workers per robot.
Set `project`, `workspace`, `experiment_name` and `experiment_id` in your local
configuration, with credentials supplied locally. No account credentials are
included. The checkpoint capture helper can retain evaluation weights every
500 updates and archival weights every 10k updates.

## Checkpoint exports

Complete demo bundles require policy and world-model weights from the same
checkpoint, plus compatible assets, metadata and motions. Policy-only exports
cannot replace a full demo bundle. Release artifacts are being prepared; no
complete pretrained demo bundle is currently published.

Use `scripts/capture_demo_checkpoint.py` to preserve both networks from a full
training checkpoint using CPU only, without copying replay or optimizer state.
`scripts/export_policy_release.py` remains available when a policy-only artifact
is explicitly wanted.

## Browser demo

[Demo setup](web/demo/README.md). Policy and world-model networks use ONNX Runtime
Web on WebGPU; MuJoCo physics uses WASM. WebGPU needs a secure browser context
(HTTPS or localhost). Export your own compatible checkpoint and motions; the
repository does not include a deployed public demo; a complete demo requires a matching world model as well as the policy.

## Validation and provenance

```bash
npm ci --prefix web/demo  # ONNX export / Node parity test dependencies
.venv/bin/python -m pytest tests
```

Tests cover algorithm/data contracts and selected numerical behavior. CUDA
training, Isaac Sim and browser rendering need their respective runtimes and
are not established by CPU unit tests. Historical private deployment reports
are not published, and this repository claims no paper-equivalent
training score or three-hour convergence result.

Third-party code and assets retain their original licenses and notices,
including UMR's MIT license and the separately licensed robot assets. Publishing
this repository does not relicense those dependencies or datasets.
