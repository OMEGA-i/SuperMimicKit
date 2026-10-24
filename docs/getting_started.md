# Getting started

[English overview](../README.md) · [中文概览](../README.zh-CN.md)

## Installation

Linux with an NVIDIA GPU is the training target. Create a separate environment
for training, UMR and SOMA Retargeter; their dependency versions differ.
Install Git LFS and uv before the commands below.

```bash
git lfs install
git clone --recurse-submodules https://github.com/OMEGA-i/SuperMimicKit.git
cd SuperMimicKit
git lfs pull
git -C third_party/soma-retargeter lfs pull
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[mjwarp,qc,test,tracking,web-refresh]'
```

For an existing checkout, run `git submodule update --init --recursive`, then
the two LFS pull commands above.

For the LaFAN / GRAIL UMR route, run `bash scripts/bootstrap_umr.sh` once to
create `.venv-umr`; it also applies the integration overlay. See
[the overlay notes](../third_party/umr_overlay/README.md). For SOMA BVH, follow
[SOMA Retargeter installation](../third_party/soma-retargeter/documentation/installation.md)
in its own environment. Adding the submodule does not install its dependencies.
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

All three packages are included: XML and provenance use ordinary Git, while
meshes use Git LFS. Each package retains source documentation and its model
license; A3 also includes its upstream notice. Run `git lfs pull` if mesh files
are still small text pointers.

`package_robot_assets.py` is for packaging additional prepared floating-base
MJCFs; it is not required to download these models. Do not substitute a model
with different joint order or inertia when resuming a checkpoint. SOMA
Retargeter has its own bundled robot variants; those are not automatically
interchangeable with the canonical training models here.

Set `SUPERMIMIC_G1_MJCF`, `SUPERMIMIC_H2_MJCF` or `SUPERMIMIC_A3_MJCF`
to the corresponding absolute `robot.xml` path. Launcher scripts derive these
from `SUPERMIMIC_ROBOT_ASSETS`, which defaults to `$PWD/assets/robots`.
See the per-robot README files and [control sources](control_sources.json).

## Data and retargeting

Full source datasets and separately licensed body-model weights are not
distributed by this toolkit. Submodules may include separately licensed sample
data; consult their notices. Obtain the required data under its own terms.

```text
datasets/
├── lafan/{source,smplx,umr,motions,splits,manifest.json}
├── grail/{source,smplx,umr,motions,scenes,manifest.json}
└── bones-seed/{source,soma-retargeter,motions,manifest.json}
```

LaFAN follows source → SMPL-X → UMR → named 50 Hz motion arrays. BONES-SEED
uses the planned SOMA BVH → 50 Hz resampling → SOMA Retargeter → training-format
motions route. The upstream submodule outputs CSV, not SuperMimicKit
`motion.npz`; the public end-to-end adapter is not yet included. Do not feed
its CSV directly to the trainer. Older UMR BONES adapters remain in the overlay
as legacy utilities. GRAIL motion and scene outputs must remain paired.

LaFAN / UMR entrypoints:

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
  --checkpoint outputs/g1_lafan/ckpt_30000.pt \
  --motions datasets/lafan/motions/g1 \
  --manifest datasets/lafan/manifest.json --motion-split train40 \
  --device cuda:0 --output outputs/g1_lafan/eval_train40

.venv/bin/python -m scripts.evaluate_full_motions \
  --checkpoint outputs/g1_lafan/ckpt_30000.pt \
  --motions datasets/lafan/motions/g1 \
  --manifest datasets/lafan/manifest.json --motion-split test37 \
  --device cuda:0 --output outputs/g1_lafan/eval_val37
```

Replace `ckpt_30000.pt` with an actual saved snapshot from your run, and use
that same snapshot for both evaluations. The CLI name `test37`
is the split reported as `val37` in tables and dashboards. Each output directory
must be new; `results.json` contains the per-clip results and protocol.

Evaluation starts each clip at frame zero, disables noise/randomization, and
ends on the first terminal event. Per-clip normalized return is
`sum(reward) / (0.1 * (frames - 1))`; frames after failure contribute zero.
Report arithmetic means separately for train40 and val37. A failed motion is
one that triggers early termination. Training's per-step `sim_reward` is a
different statistic. See [metric limitations](reproduction_scope.md).

`scripts/swanlab_tracking.py` supports independent train/eval workers per robot.
Set `project`, `workspace`, `experiment_name` and `experiment_id` in your local
configuration, with credentials supplied locally. No account credentials are
included. The checkpoint capture helper can retain evaluation weights every
500 updates and paired policy + world-model archives (`ckpt_<iteration>.pt`) every 10k updates.
For an already-running trainer, run `scripts.capture_eval_checkpoint` with
`--every 10000 --include-world`; it uses CPU mmap and does not require a training restart.

## Checkpoint exports

Complete demo bundles require policy and world-model weights from the same
checkpoint, plus compatible assets, metadata and motions. Policy-only exports
cannot replace both networks. The [published preview](preview-release.md) uses
paired G1 55.5k, H2 30k and A3 100k networks. It includes robot assets but no
motion references. Later iterations are not automatically better; see the
release evaluation and limitations.

Use `scripts/capture_demo_checkpoint.py` to preserve both networks from a full
training checkpoint using CPU only, without copying replay or optimizer state.
`scripts/export_policy_release.py` remains available when a policy-only artifact
is explicitly wanted.

## Browser demo

[Demo setup](../web/demo/README.md). Policy and world-model networks use ONNX Runtime
Web on WebGPU; MuJoCo physics uses WASM. WebGPU needs a secure browser context
(HTTPS or localhost). Export your own compatible checkpoint and motions; the
repository does not include a deployed public demo. The preview can be run
locally after supplying compatible, licensed motion references.

