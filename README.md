# SuperMimicKit

**An unofficial SuperMimic reproduction for Unitree G1, Unitree H2 and Agibot A3.**

English · [简体中文](README.zh-CN.md)

PyTorch + MuJoCo Warp training, UMR retargeting, and a browser demo powered by
ONNX Runtime Web / WebGPU. An independent implementation inspired by
[SuperMimic](https://supermimic.github.io/), with reference to
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed).

## Status

- [x] G1 / H2 / A3 configurations and a unified robot asset layout.
- [x] LaFAN retargeting and explicit train40 / held-out val37 splits.
- [x] FP32 training, optional TF32 / BF16, compilation and CUDA graphs.
- [x] Independent per-robot SwanLab tracking and full-clip evaluation.
- [x] ONNX policy / world-model export and WebGPU demo integration; MuJoCo physics runs in WASM.
- [x] Paired policy + world-model weight archives every 10k iterations, including normalizers.
- [x] Publish a LaFAN ONNX preview: G1 55.5k policy / 55.5k world, H2 30k, A3 100k.
- [x] Report saved G1/H2 train40 / val37 results; exact A3 100k evaluation remains pending.
- [ ] **BONES-SEEDs checkpoints** — retargeting pipeline preparation is underway; training and release to follow.
- [ ] **GRAIL checkpoints** — motion/scene processing is underway; scene-aware training and release to follow.

## Performance

Measured development results; these are not a claim of paper-equivalent convergence.

| Measurement | Result | Scope |
| --- | --- | --- |
| G1 training throughput | **2.18 iterations/s** · 0.459 s/iteration | H100, 4,096 environments, TF32, compile + CUDA graphs |
| GPU utilization | **84.7%** | Same short benchmark; 33 GPU samples |
| G1 policy ONNX max absolute error | **2.63 × 10⁻⁶** | Exact 50k policy, FP32 PyTorch vs ONNX Runtime CPU |
| A3 policy ONNX max absolute error | **4.06 × 10⁻⁶** | Exact 50k policy, FP32 PyTorch vs ONNX Runtime CPU |

Throughput averages 36 measured iterations and excludes initial compilation.
It is a short-run measurement, not end-to-end training time or browser speed.
ONNX checks use 16 synthetic input fixtures per robot; they establish numerical
parity on those inputs, not locomotion quality or full world-model parity.
See [measurement details](docs/performance.md).

## Pretrained preview

[Download the preview](https://github.com/OMEGA-i/SuperMimicKit/releases/tag/demo-preview-20261024)
· [Selection, evaluation and limitations](docs/preview-release.md)

| Robot | Policy / world iterations | train40 return · failures | val37 return · failures |
| --- | --- | --- | --- |
| G1 | **55.5k / 55.5k** | 0.7990 · 3/40 | 0.3103 · 31/37 |
| H2 | **30k / 30k** | 0.6526 · 6/40 | 0.2600 · 30/37 |
| A3 | **100k / 100k** | Pending | Pending |

G1 and H2 use earlier policies because later full-clip evaluation regressed:
G1 at 100k failed 40/40 training clips; H2 at 45k failed 37/40. The cause remains
under investigation. G1 now uses the original paired 55.5k export for both networks.
Returns are equally weighted full-clip `sum(reward)/(0.1*(frames-1))`, with zero
credit after termination. They are not single-step training reward or world-model
accuracy. Our exact train40 list has not been confirmed against the paper.

Bundles contain FP32 ONNX networks, metadata and robot assets. **Motion data is
not included** due to source redistribution restrictions; supply licensed motions
before running the browser demo. These are experimental previews, not deployment-ready controllers.

## Get started

```bash
git clone --recurse-submodules https://github.com/OMEGA-i/SuperMimicKit.git
cd SuperMimicKit
git lfs install
git lfs pull
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[mjwarp,qc,test,tracking,web-refresh]'
bash scripts/bootstrap_umr.sh
```

G1, H2 and A3 robot assets are included in `assets/robots`; meshes use Git LFS.
Prepare licensed retargeted motions before training.
See the [setup, data and training guide](docs/getting_started.md),
[UMR integration notes](third_party/umr_overlay/README.md) and
[browser demo guide](web/demo/README.md).

## Checkpoints and upcoming datasets

The first [public preview](https://github.com/OMEGA-i/SuperMimicKit/releases/tag/demo-preview-20261024)
uses the checkpoints listed above. Future releases will follow validated quality
rather than assuming a later iteration is better.

| Dataset | Processing route | Planned release |
| --- | --- | --- |
| LaFAN | Source → SMPL-X → UMR → 50 Hz motions | G1 55.5k/55.5k, H2 30k, A3 100k preview |
| BONES-SEEDs | 120 Hz BVH → in-memory SOMA / 50 Hz resampling → UMR → motions | G1 / H2 / A3 checkpoints; schedule TBD |
| GRAIL | Source → SMPL-X + scenes → UMR → paired motions and scenes | G1 / H2 / A3 checkpoints; schedule TBD |

BONES-SEEDs does not require a second SOMA dataset copy. GRAIL scenes remain
paired with their retargeted motions. Dataset/model redistribution depends on
their respective licenses.

## Acknowledgments

Thanks to the [SuperMimic authors](https://supermimic.github.io/),
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed),
[UMR](https://github.com/hanyang9/UMR), and the robot and dataset contributors.
Our UMR fork is [OMEGA-i/UMR](https://github.com/OMEGA-i/UMR).
Third-party code, models and datasets retain their own licenses and notices.
