# LaFAN preview — 2026-10-24

This is an **unofficial, experimental model preview**, not a uniform 100k release
or a paper-equivalent reproduction. Each package contains FP32 ONNX policy and
world model, normalization/FK metadata, a compatible robot model and its source
license. It contains no optimizer, replay buffer, raw training checkpoint or motion data.

| Robot | Policy iterations | World-model iterations | train40 return | Failed / 40 | val37 return | Failed / 37 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | 55,500 | **55,500** | 0.798961 | 3 | 0.310322 | 31 |
| H2 | 30,000 | 30,000 | 0.652555 | 6 | 0.259997 | 30 |
| A3 | 100,000 | 100,000 | Pending | Pending | Pending | Pending |

The return/failure rows measure the **policy in MuJoCo**, not world-model prediction
quality. G1 now uses the original paired 55.5k policy/world export. A3's exact 100k full-clip results have not been verified for this release.

## Why G1 and H2 are not 100k

Later training regressed in saved full-clip evaluations:

| Robot | Later policy | train40 return | Failed / 40 | val37 return | Failed / 37 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G1 | 100,000 | 0.008069 | 40 | 0.007507 | 37 |
| H2 | 45,000 | 0.158960 | 37 | 0.076222 | 37 |

We selected the earlier checkpoints above to expose usable development models.
We do not yet have a verified explanation of the later regression, and do not
attribute it to ONNX conversion. H2 is from the restarted run using the corrected
pelvis inertia; it also regressed later. H2's 100k quality is not asserted.
This selection is not an exhaustive best-checkpoint search.

## Evaluation protocol

Every clip starts at frame zero with noise and domain randomization disabled.
The policy's post-noise action is clamped to `[-20, 20]` before applying action
scale and default pose. A clip ends on its first training terminal condition;
remaining frames receive zero reward. Per-clip normalized return is
`sum(reward) / (0.1 * (frames - 1))`; the table is the arithmetic mean of clip
returns. Failure means an early terminal, not merely a low instantaneous reward.
Train40 and held-out val37 are separate sets. Our exact training list has not
been confirmed to match the paper's 40 clips. These figures are not live
single-step training reward. See `preview-evaluation.json` for the saved summary.

## Run locally

See [the browser guide](../web/demo/README.md). Unpack each robot archive under
`web/demo/dist/assets/`; each ZIP has a `g1/`, `h2/` or `a3/` top-level directory.
Install the web dependencies, then provide your own appropriately licensed,
schema-compatible 50 Hz motion references using `scripts.export_web_motions`.
Serve on localhost or trusted HTTPS in a browser supporting WebGPU.

**This is not a turnkey motion demo.** LaFAN's source license is
[CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt).
Retargeted motion files and motion-derived inference fixtures are therefore not
redistributed here. The included synthetic rest-pose fixture is a runtime smoke
test generated with ONNX Runtime CPU; it is not independent PyTorch validation.
Robot files retain their respective Unitree BSD-3-Clause and Agibot Apache-2.0
licenses/notices. See `ROBOT_SOURCE.md` inside each archive for modifications.

## Validation and limits

- ONNX checker and runtime loading for all six FP32 networks.
- Export-time PyTorch/ORT comparisons retained in each network's metadata.
- 32-step CPU/browser fixture checks, plus actual browser model loading and
  short world-model visualization on G1, H2 and A3.
- Browser actions use the training `[-20,20]` clamp. A3 loads the exported head
  constraints directly, without adding duplicate equality constraints.
- Browser physics is MuJoCo WASM; training/evaluation uses MuJoCo Warp. Short
  browser checks do not establish long-horizon or cross-backend equivalence.
- The world-model visualization periodically resynchronizes to MuJoCo. It is
  not an indefinitely open-loop prediction. No real-time performance claim.
- Generalization remains limited, as the val37 failure counts show.

BONES-SEEDs and GRAIL checkpoints remain planned; neither is included in this release.
