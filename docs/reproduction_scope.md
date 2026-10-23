# Reproduction scope

SuperMimicKit is an independent implementation for G1, H2 and A3. The primary
backend is PyTorch plus MuJoCo Warp; an Isaac Sim / PhysX adapter is also included.
The implementation includes learned dynamics, policy/critic updates, motion
sampling, observation noise and domain randomization, checkpoint export and
per-motion evaluation. It is not an official SuperMimic release.

## Metrics

The evaluation denominator is the full reference duration:
`normalized_return = sum(reward) / (0.1 * (frames - 1))`.
A first terminal event stops that motion; remaining reference frames receive
zero credit. Each motion contributes equally to the split mean. The failed
motion count is the number of clips with an early terminal event.

The train40 and held-out val37 split means must be reported separately. A live
training reward is a per-step average under the training sampler and context;
it is not a full-clip normalized return. Evaluating held-out motions is also
not interchangeable with the paper's training-set evaluation. The exact
identity of the original paper's 40 clips has not been independently confirmed.

When comparing runs, record the model/asset hashes, checkpoint iteration,
precision, simulator version, split, termination rule, initial state and noise
settings. An FP16 browser export is not an exact FP32 checkpoint. Do not label
results as equivalent without measuring the difference.

## Current limits

Policy-only ONNX weights are distributed separately through GitHub Releases.
Source motion datasets, deployment logs and empirical performance claims are
not included. Unit tests
verify targeted contracts; they do not demonstrate locomotion stability.
