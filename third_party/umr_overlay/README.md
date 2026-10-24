# Public UMR integration overlay

Base dependency: [OMEGA-i/UMR](https://github.com/OMEGA-i/UMR), pinned to
`d1cdaedaf67a827198da71c5fa831f48b2e51f91`, from [hanyang9/UMR](https://github.com/hanyang9/UMR).
The public submodule contains upstream assets and licenses. Run
`python scripts/apply_umr_overlay.py` after initializing the submodule. This
copies the integration files into that checkout; the resulting submodule is
intentionally modified. Do not use a submodule reset to discard the overlay.

The overlay retains A3 configuration, corrected H2 inertial parameters,
MJCF path handling, atomic generated files, bounded QP recovery, scene/motion
pairing and in-memory SOMA resampling. Successful solver paths stay unchanged;
recovery still checks finite values and original constraint residuals.

Distributed adapters use `UMR_CLUSTER_NODES` (a JSON object mapping node names
to positive integer shard weights), `UMR_LOCK_DIR` and
`SUPERMIMIC_ROBOT_ASSETS`. Defaults are one node named `local`, `/tmp`, and the
project's `assets/robots`. Keep node assignments fixed while resuming a dataset.
GRAIL's `--preprocessor` is an explicit external resampler dependency; it must
synchronize human and object trajectories at 30 Hz. The legacy BONES adapter
reads BVH and resamples to 50 Hz in memory. The current BONES-SEED direction
uses the separate `third_party/soma-retargeter` upstream submodule; this overlay
is not its integration layer. See the [data guide](../../docs/getting_started.md).

UMR-derived code is under the included upstream MIT license. Robot asset and
source dataset terms remain separate. No source motion datasets, licensed body
models, checkpoints, or generated robot packages are included in this overlay.
