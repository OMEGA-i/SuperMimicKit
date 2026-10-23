# Unitree H2

## Source

Official model family: https://github.com/unitreerobotics/unitree_ros
Vendored through the pinned UMR submodule.

## Local adaptations

Retains our corrected 8.5477 kg pelvis inertia. The original upstream revision is not asserted here; provenance.json records exact input SHA-256. Floating-root preparation remains governed by the H2 UMR XML policy.

## Storage

Runtime package: `$SUPERMIMIC_ROBOT_ASSETS/h2/robot.xml`, with relative `meshes/` and `textures/` dependencies. Default root: `assets/robots`.

`provenance.json` records source XML, input/output hashes, copied asset hashes and numerical model-equivalence validation. Packaging only relocates paths; it does not change physical parameters. Bulk assets are stored on persistent storage, not duplicated into Git. This README is copied into each runtime package.
