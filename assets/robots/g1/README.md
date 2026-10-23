# Unitree G1

## Source

Official model family: https://github.com/unitreerobotics/unitree_ros
The vendor snapshot is the pinned `third_party/UMR` submodule (https://github.com/OMEGA-i/UMR, forked from https://github.com/hanyang9/UMR). Exact original Unitree revision is not recorded.

## Local adaptations

The existing UMR floating-root MJCF is retained.

## Storage

Runtime package: `$SUPERMIMIC_ROBOT_ASSETS/g1/robot.xml`, with relative `meshes/` and `textures/` dependencies. Default root: `assets/robots`.

`provenance.json` records source XML, input/output hashes, copied asset hashes and numerical model-equivalence validation. Packaging only relocates paths; it does not change physical parameters. Bulk assets are stored on persistent storage, not duplicated into Git. This README is copied into each runtime package.
