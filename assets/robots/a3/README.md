# Agibot A3

## Source

Official repository: https://github.com/AgibotTech/sonic_for_a3
Revision: c9df753d31ef63c930b825b1b6767cabd37c1549
Original model: gear_sonic/data/assets/robot_description/mjcf/a3_t2d5.xml

## Local adaptations

Retains the floating root and our head constraint fix. Uses the rigid-foot A3 T2.5 asset, not the 035 passive-foot model. Original LICENSE and NOTICE are retained.

## Storage

Runtime package: `$SUPERMIMIC_ROBOT_ASSETS/a3/robot.xml`, with relative `meshes/` and `textures/` dependencies. Default root: `assets/robots`.

`provenance.json` records source XML, input/output hashes, copied asset hashes and numerical model-equivalence validation. Packaging only relocates paths; it does not change physical parameters. Bulk assets are stored on persistent storage, not duplicated into Git. This README is copied into each runtime package.
