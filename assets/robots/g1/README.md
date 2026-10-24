# Unitree G1

## Source

Official model family: https://github.com/unitreerobotics/unitree_ros
The vendor snapshot is the pinned `third_party/UMR` submodule (https://github.com/OMEGA-i/UMR, forked from https://github.com/hanyang9/UMR). Exact original Unitree revision is not recorded.

## Local adaptations

The existing UMR floating-root MJCF is retained.

## Storage

The canonical `robot.xml` and relative `meshes/` dependencies are included here.
Mesh and texture files use Git LFS; XML, provenance and license files use ordinary Git.
Run `git lfs install` and `git lfs pull` after cloning.

`provenance.json` records XML and mesh SHA-256 values and the original packaging
validation. Historical machine paths are omitted from this public manifest.
See `source_LICENSE` for the model license.
