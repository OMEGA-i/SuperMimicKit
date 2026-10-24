# Unitree H2

## Source

Official model family: https://github.com/unitreerobotics/unitree_ros
Vendored through the pinned UMR submodule.

## Local adaptations

Retains our corrected 8.5477 kg pelvis inertia. The original upstream revision is not asserted here; provenance.json records exact input SHA-256. Floating-root preparation remains governed by the H2 UMR XML policy.

## Storage

The canonical `robot.xml` and relative `meshes/` dependencies are included here.
Mesh and texture files use Git LFS; XML, provenance and license files use ordinary Git.
Run `git lfs install` and `git lfs pull` after cloning.

`provenance.json` records XML and mesh SHA-256 values and the original packaging
validation. Historical machine paths are omitted from this public manifest.
See `source_LICENSE` for the model license.
