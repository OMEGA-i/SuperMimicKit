import numpy as np

from supermimic.lafan import load_bvh


def test_bvh_intrinsic_rotation_fk_and_basis(tmp_path):
    source = tmp_path / "motion.bvh"
    source.write_text("""HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Tip
 {
  OFFSET 100 0 0
  CHANNELS 3 Zrotation Yrotation Xrotation
  End Site
  {
   OFFSET 0 0 0
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.02
100 200 300 90 0 0 0 90 0
100 200 300 0 0 0 0 0 0
""")
    names, parents, pos, quat, fps = load_bvh(source)
    assert names == ["Hips", "Tip"]
    assert parents.tolist() == [-1, 0]
    assert fps == 50
    np.testing.assert_allclose(pos[:, 0], [[1, -3, 2], [1, -3, 2]], atol=1e-10)
    np.testing.assert_allclose(pos[:, 1], [[1, -3, 3], [2, -3, 2]], atol=1e-10)
    from scipy.spatial.transform import Rotation

    vector = Rotation.from_quat(quat[0, 1], scalar_first=True).apply([0, 0, 1])
    np.testing.assert_allclose(vector, [0, 0, 1], atol=1e-10)
