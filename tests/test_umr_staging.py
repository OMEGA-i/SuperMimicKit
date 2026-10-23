import numpy as np
import pytest

mj = pytest.importorskip("mujoco")
from scripts.run_umr import stage_model


def test_external_mesh_assets_survive_staging(tmp_path):
    assets = tmp_path / "supplier" / "assets"
    assets.mkdir(parents=True)
    mesh = assets / "tetra.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    xml = assets.parent / "robot.xml"
    xml.write_text(
        '<mujoco><compiler meshdir="assets"/><asset><mesh name="shape" file="tetra.obj"/></asset><worldbody><body><freejoint/><geom type="mesh" mesh="shape"/></body></worldbody></mujoco>'
    )
    original = xml.read_bytes()
    staged = stage_model(xml, tmp_path / "instance")
    source_model = mj.MjModel.from_xml_path(str(xml))
    staged_model = mj.MjModel.from_xml_path(str(staged))
    np.testing.assert_allclose(source_model.body_mass, staged_model.body_mass)
    np.testing.assert_allclose(source_model.mesh_vert, staged_model.mesh_vert)
    assert "/assets/" not in staged.read_text()
    assert xml.read_bytes() == original
    assert (staged.parent / "meshes/0.obj").is_symlink()
