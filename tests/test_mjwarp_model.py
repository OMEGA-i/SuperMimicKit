import hashlib

import numpy as np
import pytest

mj = pytest.importorskip('mujoco')
from supermimic.mjwarp import build_model


def test_training_actuator_force_limit_and_source_preservation(tmp_path, monkeypatch):
    source = tmp_path / 'robot.xml'
    source.write_text('''<mujoco><worldbody><geom type="plane" size="1 1 .1"/>
<body name="root" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
<body name="torso" pos="0 0 .2"><joint name="hinge" damping="4"/>
<geom type="sphere" size=".05" mass=".1"/></body></body></worldbody>
<actuator><motor joint="hinge"/></actuator></mujoco>''')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setenv('TEST_MJCF', str(source))
    cfg = {'asset_env':'TEST_MJCF', 'actuators':[{'pattern':'hinge', 'armature':.01,
           'stiffness':20, 'damping':1, 'effort':2, 'velocity':10}]}
    model, _ = build_model(cfg, ['hinge'])
    data = mj.MjData(model)
    data.ctrl[:] = 10
    mj.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, [2])
    assert model.dof_damping[-1] == 0
    assert np.count_nonzero(model.geom_type == mj.mjtGeom.mjGEOM_PLANE) == 1
    assert model.opt.integrator == mj.mjtIntegrator.mjINT_IMPLICITFAST
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    cfg['mujoco_solver'] = 'CG'
    cg, _ = build_model(cfg, ['hinge'])
    assert cg.opt.solver == mj.mjtSolver.mjSOL_CG
    cfg['mujoco_multiccd'] = False
    single, _ = build_model(cfg, ['hinge'])
    assert single.opt.disableflags & mj.mjtDisableBit.mjDSBL_MULTICCD
    assert not single.opt.disableflags & mj.mjtDisableBit.mjDSBL_CONTACT
