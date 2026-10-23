import json
from pathlib import Path
import numpy as np
import pytest
from supermimic.scene_pair import digest, load_pair, export_pair


def fixture(tmp_path):
    p=tmp_path;scene=p/'pair';scene.mkdir()
    robot=p/'robot.xml';robot.write_text('<mujoco/>')
    xml=scene/'scene.xml';xml.write_text('<mujoco/>')
    source=p/'retarget.npz'
    q=np.tile([0.,0,1,1,0,0,0],(31,1));q[:,0]=np.linspace(0,1,31)
    np.savez(source,qpos=q,fps=30,frame_ids=np.arange(31),robot_name='g1',robot_xml=str(robot),source_sequence_key='test')
    np.savez(scene/'object_motion.npz',qpos=q,fps=30,frame_ids=np.arange(31))
    m=dict(schema_version=1,asset_version=4,motion_sha256=digest(source),robot='g1',sequence='test',robot_xml_sha256=digest(robot),frames=31,fps=30,dependencies={str(robot):digest(robot)},files={n:digest(scene/n) for n in ['scene.xml','object_motion.npz']},object_motion='object_motion.npz',scene_xml='scene.xml')
    (scene/'manifest.json').write_text(json.dumps(m))
    (p/'scene_pair.json').write_text(json.dumps(dict(manifest='pair/manifest.json',manifest_sha256=digest(scene/'manifest.json'))))
    return source


def test_pair_resampling_same_clock_and_endpoints(tmp_path):
    source=fixture(tmp_path);out=tmp_path/'output';out.mkdir();(out/'motion.npz').write_bytes(b'motion')
    meta=export_pair(source,out,51)
    assert meta['scene_required']
    with np.load(out/'scene/object_motion.npz') as d:
        assert d['qpos'].shape==(51,7)
        np.testing.assert_allclose(d['qpos'][:,0],np.arange(51)/50)
        np.testing.assert_allclose(np.linalg.norm(d['qpos'][:,3:],axis=1),1)
    assert json.loads((out/'scene/manifest.json').read_text())['motion_sha256']==digest(out/'motion.npz')


def test_reject_changed_scene_or_motion(tmp_path):
    source=fixture(tmp_path)
    (tmp_path/'robot.xml').write_text('changed')
    with pytest.raises(ValueError,match='Source scene changed'):
        load_pair(source)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='different motion'):
        load_pair(source)

