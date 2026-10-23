from pathlib import Path
import xml.etree.ElementTree as ET

from supermimic.physx import stage_physx_asset


def test_scene_plane_removed_without_changing_robot_or_source(tmp_path):
    source = tmp_path / "robot.xml"
    source.write_text('<mujoco><compiler meshdir="meshes"/><worldbody><geom name="ground" type="plane" size="10 10 .1"/><body name="robot"><freejoint/><geom name="foot" type="box" size=".1 .1 .1"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>')
    before = source.read_bytes()
    staged = stage_physx_asset(source, tmp_path / "simulation.xml")
    original, result = ET.fromstring(before), ET.parse(staged).getroot()
    assert source.read_bytes() == before
    assert result.findall('./worldbody/geom[@type="plane"]') == []
    assert ET.tostring(original.find('./worldbody/body')) == ET.tostring(result.find('./worldbody/body'))
    assert Path(result.find('compiler').get('meshdir')) == tmp_path / 'meshes'
