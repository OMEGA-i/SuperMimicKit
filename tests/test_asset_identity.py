import hashlib
import json
from pathlib import Path
import pytest
from supermimic.asset_identity import verify_asset


def test_relocated_asset_rejects_physics_and_mesh_changes(tmp_path):
    old=tmp_path/'old';old.mkdir();(old/'mesh.stl').write_bytes(b'mesh')
    xml=old/'model.xml';xml.write_text('<mujoco><compiler meshdir="."/><asset><mesh name="m" file="mesh.stl"/></asset><worldbody><body mass="1"/></worldbody></mujoco>')
    new=tmp_path/'new';new.mkdir();(new/'mesh.stl').write_bytes(b'mesh')
    target=new/'robot.xml';target.write_text(xml.read_text().replace(' meshdir="."',''))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof={'source_xml':str(xml),'source_xml_sha256':sha(xml),'xml_sha256':sha(target),'assets':[{'file':'mesh.stl','sha256':sha(new/'mesh.stl')}]}
    (new/'provenance.json').write_text(json.dumps(proof))
    verify_asset(target,sha(xml))
    (new/'mesh.stl').write_bytes(b'wrong')
    with pytest.raises(ValueError):verify_asset(target,sha(xml))
    (new/'mesh.stl').write_bytes(b'mesh')
    target.write_text(target.read_text().replace('mass="1"','mass="2"'))
    proof['xml_sha256']=sha(target);(new/'provenance.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError):verify_asset(target,sha(xml))
    with pytest.raises(ValueError):verify_asset(target,'bad')
