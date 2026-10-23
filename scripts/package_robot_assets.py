"""Create self-contained persistent robot packages and prove model equivalence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def package(name, source, root):
    dest=root/name;tmp=root/(name+'.partial')
    if dest.exists():raise FileExistsError(dest)
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tree=ET.parse(source);xml=tree.getroot()
    if xml.findall('include'):raise ValueError('Flatten includes before packaging')
    compiler=xml.find('compiler');compiler=compiler if compiler is not None else ET.SubElement(xml,'compiler')
    files=[]
    for kind,attr in [('mesh','meshdir'),('texture','texturedir')]:
        base=source.parent/compiler.get(attr,compiler.get('assetdir',''))
        compiler.attrib.pop(attr,None)
        for element in xml.findall('./asset/'+kind):
            if 'file' not in element.attrib:continue
            src=(base/element.get('file')).resolve(strict=True)
            digest=sha(src);rel=Path(kind+'es' if kind=='mesh' else 'textures')/(digest[:16]+src.suffix)
            (tmp/rel).parent.mkdir(exist_ok=True);shutil.copy2(src,tmp/rel);element.set('file',str(rel))
            files.append(dict(file=str(rel),source=str(src),sha256=digest))
    compiler.attrib.pop('assetdir',None)
    tree.write(tmp/'robot.xml',encoding='unicode')
    before=mujoco.MjModel.from_xml_path(str(source));after=mujoco.MjModel.from_xml_path(str(tmp/'robot.xml'))
    for attr in ['nq','nv','nbody','ngeom','nmesh','njnt','nu']:
        assert getattr(before,attr)==getattr(after,attr),attr
    checked=[]
    for attr in ['qpos0','body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat','jnt_type','jnt_axis','jnt_pos','jnt_range','dof_armature','geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','mesh_vert','mesh_face','eq_data','eq_obj1id','eq_obj2id','actuator_gear']:
        np.testing.assert_allclose(getattr(before,attr),getattr(after,attr),rtol=0,atol=1e-12,err_msg=attr);checked.append(attr)
    shutil.copy2(ROOT/'assets/robots'/name/'README.md',tmp/'README.md')
    extras=[]
    if name=='a3':extras=[root.parent/'a3_official'/f for f in ['LICENSE','NOTICE','manifest.json']]
    for f in extras:
        if f.exists():shutil.copy2(f,tmp/('source_'+f.name))
    (tmp/'provenance.json').write_text(json.dumps(dict(robot=name,source_xml=str(source),source_xml_sha256=sha(source),xml_sha256=sha(tmp/'robot.xml'),assets=files,validated_arrays=checked,nq=after.nq,nv=after.nv),indent=2))
    os.replace(tmp,dest)
    print(json.dumps(dict(robot=name,path=str(dest/'robot.xml'),nq=after.nq,meshes=after.nmesh)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',action='append',required=True,help='robot=/absolute/source.xml');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    for value in a.source:
        name,path=value.split('=',1);package(name,Path(path).resolve(),a.root)

if __name__=='__main__':main()
