"""Relocate copied absolute mesh paths for an isolated evaluation host."""
import argparse,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);a=p.parse_args()
    root=ET.parse(a.asset); files={}
    for e in root.getroot().iter():
        if 'file' not in e.attrib: continue
        source=Path(e.get('file'))
        if source.is_absolute():
            target=a.input_root/source.relative_to('/')
            if not target.is_file():raise FileNotFoundError(target)
            files[str(source)]=hashlib.sha256(target.read_bytes()).hexdigest()
            e.set('file',str(target))
    out=a.asset.with_name('evaluation.relocated.xml')
    root.write(out,encoding='unicode')
    out.with_suffix('.provenance.json').write_text(json.dumps({'source_xml_sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),'source':str(a.asset),'meshes':files},indent=2))
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(out))
    print(json.dumps({'output':str(out),'nq':m.nq,'nv':m.nv,'nbody':m.nbody}))

if __name__=='__main__':main()
