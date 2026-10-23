"""Fix nominally fixed A3 head joints without changing DOF/body indexing."""
import argparse, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('asset',type=Path);a=p.parse_args()
    original=a.asset.read_bytes();root=ET.fromstring(original)
    eq=root.find('equality')
    if eq is None: eq=ET.SubElement(root,'equality')
    for name in ('head_yaw_joint','head_pitch_joint'):
        joint=root.find(f'.//worldbody//joint[@name="{name}"]')
        if joint is None: raise ValueError(name)
        tag='fixed_'+name
        if eq.find(f'joint[@name="{tag}"]') is None:
            ET.SubElement(eq,'joint',name=tag,joint1=name,polycoef='0 0 0 0 0',solref='0.005 1',solimp='0.9999 0.9999 0.001')
    backup=a.asset.with_suffix('.before_head_fix.xml')
    if not backup.exists(): backup.write_bytes(original)
    temp=a.asset.with_suffix('.tmp');ET.ElementTree(root).write(temp,encoding='unicode');temp.replace(a.asset)
    print(json.dumps({'old_sha256':hashlib.sha256(original).hexdigest(),'new_sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),'backup':str(backup)}))

if __name__=='__main__': main()
