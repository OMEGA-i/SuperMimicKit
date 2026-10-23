"""Validated motion/scene pairs for GRAIL conversion and playback."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pair(source):
    source = Path(source)
    pointer = json.loads((source.parent/'scene_pair.json').read_text())
    manifest_path = (source.parent/pointer['manifest']).resolve()
    if pointer['manifest_sha256'] != digest(manifest_path):
        raise ValueError('Scene pair manifest hash mismatch')
    pair = json.loads(manifest_path.read_text())
    if pair.get('schema_version') != 1 or pair.get('asset_version') != 4:
        raise ValueError('Unsupported scene pair version')
    if pair['motion_sha256'] != digest(source):
        raise ValueError('Scene pair belongs to a different motion')
    for name, expected in pair['files'].items():
        if digest(manifest_path.parent/name) != expected:
            raise ValueError(f'Paired file changed: {name}')
    for name, expected in pair['dependencies'].items():
        if digest(name) != expected:
            raise ValueError(f'Source scene changed since retarget: {name}')
    with np.load(source, allow_pickle=False) as motion:
        if (str(motion['robot_name'].item()) != pair['robot'] or
            str(motion['source_sequence_key'].item()) != pair['sequence'] or
            digest(str(motion['robot_xml'].item())) != pair['robot_xml_sha256']):
            raise ValueError('Scene pair robot/sequence mismatch')
        with np.load(manifest_path.parent/pair['object_motion'], allow_pickle=False) as d:
            qpos = d['qpos'].copy()
            if (qpos.shape != (len(motion['qpos']), 7) or
                not np.array_equal(d['frame_ids'], motion['frame_ids']) or
                float(d['fps'].item()) != float(motion['fps'].item()) or
                len(qpos) != pair['frames'] or float(d['fps'].item()) != pair['fps']):
                raise ValueError('Scene pair timeline mismatch')
    if not np.isfinite(qpos).all() or not np.allclose(np.linalg.norm(qpos[:,3:7],axis=1),1,atol=1e-5):
        raise ValueError('Invalid object poses')
    return pair, manifest_path.parent/pair['scene_xml'], qpos


def export_pair(source, output, frames, fps=50):
    from .retarget import resample
    pair, xml, poses = load_pair(source)
    poses = resample(poses, pair['fps'], target=fps)
    if len(poses) != frames:
        raise ValueError('Resampled scene/motion length mismatch')
    scene = Path(output)/'scene'
    scene.mkdir()
    shutil.copy2(xml, scene/'scene.xml')
    np.savez_compressed(scene/'object_motion.npz', qpos=poses, fps=fps)
    pair = {**pair, 'source_motion_sha256':pair['motion_sha256'],
            'motion_sha256':digest(Path(output)/'motion.npz'),
            'source_fps':pair['fps'], 'fps':fps, 'frames':frames,
            'files':{n:digest(scene/n) for n in ['scene.xml','object_motion.npz']}}
    (scene/'manifest.json').write_text(json.dumps(pair,indent=2)+'\n')
    return {'scene_required':True, 'scene':'scene/manifest.json',
            'scene_manifest_sha256':digest(scene/'manifest.json')}


def playback_model(robot_xml, scene_xml):
    """Merge exactly the paired scene into a robot model; no extra scaling."""
    import mujoco as mj
    root = ET.parse(robot_xml).getroot()
    compiler = root.find('compiler')
    for kind, directory in [('mesh','meshdir'),('texture','texturedir')]:
        base = Path(robot_xml).resolve().parent
        if compiler is not None:
            base = base/compiler.get(directory,'')
        for asset in root.findall('asset/'+kind):
            if asset.get('file'):
                asset.set('file', str((base/asset.get('file')).resolve()))
    if compiler is not None:
        for key in ['meshdir','texturedir','assetdir']:
            compiler.attrib.pop(key,None)
    scene = ET.parse(scene_xml).getroot()
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root,'asset')
    for a in scene.findall('asset/*'):
        asset.append(a)
    world = root.find('worldbody')
    for body in scene.findall('worldbody/body'):
        world.append(body)
    return mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
