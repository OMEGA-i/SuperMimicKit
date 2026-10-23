"""Publish the exact GRAIL scene/trajectory used by a retarget result."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_pair(result_path, scene_dir, config):
    # Use the solver's own transforms, not a second approximation of them.
    from retarget_smpl_to_humanoid_surface_vector import (
        read_prop_motion, make_mjcf_mesh_paths_absolute, scale_mjcf_mesh_assets,
    )
    result_path, scene_dir = Path(result_path), Path(scene_dir)
    meta_path = scene_dir / 'metadata.json'
    scene_meta = json.loads(meta_path.read_text())
    if scene_meta.get('asset_version') != 4:
        raise ValueError('GRAIL pairing requires validated version4 source geometry')
    with np.load(result_path, allow_pickle=True) as d:
        frames = np.asarray(d['frame_ids'], dtype=np.int32)
        scale = float(d['smpl_scale'].item())
        fps = float(d['fps'].item())
        robot = str(d['robot_name'].item())
        sequence = str(d['source_sequence_key'].item())
        robot_xml = Path(str(d['robot_xml'].item()))
        ground = float(d['ground_z'].item())
        if len(frames) != len(d['qpos']) or not np.isfinite(d['qpos']).all():
            raise ValueError('Invalid paired motion')
    if sequence != scene_meta['sequence_key'] or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scene identity/scale mismatch')
    object_cfg = config.get('hsi_hoi', {}).get('object', {})
    mode = config.get('solver', {}).get('retarget_object_size', 'scaled')
    if mode not in ('scaled', 'original'):
        raise ValueError(f'Unsupported object size mode: {mode}')
    mesh_scale = scale if mode == 'scaled' else 1.0
    csv_path = scene_dir / 'prop_grail_object.csv'
    # Reject missing frames: the solver helper otherwise clips indices silently.
    row_count = len(np.loadtxt(csv_path, delimiter=',', skiprows=1, ndmin=2))
    if len(frames) < 2 or frames.min() < 0 or frames.max() >= row_count or not np.all(np.diff(frames) > 0):
        raise ValueError('Object and motion frame ranges do not match')
    transform = dict(output_up=object_cfg.get('output_up', 'z'),
                     convert_y_up=bool(object_cfg.get('convert_y_up', False)),
                     ground_align=bool(object_cfg.get('ground_align', False)),
                     floor_y=float(object_cfg.get('floor_y', 0)),
                     ground_offset=float(object_cfg.get('ground_offset', 0)))
    poses = read_prop_motion(csv_path, frames, smpl_scale=scale, **transform)
    object_qpos = np.c_[poses['positions'], poses['quats_wxyz']]
    if not np.isfinite(object_qpos).all():
        raise ValueError('Nonfinite object trajectory')
    xml_path = scene_dir / 'grail_object.xml'
    tree = ET.parse(xml_path)
    make_mjcf_mesh_paths_absolute(tree.getroot(), xml_path)
    scale_mjcf_mesh_assets(tree.getroot(), mesh_scale)
    # Source geometry stays shared; hash-pin every dependency to prevent silent reuse
    # of a later scene rebuild with an older motion.
    dependencies = {str(p.resolve()): digest(p) for p in (xml_path, csv_path, meta_path)}
    for mesh in tree.findall('asset/mesh'):
        p = Path(mesh.get('file')); dependencies[str(p)] = digest(p)
    pair_root = result_path.parent / 'scene_pair'
    pair_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pair-', dir=pair_root) as temp:
        stage = Path(temp)
        tree.write(stage / 'scene.xml', encoding='unicode')
        np.savez_compressed(stage / 'object_motion.npz', qpos=object_qpos,
                            frame_ids=frames, fps=np.asarray(fps))
        manifest = dict(schema_version=1, asset_version=4, robot=robot, sequence=sequence,
                        frames=len(frames), fps=fps, quaternion='wxyz', units='m', up_axis='z',
                        smpl_scale=scale, mesh_scale=mesh_scale, retarget_object_size=mode,
                        source_ground_z=ground, object_transform=transform,
                        motion_sha256=digest(result_path), robot_xml_sha256=digest(robot_xml),
                        scene_xml='scene.xml', object_motion='object_motion.npz',
                        dependencies=dependencies,
                        files={name:digest(stage/name) for name in ['scene.xml','object_motion.npz']})
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        version = digest(stage/'manifest.json')[:24]
        target = pair_root/version
        if not target.exists():
            os.rename(stage, target)
        pointer = result_path.parent/'scene_pair.json'
        # A tiny pointer is the atomic commit record. Readers never see a half pair.
        with tempfile.NamedTemporaryFile(mode='w', dir=pointer.parent, delete=False) as f:
            json.dump({'schema_version':1,'manifest':str(Path('scene_pair')/version/'manifest.json'),
                       'manifest_sha256':digest(target/'manifest.json')}, f)
        os.replace(f.name, pointer)
    return pointer
