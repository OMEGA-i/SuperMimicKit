"""Export a trusted training snapshot for the browser, without touching CUDA.

Run from the project with PYTHONPATH=. .venv/bin/python scripts/export_web_demo.py
--checkpoint RUN/last.pt --motions MOTIONS --output WEB/assets/g1 --robot g1.
The training writer atomically replaces last.pt: mmap keeps this export on one inode.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from supermimic.core_v2 import Actor, Layout, WorldModel
from supermimic.kinematics import Kinematics
from supermimic.data import select_motion_paths
from supermimic.mjwarp import build_model
from supermimic.physx import actuator_gains


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_file(output, name, value, dtype='<f4'):
    value = np.asarray(value, dtype=dtype, order='C')
    if value.dtype.kind == 'f' and not np.isfinite(value).all():
        raise ValueError(f'Nonfinite exported tensor: {name}')
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    value.tofile(path)
    return {'file': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
            'bytes': path.stat().st_size, 'sha256': digest(path)}


def network(output, name, module):
    layers = []
    for i, layer in enumerate(module):
        if isinstance(layer, torch.nn.Linear):
            layers.append({'weight': array_file(output, f'weights/{name}_{i}_weight.bin', layer.weight.detach().numpy(), '<f2'),
                           'bias': array_file(output, f'weights/{name}_{i}_bias.bin', layer.bias.detach().numpy()),
                           'in': layer.in_features, 'out': layer.out_features, 'activation': 'linear'})
        elif isinstance(layer, torch.nn.LayerNorm):
            layers[-1]['layer_norm'] = {'weight': layer.weight.detach().tolist(), 'bias': layer.bias.detach().tolist(), 'eps': layer.eps}
        elif isinstance(layer, torch.nn.ELU): layers[-1]['activation'] = 'elu'
        elif isinstance(layer, torch.nn.SiLU): layers[-1]['activation'] = 'silu'
        else: raise TypeError(type(layer))
    return {'layers': layers, 'weight_order': 'out,in', 'elu_alpha': 1}


def inference_metadata(actor, world):
    def norm(n): return {'mean': n.mean.tolist(), 'std': n.std().tolist()}
    fk=world.kinematics
    return {'normalizers': {'obs': norm(actor.obs_norm), 'goal': norm(actor.goal_norm), 'world_in': norm(world.in_norm), 'world_out': norm(world.out_norm)},
            'kinematics': {'parents': fk.parents, 'joints': fk.joints, 'tracked': fk.tracked,
                           **{k:getattr(fk,k).tolist() for k in ('offset','rotation','axis','anchor','reference')}}}


def package_model(model, output):
    """Save the actual trained model, then relocate its original mesh/texture files."""
    path = output / 'model.xml'
    mj.mj_saveLastXML(str(path), model)
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find('compiler')
    directories = {kind: Path(compiler.get(kind + 'dir', '.')) if compiler is not None else Path('.')
                   for kind in ('mesh', 'texture')}
    # Visual triangles are already exported separately for rendering. Excluding
    # non-colliding geoms avoids exhausting WASM's 2 GiB heap on dense STL BVHs.
    world = root.find('worldbody')
    saved_geoms = list(world.iter('geom'))
    if len(saved_geoms) != model.ngeom:
        raise ValueError('Saved XML geometry ordering is ambiguous')
    retained = []
    for i, element in enumerate(saved_geoms):
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            for parent in world.iter():
                if element in list(parent):
                    parent.remove(element)
                    break
        else:
            retained.append(i)
    used_meshes = {g.get('mesh') for g in world.iter('geom') if g.get('mesh')}
    asset = root.find('asset')
    if asset is not None:
        for element in list(asset):
            if element.tag == 'mesh' and element.get('name') not in used_meshes:
                asset.remove(element)
    files = []
    for element in root.iter():
        if 'file' not in element.attrib:
            continue
        source = Path(element.get('file'))
        if not source.is_absolute():
            source = directories.get(element.tag, Path('.')) / source
        source = source.resolve(strict=True)
        # Content-derived names avoid collisions between robots' mesh directories.
        dest = Path('assets') / (digest(source)[:16] + source.suffix.lower())
        (output / dest).parent.mkdir(exist_ok=True)
        if not (output / dest).exists():
            shutil.copyfile(source, output / dest)
        element.set('file', dest.as_posix())
        if dest.as_posix() not in files:
            files.append(dest.as_posix())
    if compiler is not None:
        for attr in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(attr, None)
    tree.write(path, encoding='unicode')
    restored = mj.MjModel.from_xml_path(str(path))
    for key in ('jnt_qposadr', 'jnt_dofadr', 'actuator_gainprm', 'actuator_biasprm',
                'actuator_forcerange', 'dof_armature', 'dof_damping', 'body_mass', 'body_inertia', 'body_ipos'):
        np.testing.assert_allclose(getattr(restored, key), getattr(model, key), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(restored.geom_friction, model.geom_friction[retained], rtol=2e-5, atol=1e-7)
    if restored.opt.disableflags != model.opt.disableflags or restored.opt.solver != model.opt.solver:
        raise ValueError('Export changed physics options')
    return {'xml': 'model.xml', 'files': files, 'sha256': digest(path),
            'visual_geoms_separate': True, 'physics_geom_source_ids': retained}


def render_geometry(model, output):
    meshes = []
    for i in range(model.nmesh):
        va, vn = int(model.mesh_vertadr[i]), int(model.mesh_vertnum[i])
        fa, fn = int(model.mesh_faceadr[i]), int(model.mesh_facenum[i])
        meshes.append({'id': i,
                       'vertices': array_file(output, f'geometry/mesh_{i}_vertices.bin', model.mesh_vert[va:va+vn]),
                       'faces': array_file(output, f'geometry/mesh_{i}_faces.bin', model.mesh_face[fa:fa+fn], '<u4')})
    geoms = []
    for i in range(model.ngeom):
        rgba = model.geom_rgba[i].copy()
        if model.geom_matid[i] >= 0:
            rgba = model.mat_rgba[model.geom_matid[i]].copy()
        geoms.append({'id': i, 'body': int(model.geom_bodyid[i]), 'type': int(model.geom_type[i]),
                      'mesh': int(model.geom_dataid[i]), 'group': int(model.geom_group[i]),
                      'pos': model.geom_pos[i].tolist(), 'quat': model.geom_quat[i].tolist(),
                      'size': model.geom_size[i].tolist(), 'rgba': rgba.tolist()})
    return {'meshes': meshes, 'geoms': geoms, 'quaternion_order': 'wxyz',
            'vertices_space': 'compiled mesh local; apply geom pose then body pose'}


def export(args):
    torch.set_num_threads(2)
    # Explicit trust boundary: pickle is required by pathlib/NumPy metadata.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint.get('learner_version') != 2:
        raise ValueError('Legacy checkpoints are retired')
    if checkpoint.get('training_contract', {}).get('backend') != 'mujoco-warp':
        raise ValueError('Expected a MuJoCo Warp training checkpoint')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg = checkpoint['config']
    joints, bodies = checkpoint['joint_names'], checkpoint['body_names']
    layout = Layout(len(joints), len(bodies))
    model, params = build_model(cfg, joints)
    actor = Actor(layout, checkpoint['args']['small']).eval()
    actor.load_state_dict(checkpoint['actor'])
    world = WorldModel(layout, len(joints) + 4, Kinematics(model, joints, bodies),
                       width=checkpoint['world']['dynamics.0.weight'].shape[0]).eval()
    world.load_state_dict(checkpoint['world'])
    default = np.zeros(len(joints), dtype=np.float32)
    for i, name in enumerate(joints):
        for pattern, value in cfg['default_pose'].items():
            if re.fullmatch(pattern, name):
                default[i] = value
    joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in joints]
    body_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in bodies]
    clips = []
    all_paths = select_motion_paths(sorted(args.motions.rglob('motion.npz')), 'test37')
    for label in args.clips:
        matches = [p for p in all_paths if label.lower() in p.parent.name.lower()]
        if not matches:
            raise ValueError(f'No motion matching {label}')
        source = matches[0]
        with np.load(source, allow_pickle=False) as motion:
            if (motion['joint_names'].tolist(), motion['body_names'].tolist()) != (joints, bodies):
                raise ValueError(f'Motion schema mismatch: {source}')
            if float(motion['fps']) != 50:
                raise ValueError('Expected 50 Hz motion')
            states = motion['state'][:round(args.seconds * 50) + 10].astype(np.float32)
        if len(states) < 11 or states.shape[1] != layout.dim:
            raise ValueError('Invalid motion dimensions')
        clips.append({'id': source.parent.name, 'label': label, 'fps': 50,
                      'frames': len(states), 'playable_frames': len(states) - 10,
                      'state': array_file(output, f'motions/{source.parent.name}.bin', states)})
    from scripts.export_web_onnx import export_networks, rollout_fixture
    networks = export_networks(output, actor, world)
    metadata = {
        'format': 'supermimic-web-v2', 'learner_version': 2, **inference_metadata(actor, world), 'weight_storage': 'float32 ONNX', 'robot': args.robot, 'robot_name': cfg['robot'],
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'completed_iterations': int(checkpoint['iteration']) + 1,
        'checkpoint_stage': 'numerical-validation' if 'smoke' in str(checkpoint['args'].get('output','')) else 'full-training',
        'training_backend': 'mujoco-warp', 'inference_backend': 'onnxruntime-webgpu-f32',
        'mujoco_version': mj.__version__, 'source_asset_sha256': digest(Path(os.environ[cfg['asset_env']])),
        'model': package_model(model, output), 'render': render_geometry(model, output),
        'onnx': networks, 'motions': clips,
        'layout': {'joints': layout.joints, 'bodies': layout.bodies, 'dim': layout.dim, 'obs_dim': layout.obs_dim},
        'joint_names': joints, 'body_names': bodies, 'qpos_indices': model.jnt_qposadr[joint_ids].tolist(),
        'qvel_indices': model.jnt_dofadr[joint_ids].tolist(), 'tracked_body_ids': body_ids,
        'root_body_id': int(model.jnt_bodyid[0]), 'body_root_ids': model.body_rootid.tolist(),
        'default_pose': default.tolist(), 'action_scale': [.25 * p['effort'] / actuator_gains(p)[0] for p in params],
        'context': [1.] + [0.] * (len(joints) + 3), 'physics_dt': float(model.opt.timestep),
        'control_dt': .02, 'substeps': 4, 'history_length': 10, 'goal_frames': 11,
        'encoder_output_activation': 'tanh', 'world_output': 'local minimal-state integration + exact named FK',
        'quaternion_order': 'wxyz', 'nq': model.nq, 'nv': model.nv, 'nbody': model.nbody,
    }
    # A deterministic inference fixture checks JS tensor ordering and ELU/tanh.
    with torch.no_grad():
        state = torch.from_numpy(np.fromfile(output / clips[0]['state']['file'], dtype='<f4').reshape(-1, layout.dim).copy())
        previous = torch.zeros(1, layout.joints)
        history = layout.obs(state[:1], previous, torch.from_numpy(default), noise=False, goal=state[:1])[:, None].repeat(1, 10, 1)
        goals = state[:11][None]
        action = actor(history, goals)
        fixture = {'history': history.flatten().tolist(), 'goals': goals.flatten().tolist(),
                   'action': action.flatten().tolist(), 'state': state[0].tolist(),
                   'world_next': world(state[:1], action, torch.tensor(metadata['context'])[None]).flatten().tolist()}
    fixture['rollout'] = rollout_fixture(actor, world, fixture, metadata['context'], default.tolist())
    (output / 'inference_fixture.json').write_text(json.dumps(fixture))
    metadata['inference_fixture'] = 'inference_fixture.json'
    metadata['total_bytes'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    from scripts.export_web_motions import export as export_motions
    export_motions(args.motions, output, 37)
    print(json.dumps({'robot': args.robot, 'iterations': metadata['completed_iterations'],
                      'output': str(output), 'bytes': metadata['total_bytes'], 'clips': len(clips)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted local checkpoint only')
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--robot', required=True)
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--clips', nargs='+', default=['aiming', 'ground', 'push'])
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    export(args)


if __name__ == '__main__':
    main()
