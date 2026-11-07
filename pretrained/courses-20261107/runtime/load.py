"""Strict CPU loading of the course inference bundle with a matching MJCF."""
import hashlib
import json
from pathlib import Path
import torch
from .core import Actor, Layout, WorldModel
from .kinematics import Kinematics

def load(bundle, xml):
    import mujoco
    bundle, xml = Path(bundle), Path(xml)
    m = json.loads((bundle/'metadata.json').read_text())
    if hashlib.sha256(xml.read_bytes()).hexdigest() != m['training_contract']['asset_sha256']:
        raise ValueError('Use the exact training XML; do not bypass the asset hash')
    if hashlib.sha256((bundle/'weights.pt').read_bytes()).hexdigest() != m['weights_sha256']:
        raise ValueError('Weight checksum mismatch')
    state = torch.load(bundle/'weights.pt', map_location='cpu', weights_only=True)
    layout = Layout(len(m['joint_names']), len(m['body_names']))
    actor = Actor(layout, m['small']).eval()
    actor.load_state_dict(state['actor'], strict=True)
    model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
    fk = Kinematics(model, m['joint_names'], m['body_names'])
    world = WorldModel(layout, layout.joints+4, fk, width=m['world_width']).eval()
    world.load_state_dict(state['world'], strict=True)
    return actor, world, m
