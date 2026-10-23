"""Deterministic full-clip evaluation with training-compatible MuJoCo Warp."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import BadZipFile

import numpy as np
import torch

from supermimic.core_v2 import Actor, Layout, VERSION
from supermimic.data import Motions, select_motion_paths
from supermimic.mjwarp import MJWarp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--motions', type=Path, required=True)
    p.add_argument('--motion-split', choices=('train40','test37','all77'), default='test37')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('datasets/lafan/manifest.json'))
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--wait-for-data', action='store_true', help='Wait until every selected motion is fully readable')
    p.add_argument('--sleep', type=float, default=.02)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if a.wait_for_data:
        while True:
            try:
                ready = select_motion_paths(sorted(a.motions.rglob('motion.npz')), a.motion_split, a.manifest)
                for path in ready:
                    with np.load(path, allow_pickle=False) as motion:
                        if not np.isfinite(motion['state']).all():
                            raise FloatingPointError(f'Nonfinite motion: {path}')
                break
            except (ValueError, OSError, EOFError, BadZipFile) as exc:
                print(json.dumps({'event': 'waiting_for_data', 'reason': str(exc)}), flush=True)
                time.sleep(15)
    a.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint.get('learner_version') != VERSION:
        raise ValueError('Legacy checkpoints are retired; use a v2 checkpoint')
    if checkpoint.get('training_contract', {}).get('backend') != 'mujoco-warp':
        raise ValueError('Expected MuJoCo Warp checkpoint')
    # Copy only inference tensors; do not read/copy the large replay allocation.
    saved = {k: checkpoint[k] for k in ['config', 'joint_names', 'body_names', 'args', 'iteration', 'learner_version']}
    saved['training_contract'] = checkpoint['training_contract']
    saved['actor'] = {k: v.clone() for k, v in checkpoint['actor'].items()}
    torch.save(saved, a.output / 'actor_snapshot.pt')
    del checkpoint
    paths = select_motion_paths(sorted(a.motions.rglob('motion.npz')), a.motion_split, a.manifest)
    motions = Motions(paths, a.device)
    if motions.names != (saved['joint_names'], saved['body_names']):
        raise ValueError('Checkpoint/motion schema mismatch')
    env = MJWarp(saved['config'], motions.names, len(paths), a.device, randomize=False)
    env.layout = Layout(env.layout.joints, env.layout.bodies)
    actor = Actor(env.layout, saved['args']['small']).to(a.device).eval()
    actor.load_state_dict(saved['actor'])
    ids = torch.arange(len(paths), device=a.device)
    times = torch.zeros_like(ids)
    initial = motions.get(ids, times)[:, 0]
    env.reset(ids, initial, resample_context=False)
    env.verify_context()
    state = env.state()
    previous = torch.zeros((len(paths), env.layout.joints), device=a.device)
    history = env.layout.obs(state, previous, env.default, noise=False, goal=initial)[:, None].repeat(1, 10, 1)
    active = torch.ones(len(paths), device=a.device, dtype=torch.bool)
    failed = torch.zeros_like(active)
    total = torch.zeros(len(paths), device=a.device, dtype=torch.float64)
    executed = torch.zeros_like(ids)
    expected = motions.lengths - 1
    started = time.monotonic()
    print(json.dumps({'event': 'start', 'checkpoint': saved['iteration']+1, 'motions': len(paths), 'device': a.device}), flush=True)
    with torch.inference_mode():
        while bool(active.any()):
            goals = motions.get(ids, times, 11)
            action = actor(history, goals)
            if not torch.isfinite(action).all():
                raise FloatingPointError('Nonfinite policy action; evaluation invalid')
            # Match the action actually applied by train.py (without exploration noise).
            action = action.clamp(-20, 20)
            action[~active] = 0
            ns = env.step(action)
            next_goal = motions.get(ids, times+1)[:, 0]
            rewards = env.layout.reward(ns, next_goal, action, previous, env.limits)
            terminal = env.layout.done(ns, next_goal)
            if not torch.isfinite(rewards).all():
                raise FloatingPointError('Nonfinite reward; evaluation invalid')
            total += torch.where(active, rewards.double(), 0)
            executed += active.long()
            failed |= active & terminal
            times += active.long()
            active &= ~terminal & (times < expected)
            previous = action
            history = torch.cat((history[:, 1:], env.layout.obs(ns, previous, env.default, noise=False, goal=next_goal)[:, None]), 1)
            # Completed episodes no longer count. Keep their unused simulation
            # slots bounded without restarting any active evaluation trajectory.
            inactive = (~active).nonzero().flatten()
            if len(inactive):
                env.reset(inactive, initial[inactive], resample_context=False)
            if int(executed.max()) % 250 == 0:
                print(json.dumps({'step': int(executed.max()), 'active': int(active.sum()), 'failures': int(failed.sum()), 'seconds': time.monotonic()-started}), flush=True)
            time.sleep(a.sleep)
    scores = (total / (expected * .1)).cpu().tolist()
    records = [{'motion': path.parent.name, 'source_frames': int(motions.lengths[i]),
                'expected_transitions': int(expected[i]), 'executed_transitions': int(executed[i]),
                'failed': bool(failed[i]), 'return': float(total[i]), 'normalized_return': scores[i]}
               for i, path in enumerate(paths)]
    result = {'learner_version': VERSION, 'motion_split': a.motion_split, 'checkpoint_iterations': saved['iteration']+1, 'robot': saved['config']['robot'],
              'source_asset_sha256': saved['training_contract']['asset_sha256'],
              'motion_count': len(paths), 'normalized_return_mean': float(np.mean(scores)),
              'normalized_return_median': float(np.median(scores)),
              'normalized_return_frame_weighted': float(total.sum()/(expected.sum()*.1)),
              'failed_motions': int(failed.sum()), 'completed_motions': int((~failed).sum()),
              'evaluation_seconds': time.monotonic()-started,
              'protocol': {'reward': 'v2 BeyondMimic xy/yaw-aligned bodies, global root, 90% soft limits', 'start': 'frame 0, reference qpos/qvel', 'noise': False,
                           'domain_randomization': False, 'friction': 1.0, 'action_clip': [-20, 20],
                           'physics': 'MuJoCo Warp, same model/solver/PD as training',
                           'termination': 'reference root height error >0.25m or gravity-z difference >0.8',
                           'return': 'undiscounted reward through terminal step / (0.1 * (source_frames-1)); no clipping',
                           'aggregation': 'arithmetic mean of per-motion normalized returns',
                           'future_goals': 'last frame repeated at clip end'},
              'snapshot_sha256': hashlib.sha256((a.output/'actor_snapshot.pt').read_bytes()).hexdigest(),
              'motions': records}
    provenance = a.checkpoint.with_suffix('.provenance.json')
    if provenance.exists():
        result['checkpoint_provenance'] = json.loads(provenance.read_text())
    (a.output/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['motions','protocol']}), flush=True)


if __name__ == '__main__':
    main()
