"""Replace demo snippets with every full retargeted LaFAN motion (CPU only)."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from supermimic.data import select_motion_paths


def export(motions: Path, assets: Path, expected_count: int):
    metadata_path = assets / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    sources = select_motion_paths(sorted(motions.rglob('motion.npz')), 'test37')
    if len(sources) != expected_count:
        raise ValueError(f'Expected {expected_count} motions, found {len(sources)}')
    ids = [p.parent.name for p in sources]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate motion IDs')
    output = assets / 'motions_full'
    output.mkdir(exist_ok=True)
    clips = []
    for source in sources:
        with np.load(source, allow_pickle=False) as data:
            if data['joint_names'].tolist() != metadata['joint_names'] or data['body_names'].tolist() != metadata['body_names']:
                raise ValueError(f'Schema mismatch: {source}')
            fps = float(data['fps'])
            state = np.asarray(data['state'], dtype='<f4')
            if fps != 50 or state.ndim != 2 or state.shape[1] != metadata['layout']['dim'] or len(state) < 2 or not np.isfinite(state).all():
                raise ValueError(f'Invalid motion: {source}')
            original_frames = len(state)
            # Preserve every source frame. Terminal padding supplies future goals,
            # but is excluded from playback duration.
            state = np.concatenate([state, np.repeat(state[-1:], metadata['goal_frames'], axis=0)])
        raw = state.tobytes(order='C')
        packed = gzip.compress(raw, compresslevel=1, mtime=0)
        if gzip.decompress(packed) != raw:
            raise ValueError('Lossless motion compression verification failed')
        motion_id = source.parent.name
        if not re.fullmatch(r'[A-Za-z0-9_-]+', motion_id):
            raise ValueError(f'Unsafe motion ID: {motion_id}')
        name = f'motions_full/{motion_id}.bin.packed'
        temporary = assets / (name + '.tmp')
        temporary.write_bytes(packed)
        os.replace(temporary, assets / name)
        clips.append({'id': motion_id, 'label': re.sub(r'\d+$', '', motion_id.split('_subject')[0]),
                      'fps': fps, 'frames': len(state), 'playable_frames': original_frames,
                      'source_frames': original_frames, 'duration_seconds': original_frames / fps,
                      'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'state': {'file': name, 'shape': list(state.shape), 'dtype': 'float32',
                                'compression': 'gzip', 'bytes': len(packed),
                                'sha256': hashlib.sha256(packed).hexdigest()}})
        print(f'{metadata["robot"]} {len(clips)}/{len(sources)} {motion_id} {original_frames} frames', flush=True)
    metadata['motions'] = clips
    metadata['motion_dataset'] = {'name': 'LaFAN', 'retargeter': 'UMR', 'scope': 'held-out test37',
                                  'count': len(clips), 'source_frames': sum(c['source_frames'] for c in clips),
                                  'duration_seconds': sum(c['duration_seconds'] for c in clips)}
    temporary = assets / 'metadata.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary, metadata_path)
    print(json.dumps(metadata['motion_dataset']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=37)
    args = parser.parse_args()
    export(args.motions, args.assets, args.expected_count)
