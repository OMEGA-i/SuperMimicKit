"""Convert exported MLP matrices to float16 storage, preserving float32 biases.

Idempotent: only tensors whose metadata says float32 are converted. Browser
inference must decode storage to float32 before doing any computation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compact(root):
    reports = []
    for metadata_path in sorted(root.glob('*/metadata.json')):
        base = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        saved = 0
        for network in metadata['networks'].values():
            for layer in network['layers']:
                spec = layer['weight']
                path = base / spec['file']
                dtype = np.dtype(spec['dtype'])
                if dtype == np.dtype('float16'):
                    continue
                if dtype != np.dtype('float32'):
                    raise ValueError(f'Unexpected matrix dtype: {dtype}')
                weights = np.fromfile(path, dtype='<f4')
                if weights.size != int(np.prod(spec['shape'])):
                    raise ValueError(f'Invalid weight shape: {path}')
                compacted = weights.astype('<f2')
                if not np.isfinite(compacted).all():
                    raise ValueError(f'Float16 overflow: {path}')
                temporary = path.with_suffix('.tmp')
                compacted.tofile(temporary)
                saved += path.stat().st_size - temporary.stat().st_size
                temporary.replace(path)
                spec.update(dtype='float16', bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        metadata['weight_storage'] = 'float16 matrices; float32 biases; float32 inference'
        metadata['total_bytes'] = sum(p.stat().st_size for p in base.rglob('*') if p.is_file() and p != metadata_path)
        temporary = metadata_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(metadata_path)
        reports.append({'robot': metadata['robot'], 'saved_bytes': saved,
                        'total_bytes': metadata['total_bytes']})
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path, nargs='?', default=Path(__file__).parent / 'dist/assets')
    compact(parser.parse_args().assets)
