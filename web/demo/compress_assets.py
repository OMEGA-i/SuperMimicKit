"""Losslessly pack browser binaries with deterministic gzip under .packed names.

The browser explicitly decompresses these files; .packed avoids a host treating
precompressed files as transparent HTTP gzip. Repeating the command is safe.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def compress(root):
    reports = []
    for metadata_path in sorted(root.glob('*/metadata.json')):
        base = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        originals = set()
        verified = 0

        def pack(name):
            nonlocal verified
            source = base / name
            dest_name = name + '.packed'
            dest = base / dest_name
            raw = source.read_bytes()
            packed = gzip.compress(raw, compresslevel=9, mtime=0)
            if hashlib.sha256(gzip.decompress(packed)).digest() != hashlib.sha256(raw).digest():
                raise ValueError(f'Compression verification failed: {source}')
            temporary = dest.with_suffix('.tmp')
            temporary.write_bytes(packed)
            temporary.replace(dest)
            # Read back from disk before publishing descriptors/deleting originals.
            if gzip.decompress(dest.read_bytes()) != raw:
                raise ValueError(f'Written compression verification failed: {dest}')
            originals.add(source)
            verified += 1
            return dest_name, len(packed), hashlib.sha256(packed).hexdigest(), hashlib.sha256(raw).hexdigest()

        def descriptors(value):
            if isinstance(value, dict):
                if 'file' in value and 'dtype' in value and not value.get('compression'):
                    name, size, digest, unpacked_digest = pack(value['file'])
                    value.update(file=name, compression='gzip', bytes=size, sha256=digest,
                                 uncompressed_sha256=unpacked_digest)
                else:
                    for child in value.values():
                        descriptors(child)
            elif isinstance(value, list):
                for child in value:
                    descriptors(child)

        descriptors(metadata)
        mapping = metadata['model'].setdefault('compressed_files', {})
        for name in metadata['model']['files']:
            if name not in mapping:
                mapping[name] = pack(name)[0]
        # Metadata is replaced first: an interruption can leave redundant originals,
        # never a descriptor pointing to an absent asset.
        metadata['asset_compression'] = 'gzip payloads; explicit DecompressionStream; lossless'
        temporary = metadata_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(metadata_path)
        for source in originals:
            source.unlink()
        metadata['total_bytes'] = sum(p.stat().st_size for p in base.rglob('*') if p.is_file() and p != metadata_path)
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(metadata_path)
        reports.append({'robot': metadata['robot'], 'lossless_files_verified': verified,
                        'total_bytes': metadata['total_bytes']})
    total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    print(json.dumps({'robots': reports, 'total_bytes': total, 'total_MiB': total / 2**20}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path, nargs='?', default=Path(__file__).parent / 'dist/assets')
    compress(parser.parse_args().assets)
