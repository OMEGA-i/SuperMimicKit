"""Apply the public integration overlay to the pinned public UMR checkout."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = 'd1cdaedaf67a827198da71c5fa831f48b2e51f91'


def main():
    target = ROOT / 'third_party/UMR'
    revision = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError(f'Expected UMR {PIN}; got {revision}')
    source = ROOT / 'third_party/umr_overlay'
    files = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name == 'README.md' or '__pycache__' in path.parts:
            continue
        dest = target / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files.append(str(path.relative_to(source)))
    print(json.dumps({'upstream': revision, 'overlay_files': len(files)}))


if __name__ == '__main__':
    main()
