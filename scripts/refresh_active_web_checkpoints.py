"""Publish only newer checkpoints from the active training processes, on CPU."""
import fcntl
import json
import os
from pathlib import Path
from scripts.refresh_web_checkpoint import refresh

root = Path(os.environ.get('SUPERMIMIC_OUTPUT_ROOT', 'outputs'))
assets = Path(os.environ.get('SUPERMIMIC_WEB_ASSETS', 'web/demo/dist/assets'))
with open('/tmp/supermimic-web-checkpoint-refresh.lock', 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runs = {}
    for pidfile in root.glob('*.pid'):
        run = root / pidfile.stem
        try:
            pid = int(pidfile.read_text())
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'supermimic.train' not in args or os.fsencode(run) not in args:
                continue
            config = json.loads((run / 'run.json').read_text())
            robot = Path(config['config']).stem
            if robot not in ('g1', 'h2', 'a3'):
                continue
            if robot in runs:
                raise RuntimeError(f'Multiple active runs for {robot}')
            runs[robot] = run
        except (FileNotFoundError, ProcessLookupError):
            continue
    errors = []
    for robot, run in sorted(runs.items()):
        try:
            refresh(run / 'last.pt', assets / robot)
        except Exception as error:
            errors.append(f'{robot}: {error}')
    if errors:
        raise RuntimeError('; '.join(errors))
    print(json.dumps({'active_robots': sorted(runs)}), flush=True)
