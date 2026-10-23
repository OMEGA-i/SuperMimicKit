"""Preserve exact evaluation actor weights without changing a running trainer."""
import argparse
import json
import time
from pathlib import Path
import torch
from supermimic.checkpoints import save_evaluation_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--iteration', type=int, default=20000)
    p.add_argument('--every', type=int, default=0, help='Continuously preserve multiples of this interval')
    a = p.parse_args()
    if a.every < 0:
        p.error('--every must be nonnegative')
    torch.set_num_threads(2)
    pending = set(a.runs)
    seen = {}
    while pending:
        for run in sorted(pending):
            if not a.every and (run / f'eval_{a.iteration}.pt').exists():
                pending.remove(run)
                continue
            path = run / 'last.pt'
            try:
                stat = path.stat()
            except FileNotFoundError:
                # A fresh run has no checkpoint until its first save interval.
                continue
            version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if seen.get(run) == version:
                continue
            d = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            n = d['iteration'] + 1
            seen[run] = version
            if not a.every and n > a.iteration:
                raise RuntimeError(f'{run}: missed {a.iteration}, current {n}')
            if (a.every and n % a.every == 0) or (not a.every and n == a.iteration):
                out = run / f'eval_{n}.pt'
                if save_evaluation_checkpoint(d, out):
                    print(json.dumps({'saved': str(out), 'iterations': n}), flush=True)
                if not a.every:
                    pending.remove(run)
            del d
        if pending:
            time.sleep(15)


if __name__ == '__main__':
    main()
