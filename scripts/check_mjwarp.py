"""GPU integration check: UMR FK/velocities, selective resets and physical context."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.data import Motions
from supermimic.mjwarp import MJWarp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--motions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--failure-snapshot', type=Path, help='Trusted local transition snapshot to replay')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    motions = Motions([sorted(args.motions.rglob('motion.npz'))[0]], args.device)
    env = MJWarp(cfg, motions.names, 4, args.device)
    ids = torch.arange(4, device=args.device)
    frames = torch.linspace(0, len(motions.clips[0])-1, 4, device=args.device).long()
    target = motions.get(torch.zeros_like(ids), frames)[:, 0]
    env.reset(ids, target)
    actual = env.state()
    # Quaternion signs are physically equivalent; align before comparison.
    def align_quaternion(a, b):
        a *= torch.where((a*b).sum(-1, keepdim=True) < 0, -1, 1)
    align_quaternion(actual[:, 3:7], target[:, 3:7])
    _, _, _, bodies = env.layout.parts(actual)
    _, _, _, reference = env.layout.parts(target)
    align_quaternion(bodies[..., 3:7], reference[..., 3:7])
    error = float((actual-target).abs().max())
    torch.testing.assert_close(actual, target, atol=3e-4, rtol=3e-4)
    env.randomize = True
    env.reset(ids, target)
    saved = {k:v.clone() for k,v in env.randomization_state().items()}
    initial = env.state().clone()
    env.reset(ids[:2], target[:2])
    torch.testing.assert_close(env.context[2:], saved['context'][2:], rtol=0, atol=0)
    torch.testing.assert_close(env.state()[2:], initial[2:], atol=3e-4, rtol=3e-4)
    env.restore_randomization(saved)
    torch.testing.assert_close(env.context, saved['context'], rtol=0, atol=0)
    env.verify_context()
    for _ in range(10):
        state = env.step(torch.zeros(4, env.layout.joints, device=args.device))
        assert torch.isfinite(state).all()
    report = {'backend':'mujoco-warp','config':str(args.config),'max_state_reset_error':error,
              'selective_reset':True,'context_restore':True,'finite_steps':10}
    if args.failure_snapshot:
        snapshot = torch.load(args.failure_snapshot, map_location=args.device, weights_only=True)
        assert len(snapshot['context']) == 1, 'Expected a single failed world'
        env.apply_context(ids, snapshot['context'].expand(4, -1))
        for key in ('qpos', 'qvel', 'qacc_warmstart'):
            env.view(key).copy_(snapshot[key].expand_as(env.view(key)))
        state = env.step(snapshot['action'].expand(4, -1))
        assert torch.isfinite(state).all()
        report['failed_transition_replay_finite'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
