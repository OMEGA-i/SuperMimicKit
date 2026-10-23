"""Recover an evaluation actor from a preserved Web export, explicitly quantized."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.core_v2 import Actor, Layout
from supermimic.checkpoints import save_evaluation_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    meta = json.loads((a.assets/'metadata.json').read_text())
    saved = torch.load(a.template, map_location='cpu', weights_only=False)
    assert saved['joint_names'] == meta['joint_names']
    assert saved['body_names'] == meta['body_names']
    layout = Layout(len(meta['joint_names']), len(meta['body_names']))
    actor = Actor(layout, saved['args']['small']).eval()
    with torch.no_grad():
        for name in ('encoder', 'decoder'):
            layers = [m for m in getattr(actor, name) if isinstance(m, torch.nn.Linear)]
            exported = meta['networks'][name]['layers']
            assert len(layers) == len(exported)
            for layer, spec in zip(layers, exported):
                for key in ('weight', 'bias'):
                    descriptor = spec[key]
                    data = (a.assets/descriptor['file']).read_bytes()
                    assert hashlib.sha256(data).hexdigest() == descriptor['sha256']
                    value = np.frombuffer(data, dtype=descriptor['dtype']).reshape(descriptor['shape']).copy()
                    getattr(layer, key).copy_(torch.from_numpy(value).float())
        for name in ('obs', 'goal'):
            norm = getattr(actor, name+'_norm')
            spec = meta['normalizers'][name]
            norm.mean.copy_(torch.tensor(spec['mean']))
            norm.var.copy_(torch.tensor(spec['std']).square() - norm.eps**2)
            torch.testing.assert_close(norm.std(), torch.tensor(spec['std']))
        f = json.loads((a.assets/meta['inference_fixture']).read_text())
        actual = actor(torch.tensor(f['history']).reshape(1, 10, layout.obs_dim), torch.tensor(f['goals']).reshape(1, 11, layout.dim))
        error = (actual.flatten()-torch.tensor(f['action'])).abs().max().item()
        assert np.isfinite(error) and error <= .005, error
    saved['actor'] = actor.state_dict()
    saved['iteration'] = meta['completed_iterations'] - 1
    save_evaluation_checkpoint(saved, a.output)
    provenance = {'source': str(a.assets), 'checkpoint_iterations': meta['completed_iterations'], 'precision': 'Web FP16 linear weights restored to FP32; biases and normalization from export', 'fixture_action_max_error': error, 'template_used_for_config_only': str(a.template)}
    a.output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
