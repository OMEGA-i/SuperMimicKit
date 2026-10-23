"""Export a trusted evaluation checkpoint as a portable FP32 policy-only bundle."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from scripts.export_web_onnx import PolicyGraph
from supermimic.core_v2 import Actor, Layout
from supermimic.physx import actuator_gains

ROBOTS = ('g1', 'h2', 'a3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(checkpoint, output, robot, iterations):
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['iteration'] + 1 != iterations:
        raise ValueError('Checkpoint is not the requested exact iteration')
    if saved.get('learner_version') != 2:
        raise ValueError('Checkpoint robot or learner does not match')
    # Only the published config contract is permitted in the release bundle.
    public = json.loads((Path(__file__).resolve().parents[1] / 'configs' / f'{robot}.json').read_text())
    if saved['config'] != public:
        raise ValueError('Checkpoint config differs from published robot config')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    layout = Layout(len(saved['joint_names']), len(saved['body_names']))
    actor = Actor(layout, saved['args']['small']).cpu().eval()
    actor.load_state_dict(saved['actor'])
    graph = PolicyGraph(actor).eval()
    path = output / 'policy.onnx'
    inputs = (torch.zeros(1, 10, layout.obs_dim), torch.zeros(1, layout.goal_dim))
    torch.onnx.export(graph, inputs, str(path), input_names=['history', 'goal_features'],
                      output_names=['action'], opset_version=17, dynamo=False)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer):
        raise ValueError('Unexpected FP16 initializer')
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(50_000)
    error = 0.; fixtures = []
    with torch.no_grad():
        for _ in range(16):
            history = actor.obs_norm.mean.numpy() + actor.obs_norm.std().numpy() * rng.normal(0, .3, (1, 10, layout.obs_dim)).astype('float32')
            goals = actor.goal_norm.mean.numpy() + actor.goal_norm.std().numpy() * rng.normal(0, .3, (1, layout.goal_dim)).astype('float32')
            expected = graph(torch.from_numpy(history), torch.from_numpy(goals)).numpy()
            actual = session.run(None, {'history': history, 'goal_features': goals})[0]
            if not np.isfinite(actual).all():raise ValueError('Nonfinite ONNX output')
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            error = max(error, float(np.max(np.abs(actual - expected))))
            fixtures.append(dict(history=history.tolist(), goal_features=goals.tolist(), action=expected.tolist()))
    default = []; scales = []; gains = []
    for joint in saved['joint_names']:
        values = [v for pattern, v in public['default_pose'].items() if re.fullmatch(pattern, joint)]
        if len(values)>1:raise ValueError('Ambiguous default pose')
        default.append(values[0] if values else 0.)
        matches = [a for a in public['actuators'] if re.fullmatch(a['pattern'], joint)]
        if len(matches)!=1:raise ValueError('Ambiguous actuator')
        kp,kd = actuator_gains(matches[0]);scales.append(.25*matches[0]['effort']/kp)
        gains.append(dict(stiffness=kp,damping=kd,effort=matches[0]['effort']))
    metadata = dict(format='supermimickit-policy-v1',robot=robot,completed_iterations=iterations,
        learner_version=2,precision='float32',opset=17,contains_world_model=False,
        source_checkpoint_sha256=sha(checkpoint),source_asset_sha256=saved['training_contract']['asset_sha256'],
        training_backend=saved['training_contract']['backend'],training_precision=saved['training_contract']['precision'],
        motion_split=saved['training_contract']['motion_split'],policy_sha256=sha(path),
        inputs={'history':[1,10,layout.obs_dim],'goal_features':[1,layout.goal_dim]},output={'action':[1,layout.joints]},
        normalization='embedded; pass unnormalized observation/goal features',
        history_order='oldest to newest',joint_names=saved['joint_names'],body_names=saved['body_names'],
        default_pose=default,action_scale=scales,actuator_gains=gains,control_dt=.02,
        action_interpretation='PD target = default_pose + action_scale * action; apply matching training joint-limit handling',
        feature_implementation='supermimic.core_v2.Layout.obs and Layout.goals',
        normalizers={name:{'mean':norm.mean.tolist(),'std':norm.std().tolist()} for name,norm in [('obs',actor.obs_norm),('goal',actor.goal_norm)]})
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(public,indent=2)+'\n')
    (output/'inference_fixture.json').write_text(json.dumps(fixtures)+'\n')
    report=dict(checks=16,onnx_checker=True,execution_provider='CPUExecutionProvider',torch_onnx_max_abs=error,
                atol=2e-4,rtol=2e-4,torch_version=torch.__version__,onnxruntime_version=ort.__version__)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'README.md').write_text(f'''# {robot.upper()} policy — {iterations:,} iterations

Unofficial SuperMimicKit policy, exported from the exact saved iteration in FP32.
This archive contains policy only: no world model, critic, optimizer, robot mesh,
or motion dataset. It is not a drop-in full browser-demo asset directory.

Inputs are float32 `history` {metadata['inputs']['history']} (oldest first) and
`goal_features` {metadata['inputs']['goal_features']}. Output `action` is
{metadata['output']['action']}. Batch size is fixed at 1. Normalization is
embedded: do not normalize inputs again. Build observations and ten future goal
features with `supermimic.core_v2.Layout.obs` / `Layout.goals`; eleven reference
states include the current frame. Joint/body order and PD control parameters
are in metadata.json. Use compatible robot physics and joint-limit handling.

```python
import numpy as np
import onnxruntime as ort
session = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])
action = session.run(None, {{"history": history.astype(np.float32),
                            "goal_features": goals.astype(np.float32)}})[0]
```

The graph is intended for ONNX Runtime, including its WebGPU backend.
Validation here compares PyTorch and ONNX Runtime CPU on 16 synthetic fixtures;
it does not establish browser GPU performance or locomotion quality. No
paper-equivalent return or success rate is claimed. metadata.json records the
checkpoint and model hashes without private runtime paths.
''')
    print(json.dumps(dict(robot=robot,iterations=iterations,policy_bytes=path.stat().st_size,**report)))
    return metadata, report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True,help='Trusted local checkpoint only')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--robot',choices=list(ROBOTS),required=True)
    p.add_argument('--iterations',type=int,default=50000)
    a=p.parse_args();export(a.checkpoint,a.output,a.robot,a.iterations)
