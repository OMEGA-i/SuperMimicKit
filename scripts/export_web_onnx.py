"""FP32 ONNX network graphs; geometric feature/FK code stays outside the graph."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class PolicyGraph(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, history, goal_features):
        a = self.actor
        z = torch.tanh(a.encoder(a.goal_norm(goal_features)))
        return a.decoder(torch.cat((a.obs_norm(history).flatten(-2), z), -1))


class DynamicsGraph(nn.Module):
    def __init__(self, world):
        super().__init__()
        self.world = world

    def forward(self, features):
        w = self.world
        return w.dynamics(w.in_norm(features)) * w.out_norm.std() + w.out_norm.mean


def export_networks(output, actor, world):
    import onnx
    import onnxruntime as ort
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    layout = actor.layout
    graphs = {
        'policy': (PolicyGraph(actor).eval(), (torch.zeros(1, 10, layout.obs_dim), torch.zeros(1, layout.goal_dim)), ['history', 'goal_features'], 'action'),
        'dynamics': (DynamicsGraph(world).eval(), (torch.zeros(1, world.in_norm.mean.numel()),), ['features'], 'delta'),
    }
    result = {}
    rng = np.random.default_rng(17)
    for name, (module, inputs, names, outname) in graphs.items():
        path = output / (name + '.onnx')
        torch.onnx.export(module, inputs, str(path), input_names=names, output_names=[outname],
                          opset_version=17, dynamo=False)
        model = onnx.load(path)
        onnx.checker.check_model(model)
        if any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer):
            raise ValueError('FP16 weights are not allowed')
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
        error = 0.0
        with torch.no_grad():
            for _ in range(8):
                values = [rng.normal(0, .1, tuple(x.shape)).astype('float32') for x in inputs]
                expected = module(*(torch.from_numpy(x) for x in values)).numpy()
                actual = session.run(None, dict(zip(names, values)))[0]
                np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
                error = max(error, float(np.max(np.abs(actual-expected))))
        result[name] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'bytes': path.stat().st_size, 'inputs': {n:list(x.shape) for n,x in zip(names,inputs)},
                        'output': outname, 'cpu_parity_max_abs': error}
    return result


def rollout_fixture(actor, world, fixture, context, default, steps=32):
    """Closed-loop world rollout reference, including evolving actor history."""
    layout = actor.layout
    h = torch.tensor(fixture['history']).reshape(1,10,layout.obs_dim)
    g = torch.tensor(fixture['goals']).reshape(1,11,layout.dim)
    s = torch.tensor(fixture['state']).reshape(1,layout.dim)
    c = torch.tensor(context).reshape(1,-1)
    rows=[]
    with torch.no_grad():
        for _ in range(steps):
            a=actor(h,g); s=world(s,a,c)
            rows.append({'action':a.flatten().tolist(),'state':s.flatten().tolist()})
            obs=layout.obs(s,a,torch.tensor(default),noise=False,goal=g[:,1])
            h=torch.cat((h[:,1:],obs[:,None]),1)
    return rows
