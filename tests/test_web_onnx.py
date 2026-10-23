import json
import subprocess
from pathlib import Path
import torch
from test_learner_v2 import fixture
from supermimic.core_v2 import Actor, WorldModel
from scripts.export_web_onnx import export_networks, rollout_fixture
from scripts.export_web_demo import inference_metadata


def test_onnx_fp32_browser_boundary_and_rollout(tmp_path):
    torch.manual_seed(123)
    _, fk, layout, states = fixture()
    actor=Actor(layout, small=True).eval()
    world=WorldModel(layout, layout.joints+4, fk, width=32).eval()
    # Nonzero dynamics and nontrivial normalization catch missing transforms.
    with torch.no_grad():
        world.dynamics[-1].weight.normal_(0,.001)
        actor.obs_norm.mean.fill_(.03)
        actor.goal_norm.var.fill_(1.7)
        world.in_norm.mean.fill_(.02)
        world.out_norm.var.fill_(.001)
    s=states[:1];g=s[:,None].repeat(1,11,1);default=[0.]
    h=layout.obs(s,torch.zeros(1,1),torch.tensor(default),noise=False,goal=s)[:,None].repeat(1,10,1)
    c=[1.,0.,0.,0.,0.]
    with torch.no_grad():
        a=actor(h,g);w=world(s,a,torch.tensor(c)[None])
    f=dict(history=h.flatten().tolist(),goals=g.flatten().tolist(),state=s.flatten().tolist(),action=a.flatten().tolist(),world_next=w.flatten().tolist())
    f['rollout']=rollout_fixture(actor,world,f,c,default)
    meta=dict(inference_backend='onnxruntime-webgpu-f32',onnx=export_networks(tmp_path,actor,world),layout=dict(joints=1,bodies=3,dim=layout.dim,obs_dim=layout.obs_dim),default_pose=default,context=c,inference_fixture='fixture.json',**inference_metadata(actor,world))
    (tmp_path/'fixture.json').write_text(json.dumps(f));(tmp_path/'metadata.json').write_text(json.dumps(meta))
    subprocess.run(['node','scripts/check_web_v2.mjs',str(tmp_path)],check=True)
