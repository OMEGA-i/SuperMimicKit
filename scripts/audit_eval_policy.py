"""Compare evaluation and training policy/observation math on fixed references."""
import argparse,json,re
from pathlib import Path
import numpy as np
import torch
from supermimic.core_v2 import Actor,Layout,Learner
from supermimic.precision import configure_precision


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--motions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 l=Layout(len(c['joint_names']),len(c['body_names']));actor=Actor(l,c['args']['small']).cuda().eval();actor.load_state_dict(c['actor'])
 seq=[]
 for path in sorted(a.motions.rglob('motion.npz')):
  with np.load(path) as d:
   for t in [0,len(d['state'])//2]:seq.append(torch.tensor(d['state'][t:t+11]))
 goals=torch.stack(seq).cuda();state=goals[:,0];previous=torch.zeros(len(seq),l.joints,device='cuda');default=torch.zeros(l.joints,device='cuda')
 for j,name in enumerate(c['joint_names']):
  for pattern,value in c['config']['default_pose'].items():
   if re.fullmatch(pattern,name):default[j]=value
 context=torch.zeros(len(seq),l.joints+4,device='cuda');context[:,0]=1
 obs=l.obs(state,previous,default,noise=False,goal=state)
 with_context=l.obs(state,previous,default,noise=False,context=context,goal=state)
 h=obs[:,None].repeat(1,10,1)
 with torch.inference_mode():
  configure_precision('fp32');fp32=actor(h,goals).clone()
  configure_precision('tf32');tf32=actor(h,goals).clone()
  compiled=torch.compile(actor.forward);compiled_out=compiled(h,goals).clone()
  error=float((compiled_out-tf32).abs().max())
 result={'reference_windows':len(seq),'zero_context_observation_error':float((obs-with_context).abs().max()),'tf32_vs_fp32_max_action_error':float((tf32-fp32).abs().max()),'compiled_vs_eager_tf32_max_action_error':error,'max_abs_action':float(fp32.abs().max()),'train_clamp_20_changes_any':bool((fp32.abs()>20).any()),'finite':bool(torch.isfinite(compiled_out).all())}
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
