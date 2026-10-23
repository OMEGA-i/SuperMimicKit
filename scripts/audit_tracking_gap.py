"""Read-only short rollouts separating reset distribution and evaluation settings."""
import argparse,json,os,time
from pathlib import Path
import torch
from supermimic.core_v2 import Actor,Layout
from supermimic.data import Motions,select_motion_paths
from supermimic.mjwarp import MJWarp
from supermimic.precision import configure_precision

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--motion-split',default='train40');p.add_argument('--reset-inactive',action='store_true');p.add_argument('--repeat',type=int,default=1);p.add_argument('--steps',type=int,default=500);p.add_argument('--root-inertia-probe',action='store_true');p.add_argument('--modes',nargs='+',default=['frame0','frame0_noise','random_reference','saved_runtime','saved_runtime_noise']);a=p.parse_args();c=json.loads(a.config.read_text());os.environ.update(c['asset_env']);torch.set_num_threads(2);torch.manual_seed(42)
 ck=torch.load(a.checkpoint or Path(c['training_run'])/'last.pt',map_location='cpu',mmap=True,weights_only=False);dev='cuda:0';configure_precision('tf32')
 if a.root_inertia_probe:
  import mujoco as mj
  import supermimic.mjwarp as backend
  original_build=backend.build_model
  def patched_build(cfg,names):
   model,params=original_build(cfg,names);root=int(model.jnt_bodyid[0]);model.body_mass[root]=7.60468;model.body_inertia[root]=[.0355137,.0275324,.0228779];model.body_ipos[root]=[.00046244,-.00004214,-.0147344];model.body_iquat[root]=[.707546,.706201,-.0235095,-.0102787];mj.mj_setConst(model,mj.MjData(model));return model,params
  backend.build_model=patched_build
 paths=select_motion_paths(sorted(Path(c['motions']).rglob('motion.npz')),a.motion_split,Path(c['manifest']));motions=Motions(paths,dev);original_n=len(paths);n=original_n*a.repeat;idx=torch.arange(n,device=dev);env=MJWarp(ck['config'],motions.names,n,dev,randomize=False);env.layout=Layout(env.layout.joints,env.layout.bodies);actor=Actor(env.layout,ck['args']['small']).to(dev).eval();actor.load_state_dict(ck['actor']);results=[]
 for mode in a.modes:
  configure_precision('fp32' if mode in ['frame0_fp32','frame0_original'] else 'tf32')
  torch.manual_seed(42);saved=mode.startswith('saved');noisy=mode.endswith('noise')
  def observe(s, previous, goal):
   clean=env.layout.obs(s,previous,env.default,noise=False,context=env.context,goal=goal)
   if mode in ['frame0_action_only','frame0_dr'] or not (noisy or mode in ['frame0_obs_only','frame0_paper_obs','frame0_extra_obs']):return clean
   full=env.layout.obs(s,previous,env.default,noise=True,context=env.context,goal=goal)
   if mode=='frame0_paper_obs':full[...,-6:]=clean[...,-6:]
   if mode=='frame0_extra_obs':full[...,:-6]=clean[...,:-6]
   return full
  if saved:
   rt=ck['runtime'];ids=rt['ids'][:n].to(dev);times=rt['times'][:n].to(dev);initial=rt['state'][:n].to(dev);prev=rt['previous'][:n].to(dev);hist=rt['history'][:n].to(dev);env.apply_context(idx,rt['randomization']['context'][:n].to(dev))
  else:
   ids=idx.remainder(original_n);times=torch.zeros_like(ids) if mode.startswith('frame0') else (torch.rand(n,device=dev)*(motions.lengths-550).clamp_min(1)).long();initial=motions.get(ids,times)[:,0];prev=torch.zeros(n,env.layout.joints,device=dev);ctx=torch.zeros_like(env.context);ctx[:,0]=1
   if mode=='frame0_dr':
    ctx[:,0]=.3+.9*torch.rand(n,device=dev);ctx[:,1:-3]=.02*torch.rand_like(ctx[:,1:-3])-.01;ctx[:,-3:]=(2*torch.rand_like(ctx[:,-3:])-1)*ctx.new_tensor([.025,.05,.05])
   if mode not in ['frame0_raw','frame0_original']:env.apply_context(idx,ctx)
  env.reset(idx,initial,resample_context=False);state=env.state();initial_error=(state-initial).abs().max().item()
  if not saved:hist=observe(state,prev,motions.get(ids,times)[:,0])[:,None].repeat(1,10,1)
  alive=torch.ones(n,device=dev,dtype=torch.bool);total=torch.zeros(n,device=dev);steps=torch.zeros(n,device=dev);failed=torch.zeros_like(alive);maxaction=0
  with torch.inference_mode():
   for k in range(a.steps):
    goals=motions.get(ids,times,11);act=actor(hist,goals)
    if noisy or mode=='frame0_action_only':act=act+.1*torch.randn_like(act)
    if mode=='frame0_zero_add':act=act+torch.zeros_like(act)
    if mode=='frame0_clone':act=act.clone()
    maxaction=max(maxaction,float(act.abs().max()));act=act if mode in ['frame0_unclamped','frame0_original'] else act.clamp(-20,20);ns=env.step(act);ng=motions.get(ids,times+1)[:,0];reward=env.layout.reward(ns,ng,act,prev,env.limits);terminal=env.layout.done(ns,ng)
    total+=torch.where(alive,reward,0);steps+=alive;failed|=alive&terminal;times+=1;alive&=~terminal&(times<motions.lengths[ids]-1);hist=torch.cat((hist[:,1:],observe(ns,act,ng)[:,None]),1);prev=act
    if a.reset_inactive:
     inactive=(~alive).nonzero().flatten()
     if len(inactive):env.reset(inactive,initial[inactive],resample_context=False)
    if not alive.any():break
  row={'mode':mode,'checkpoint':ck['iteration']+1,'failed':int(failed.sum()),'alive':int(alive.sum()),'mean_steps':float(steps.mean()),'mean_step_reward':float(total.sum()/steps.sum()/.1),'return_500':float((total/50).mean()),'initial_state_max_error':initial_error,'max_action':maxaction};results.append(row);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
if __name__=='__main__':main()
