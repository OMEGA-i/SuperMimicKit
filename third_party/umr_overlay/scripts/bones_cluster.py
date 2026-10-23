#!/usr/bin/env python3
"""Resume BONES uniform BVH -> in-memory SOMA 50 Hz -> UMR -> named motions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from grail_cluster import ROOT, ROBOTS, read, write, run, execute_phase
NODES = list(json.loads(os.environ.get('UMR_CLUSTER_NODES', '{"local": 1}')))


def initialize(root, original, supermimic):
    """Wait for the independent rsync, then verify relative names and byte sizes."""
    source=root/'source/motions_uniform/bvh'
    manifest=root/'source/inventory.json'
    if manifest.exists():return read(manifest)
    originals={str(p.relative_to(original)):p.stat().st_size for p in original.rglob('*.bvh')}
    if not originals:raise ValueError('Empty source')
    while True:
        missing=[k for k,size in originals.items() if not (source/k).is_file() or (source/k).stat().st_size!=size]
        write(root/'umr/pipeline/source.json',dict(status='copying' if missing else 'complete',total=len(originals),remaining=len(missing),updated_at=time.time()))
        if not missing:break
        print('Waiting for source copy:',len(missing),flush=True);time.sleep(60)
    source_assets=original.parents[1]
    for name in ['shapes','soma_assets']:
        dst=root/'source'/name
        if not dst.exists():shutil.copytree(source_assets/name,dst,symlinks=False)
    plan=dict(source='bones-studio/seed',source_revision='2f59b2077b9da34dd4e43618e705c7cb962c9a66',
              sequences=sorted(originals),sizes=originals,nodes=NODES,target_fps=50)
    plan['plan_id']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    write(manifest,plan)
    write(root/'manifest.json',dict(dataset='bones-seed',source=plan['source'],source_revision=plan['source_revision'],
          status='preparing',target_fps=50,pipeline=['source','umr','motions'],expected_per_robot=len(originals),robots=list(ROBOTS)))
    return plan


def configuration(root, robot, motion, out):
    cfgpath=ROOT/f'robot_configs/humanoid_retarget_{ROBOTS[robot]}_example.json'
    cfg=read(cfgpath)
    cfg['robot']['xml']=str(Path(os.environ.get('SUPERMIMIC_ROBOT_ASSETS', str(ROOT.parents[1]/'assets/robots')))/robot/'robot.xml')
    usd=str(root/'source/shapes/soma_base_rig/soma_base_skel_minimal.usd')
    cfg.update(soma_usd_path=usd,smpl_template=dict(source='motion',type='soma',name='auto',use_gender=False,soma_usd_path=usd),
               motion=dict(data=str(motion),seq_key=motion.stem,target_fps=50,stride=1),view=dict(enabled=False))
    corr=cfg.setdefault('correspondence',{});corr.update(smpl_name='auto',slots=str(root/'umr'/robot/'correspondence/correspondence_slots_final.npz'))
    corr.setdefault('dataset',{}).update(out=str(root/'umr'/robot/'correspondence.npz'),smpl_models=[])
    corr.setdefault('train',{}).update(out_dir=str(root/'umr'/robot/'correspondence'),device='cpu',template_name='auto')
    cfg['retarget']=dict(out=str(out),stream_chunk_frames=64)
    return cfg


def prepare(root,plan,robot):
    ready=root/'umr'/robot/'ready.json'
    if ready.exists() and read(ready).get('plan_id')==plan['plan_id']:return
    motion=root/'source/motions_uniform/bvh'/plan['sequences'][0]
    job=root/'umr'/robot/'validation';job.mkdir(parents=True,exist_ok=True)
    cfg=configuration(root,robot,motion,job/'retarget.npz');cfg['motion']['max_frames']=64
    config=job/'config.json';write(config,cfg)
    run([sys.executable,ROOT/'scripts/humanoid_retarget_pipeline.py','--config',config,
         '--defaults',ROOT/'humanoid_retarget_defaults_bones_seed.json','--skip-view'],job/'umr.log')
    import numpy as np
    with np.load(job/'retarget.npz',allow_pickle=True) as a:
        assert float(a['fps'].item())==50 and np.isfinite(a['qpos']).all()
    write(ready,dict(plan_id=plan['plan_id'],status='ready',visual_qc='pending',updated_at=time.time()))


def process(root,plan,item,supermimic):
    robot,key=item;job=root/'umr'/robot/'clips'/Path(key).with_suffix('');marker=job/'status.json'
    if marker.exists() and read(marker).get('status')=='completed' and read(marker).get('plan_id')==plan['plan_id']:return
    job.mkdir(parents=True,exist_ok=True);out=job/'retarget.npz'
    cfg=configuration(root,robot,root/'source/motions_uniform/bvh'/key,out)
    config=job/'config.json';write(config,cfg)
    try:
        write(marker,dict(status='running',plan_id=plan['plan_id']))
        run([sys.executable,ROOT/'scripts/humanoid_retarget_pipeline.py','--config',config,
             '--defaults',ROOT/'humanoid_retarget_defaults_bones_seed.json','--stage','retarget','--skip-view'],job/'umr.log')
        import numpy as np
        with np.load(out,allow_pickle=True) as a:
            assert float(a['fps'].item())==50 and np.isfinite(a['qpos']).all()
            xml=Path(str(a['robot_xml'].item()))
        sys.path.insert(0,str(supermimic))
        from supermimic.retarget import convert
        dest=root/'motions'/robot/Path(key).with_suffix('');temp=dest.with_name(dest.name+'.partial')
        if not dest.exists():
            if temp.exists():shutil.rmtree(temp)
            bodies=read(supermimic/'configs/retarget'/f'{robot}.json')['tracked_bodies']
            convert(out,temp,xml,bodies,ROBOTS[robot]);os.replace(temp,dest)
        write(marker,dict(status='completed',plan_id=plan['plan_id'],fps=50,motion=str(dest),updated_at=time.time()))
    except Exception as e:
        write(marker,dict(status='failed',plan_id=plan['plan_id'],error=str(e)));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original',type=Path)
    p.add_argument('--supermimic',type=Path,default=ROOT.parents[1])
    p.add_argument('--prepare',action='store_true');p.add_argument('--prepare-robot',choices=list(ROBOTS));p.add_argument('--node',choices=NODES);p.add_argument('--workers',type=int,default=40)
    a=p.parse_args();root=a.root;os.environ['GRAIL_WORKER_THREADS']='1'
    os.environ['UMR_SOMA_ASSETS_PATH']=str(root/'source/soma_assets');os.environ['UMR_SOMA_DEVICE']='cpu'
    if a.prepare_robot:
        plan=read(root/'source/inventory.json');prepare(root,plan,a.prepare_robot);return
    if a.prepare:
        with (Path(os.environ.get('UMR_LOCK_DIR', '/tmp'))/'.bones-prepare.lock').open('w') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            plan=initialize(root,a.original,a.supermimic)
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda r:prepare(root,plan,r),ROBOTS))
        return
    if not a.node:p.error('--node required')
    with (Path(os.environ.get('UMR_LOCK_DIR', '/tmp'))/f'.bones-{a.node}.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=root/'umr/pipeline'/a.node/'waiting.json'
        while True:
            grail=root.parent/'grail/umr/pipeline'/a.node/'retarget.json'
            inv=root/'source/inventory.json'
            if grail.exists() and read(grail).get('status')=='complete' and inv.exists():
                plan=read(inv)
                if all((root/'umr'/r/'ready.json').exists() and read(root/'umr'/r/'ready.json').get('plan_id')==plan['plan_id'] for r in ROBOTS):break
            write(state,dict(status='waiting_for_grail_and_bones_preparation',updated_at=time.time()));time.sleep(30)
        tasks=[(r,s) for i,(r,s) in enumerate((r,s) for s in plan['sequences'] for r in ROBOTS) if i%len(NODES)==NODES.index(a.node)]
        execute_phase(root,plan,a.node,'retarget',tasks,a.workers,lambda t:process(root,plan,t,a.supermimic))


if __name__=='__main__':main()
