#!/usr/bin/env python3
"""Run the upstream GRAIL HSI/HOI adapter, isolated by robot and sequence.

One worker per embodiment. Scene conversion is shared and locked. Source files
stay in HF_HOME; each robot trains its own GRAIL-overlay correspondence once.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
REPO='nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL'
ROBOTS={'g1':'unitree_g1','h2':'unitree_h2','a3':'agibot_a3'}


def atomic_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def run(command,log,env=None):
    with log.open('a') as f:
        f.write('\nCOMMAND '+repr([str(x) for x in command])+'\n');f.flush()
        subprocess.run([str(x) for x in command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)


def stage_source(snapshot, category, output):
    # UMR identifies GRAIL by recon/<stem>.pkl; resolving HF symlinks would
    # otherwise turn a sequence into an extensionless content-addressed blob.
    dest=output/'source'/category;dest.mkdir(parents=True,exist_ok=True)
    for name in ['recon','object_usd']:
        target=dest/name
        if target.is_symlink():target.unlink()
        target.mkdir(exist_ok=True)
        for source in (snapshot/'data'/category/name).rglob('*'):
            if not source.is_file():continue
            path=target/source.relative_to(snapshot/'data'/category/name)
            if path.exists():continue
            path.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(source.resolve(),path)
            except OSError:shutil.copyfile(source,path)
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--category',action='append')
    p.add_argument('--revision',help='Pinned HF dataset revision; reuse across pods')
    p.add_argument('--smplx-preprocessor',type=Path,required=True,help='SuperMimic supermimic/resample_smplx.py; 30 Hz before UMR')
    p.add_argument('--gpus',default='0,1,2,3',help='Four GPU ids for first correspondence jobs; none for CPU-only pods')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0,help='Debug only; default entire category')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if not 0<=a.shard_index<a.num_shards:raise ValueError('Invalid shard')
    gpu_ids=a.gpus.split(',') if a.gpus!='none' else []
    if gpu_ids and len(gpu_ids)!=4:raise ValueError('Provide four GPU ids or none')
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    api=HfApi();revision=a.revision or api.dataset_info(REPO).sha
    categories=a.category or ['slope','curb','stair_p1','stair_p2','sitting','pickup_table','pickup_ground']
    queues={r:queue.Queue() for r in ROBOTS};lock=threading.Lock()
    status={'source':REPO,'revision':revision,'workers':4,'robots':{r:{'completed':0,'failed':0,'status':'waiting_for_data','current':'','error':''} for r in ROBOTS},'categories':{c:{'status':'pending','motions':0} for c in categories},'started_at':time.time(),'status':'running'}
    def update():
        with lock:atomic_json(a.output/f'status_shard{a.shard_index}.json',status)
    def worker(robot):
        output=a.output/'umr'/robot;output.mkdir(parents=True,exist_ok=True)
        update()
        cfg=json.loads((ROOT/f'robot_configs/humanoid_retarget_{ROBOTS[robot]}_example.json').read_text())
        xml=cfg['robot'].get('xml')
        cfg['robot']['xml']=str((ROOT/'robot_configs'/xml).resolve()) if xml else str(a.assets/robot/'retarget.floating_mjcf.xml')
        cfg['smplx_model_dir']=str(ROOT/'sample_data/grail/g1_smplx_model')
        cfg['hsi_hoi']={'object':{'grail_mjcf_python':sys.executable}}
        cfg['view']={'enabled':False}
        cfg.setdefault('correspondence',{}).setdefault('train',{})['device']='cuda' if gpu_ids else 'cpu'
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        gpu=gpu_ids[list(ROBOTS).index(robot)] if gpu_ids else ''
        config=output/'robot.json';shared=output/'correspondence_slots_final.npz'
        consecutive=0
        while True:
            task=queues[robot].get()
            if task is None:break
            category,source,key=task
            job=output/category/key;job.mkdir(parents=True,exist_ok=True);result=job/'retarget.npz'
            manifest=job/'status.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('status')=='completed' and result.is_file():
                status['robots'][robot]['completed']+=1;update();continue
            status['robots'][robot].update(status='running',current=f'{category}/{key}');update()
            atomic_json(manifest,{'status':'running','robot':robot,'source_revision':revision,'category':category,'sequence':key})
            try:
                if shutil.disk_usage(a.output).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; batch paused')
                # Normalize the human and object together before scene conversion.
                raw_source=source
                source=a.output/'smplx'/category
                source.mkdir(parents=True,exist_ok=True)
                (source/'object_usd').mkdir(exist_ok=True)  # Official GRAIL format marker.
                # CoACD and the source scene are identical for all embodiments.
                locks=a.output/'umr'/'scene_locks'/category;locks.mkdir(parents=True,exist_ok=True)
                with (locks/(key+'.lock')).open('w') as guard:
                    fcntl.flock(guard,fcntl.LOCK_EX)
                    run([sys.executable,a.smplx_preprocessor,raw_source/'recon'/(key+'.pkl'),source/'recon'/(key+'.pkl')],job/'resample.log',dict(env,CUDA_VISIBLE_DEVICES=''))
                    scene_root=a.output/'scenes'/category
                    if not (scene_root/key/'metadata.json').is_file():
                        run([sys.executable,ROOT/'scripts/prepare_grail_object_mjcf.py','--data-root',raw_source,'--output-dir',scene_root,'--seq-key',key,'--fail-fast'],job/'scene.log',dict(env,CUDA_VISIBLE_DEVICES=''))
                    # Geometry is source data; object poses follow normalized SMPL-X.
                    import pickle, numpy as np
                    from scipy.spatial.transform import Rotation
                    with (source/'recon'/(key+'.pkl')).open('rb') as handle:
                        obj=pickle.load(handle)['obj_data']
                    poses=np.c_[obj['obj_t'],Rotation.from_matrix(obj['obj_R']).as_quat()]
                    np.savetxt(scene_root/key/'prop_grail_object.csv',poses,delimiter=',',header='px,py,pz,qx,qy,qz,qw',comments='')
                cfg['hsi_hoi']['object']['grail_mjcf_dir']=str(scene_root)
                cfg.setdefault('correspondence',{})
                if shared.is_file():cfg['correspondence']['slots']=str(shared)
                atomic_json(config,cfg)
                run([sys.executable,ROOT/'scripts/humanoid_retarget_pipeline_hsi_hoi.py','--config',config,'--defaults',ROOT/'humanoid_retarget_defaults_hsi_hoi_grail.json','--stage','retarget','--data',source,'--seq-key',key,'--out',result,'--work-dir',job/'work','--stream-chunk-frames','64','--skip-view'],job/'umr.log',dict(env,CUDA_VISIBLE_DEVICES=gpu if not shared.is_file() else ''))
                import numpy as np
                with np.load(result,allow_pickle=True) as data:
                    qpos=data['qpos'];frames=len(qpos)
                    if frames==0 or not np.isfinite(qpos).all():raise ValueError('Invalid retarget qpos')
                    fps=float(data['fps'].item())
                    if fps!=30:raise ValueError('Expected 30 Hz SMPL-X retarget')
                if not shared.is_file():
                    runtime=list((job/'work/configs').glob('*runtime_config.json'))
                    if len(runtime)!=1:raise RuntimeError('Ambiguous runtime config')
                    sys.path.insert(0,str(ROOT/'scripts'))
                    from humanoid_retarget_config import load_config
                    from humanoid_retarget_pipeline import slots_out
                    trained=slots_out(load_config(runtime[0]));shutil.copyfile(trained,shared)
                atomic_json(manifest,{'status':'completed','robot':robot,'source_revision':revision,'category':category,'sequence':key,'frames':frames,'fps':fps,'visual_qc':'pending','result':str(result)})
                status['robots'][robot]['completed']+=1;consecutive=0
                print(f'[{robot}] completed {category}/{key} frames={frames}',flush=True)
            except Exception as exc:
                atomic_json(manifest,{'status':'failed','robot':robot,'category':category,'sequence':key,'error':str(exc)})
                status['robots'][robot]['failed']+=1;consecutive+=1
                print(f'[{robot}] FAILED {category}/{key}: {exc}',flush=True)
                if not status['robots'][robot]['completed'] or consecutive>=3:
                    status['robots'][robot].update(status='blocked',error=str(exc));update();return
            update()
        status['robots'][robot]['status']='complete';update()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,r) for r in ROBOTS]
        try:
            for category in categories:
                status['categories'][category]={'status':'downloading'};update()
                first=None
                if category==categories[0] and a.shard_index==0:
                    first=next(x.path for x in api.list_repo_tree(REPO,path_in_repo=f'data/{category}/recon',repo_type='dataset',revision=revision) if x.path.endswith('.pkl'))
                    key=Path(first).stem
                    initial=Path(hf_hub_download(REPO,filename=first,repo_type='dataset',revision=revision)).parents[3]
                    hf_hub_download(REPO,filename=f'data/{category}/object_usd/{key}.usd',repo_type='dataset',revision=revision)
                    dest=stage_source(initial,category,a.output)
                    for robot in ROBOTS:queues[robot].put((category,dest,key))
                    print(f'[data] bootstrap {category}/{key}',flush=True)
                snap=Path(snapshot_download(REPO,repo_type='dataset',revision=revision,allow_patterns=[f'data/{category}/recon/*',f'data/{category}/object_usd/*'],max_workers=4))
                dest=stage_source(snap,category,a.output)
                paths=sorted((dest/'recon').glob('*.pkl'))
                if a.limit:paths=paths[:a.limit]
                if not paths:raise RuntimeError(f'No recon sequences: {category}')
                missing=[x.stem for x in paths if not (dest/'object_usd'/(x.stem+'.usd')).is_file()]
                if missing:raise RuntimeError(f'Missing USD companions in {category}: {missing[:10]}')
                status['categories'][category]={'status':'queued','motions':len(paths)};update()
                for index,path in enumerate(paths):
                    if index % a.num_shards != a.shard_index or (first is not None and path.stem==Path(first).stem):continue
                    for robot in ROBOTS:queues[robot].put((category,dest,path.stem))
                print(f'[data] queued {category}: {len(paths)} motions x 4 robots',flush=True)
        finally:
            for q in queues.values():q.put(None)
        for f in futures:f.result()
    status['status']='complete' if all(x['status']=='complete' and not x['failed'] for x in status['robots'].values()) else 'attention_required';update()


if __name__=='__main__':main()
