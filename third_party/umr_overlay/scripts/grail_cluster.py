#!/usr/bin/env python3
"""Seven-pod GRAIL pipeline with global download/SMPL-X/retarget barriers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPO = 'nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL'
CATEGORIES = ['slope','curb','stair_p1','stair_p2','sitting','pickup_table','pickup_ground']
ROBOTS = {'g1':'unitree_g1','a3':'agibot_a3','h2':'unitree_h2'}
NODES = json.loads(os.environ.get('UMR_CLUSTER_NODES', '{"local": 1}'))


def read(p): return json.loads(p.read_text())


def scene_name(key):
    # Match UMR's safe_name used by both scene conversion and retarget loading.
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(key)).strip('_') or 'sequence'
    return name if len(name) <= 180 else f'{name[:160]}_{hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]}'


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=p.parent, delete=False) as f:
        json.dump(value,f,indent=2);f.write('\n')
    os.replace(f.name,p)


def assigned(index,node,nodes):
    names=list(nodes); start=sum(nodes[n] for n in names[:names.index(node)])
    return start <= index % sum(nodes.values()) < start+nodes[node]


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:return False
    kind = 'sha256' if entry.get('sha256') else 'sha1'
    expected = entry.get('sha256') or entry.get('blob_id')
    if not expected:raise ValueError(f'Missing checksum: {entry["path"]}')
    h=hashlib.new(kind)
    if kind=='sha1':h.update(f'blob {entry["size"]}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()==expected


def initialize(root,revision):
    target=root/'source/inventory.json'
    if target.exists():
        p=read(target)
        if p['revision']!=revision or p['nodes']!=NODES:raise ValueError('Existing inventory differs')
        print('Reusing inventory',len(p['files']),len(p['sequences']),flush=True);return
    from huggingface_hub import HfApi
    files=[]
    api=HfApi()
    for category in CATEGORIES:
        for sub in ['recon','object_usd']:
            for entry in api.list_repo_tree(REPO,repo_type='dataset',revision=revision,
                                           path_in_repo=f'data/{category}/{sub}',recursive=True):
                if not hasattr(entry,'size'):continue
                lfs=getattr(entry,'lfs',None)
                sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
                files.append(dict(path=entry.path,size=entry.size,sha256=sha,blob_id=entry.blob_id))
        print('Indexed',category,len(files),flush=True)
    files.sort(key=lambda x:x['path'])
    paths={f['path'] for f in files}
    seq=sorted(f['path'].removeprefix('data/').replace('/recon/','/').removesuffix('.pkl')
               for f in files if '/recon/' in f['path'] and f['path'].endswith('.pkl'))
    for motion in seq:
        cat,key=motion.split('/',1)
        if f'data/{cat}/object_usd/{key}.usd' not in paths:raise ValueError(f'Missing USD: {motion}')
    if not seq:raise ValueError('Empty inventory')
    plan=dict(schema_version=1,source=REPO,revision=revision,nodes=NODES,
              threads_per_worker=2,download_workers=4,files=files,sequences=seq)
    plan['plan_id']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    write(target,plan)
    print('Inventory ready',len(files),'files',len(seq),'motions',sum(x['size'] for x in files),'bytes',flush=True)


def download(root,plan,entry,temp_suffix='.part'):
    import requests
    dst=root/'source'/entry['path'].removeprefix('data/')
    if verified(dst,entry):return
    dst.parent.mkdir(parents=True,exist_ok=True)
    cache=Path(os.environ['HF_HOME'])/'hub'/('datasets--'+REPO.replace('/','--'))/'snapshots'/plan['revision']/entry['path']
    if verified(cache,entry):
        temp=dst.with_suffix(dst.suffix+temp_suffix)
        if temp.exists():temp.unlink()
        try:os.link(cache.resolve(),temp)
        except OSError:shutil.copyfile(cache,temp)
        os.replace(temp,dst);return
    temp=dst.with_suffix(dst.suffix+temp_suffix)
    error=None
    for attempt in range(6):
        try:
            if verified(dst,entry):
                temp.unlink(missing_ok=True)
                return
            offset=temp.stat().st_size if temp.exists() else 0
            if offset>=entry['size']:temp.unlink();offset=0
            # Independent pod connections; alternate configured proxy and direct
            # HTTPS routes. Direct HTTP avoids the failing Xet/CAS reconstruction.
            session=requests.Session();session.trust_env=(attempt%2==0)
            host='https://huggingface.co' if attempt<4 else 'https://hf.co'
            url=f'{host}/datasets/{REPO}/resolve/{plan["revision"]}/{quote(entry["path"],safe="/")}?download=true&retry={int(time.time())}'
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with session.get(url,headers=headers,stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                if response.status_code==206:
                    if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Invalid resume range')
                    mode='ab'
                else:mode='wb'
                with temp.open(mode) as f:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:f.write(chunk)
            session.close()
            if not verified(temp,entry):
                temp.unlink(missing_ok=True);raise ValueError('Downloaded checksum mismatch')
            os.replace(temp,dst);return
        except Exception as exc:
            # Do not persist CDN signed URLs in status logs.
            error=f'{type(exc).__name__}: download failed for {entry["path"]}'
            time.sleep(min(2**attempt,30))
    raise RuntimeError(error)


def run(cmd,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    threads=os.environ.get('GRAIL_WORKER_THREADS','2')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,
             OPENBLAS_NUM_THREADS=threads,NUMEXPR_NUM_THREADS=threads)
    with log.open('a') as f:
        subprocess.run([str(x) for x in cmd],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def prepare(root,plan,motion,preprocessor):
    cat,key=motion.split('/',1); target=root/'smplx'/cat
    from link_grail_usd_sources import link
    link(root,cat,key)
    scene=root/'scenes'/cat/scene_name(key); marker=scene/'pipeline.json'
    old=read(marker) if marker.exists() else {}
    if old.get('plan_id')==plan['plan_id'] and old.get('asset_version')==4:return
    target.mkdir(parents=True,exist_ok=True);(target/'object_usd').mkdir(exist_ok=True)
    if old.get('plan_id')!=plan['plan_id'] or not (target/'recon'/(key+'.pkl')).exists():
        run([sys.executable,preprocessor,root/'source'/cat/'recon'/(key+'.pkl'),target/'recon'/(key+'.pkl')],target/'logs'/(key+'.log'))
    run([sys.executable,ROOT/'scripts/prepare_grail_object_mjcf.py','--data-root',root/'source'/cat,
         '--output-dir',root/'scenes'/cat,'--seq-key',key,'--fail-fast'],target/'logs'/(key+'.scene.log'))
    import numpy as np
    from scipy.spatial.transform import Rotation
    with (target/'recon'/(key+'.pkl')).open('rb') as f:data=pickle.load(f)
    assert data['human_data']['mocap_frame_rate']==30
    obj=data['obj_data']; poses=np.c_[obj['obj_t'],Rotation.from_matrix(obj['obj_R']).as_quat()]
    if len(poses)!=len(data['human_data']['poses']):raise ValueError('Human/object frame mismatch')
    if not np.isfinite(poses).all():raise ValueError('Nonfinite object poses')
    np.savetxt(scene/'prop_grail_object.csv',poses,delimiter=',',header='px,py,pz,qx,qy,qz,qw',comments='')
    meta=read(scene/'metadata.json');meta['frames']=len(poses);meta['fps']=30;write(scene/'metadata.json',meta)
    write(marker,dict(plan_id=plan['plan_id'],status='complete',fps=30,frames=len(poses),asset_version=4))


def retarget(root,plan,item):
    robot,motion=item;cat,key=motion.split('/',1)
    job=root/'umr'/robot/cat/key;job.mkdir(parents=True,exist_ok=True)
    result=job/'retarget.npz';marker=job/'status.json'
    if marker.exists() and result.exists():
        d=read(marker)
        if d.get('status')=='completed' and d.get('fps')==30 and d.get('source_revision')==plan['revision'] and d.get('asset_version')==4 and (job/'scene_pair.json').is_file():return
    write(marker,dict(status='running',robot=robot,category=cat,sequence=key,source_revision=plan['revision']))
    cfg_path=ROOT/f'robot_configs/humanoid_retarget_{ROBOTS[robot]}_example.json'
    cfg=read(cfg_path);xml=cfg['robot'].get('xml')
    cfg['robot']['xml']=str(Path(os.environ.get('SUPERMIMIC_ROBOT_ASSETS', str(ROOT.parents[1]/'assets/robots')))/robot/'robot.xml')
    cfg['smplx_model_dir']=str(ROOT/'sample_data/grail/g1_smplx_model')
    cfg['view']={'enabled':False}
    cfg['hsi_hoi']={'object':{'grail_mjcf_dir':str(root/'scenes'/cat),'grail_mjcf_python':sys.executable,'grail_auto_prepare_mjcf':False}}
    slots=root/'umr'/robot/'correspondence_slots_final.npz'
    if not slots.is_file():raise FileNotFoundError(slots)
    cfg.setdefault('correspondence',{})['slots']=str(slots)
    cfg['correspondence'].setdefault('train',{})['device']='cpu'
    # Job-local config: multiple workers for one embodiment never overwrite it.
    config=job/'robot.json';write(config,cfg)
    try:
        run([sys.executable,ROOT/'scripts/humanoid_retarget_pipeline_hsi_hoi.py','--config',config,
             '--defaults',ROOT/'humanoid_retarget_defaults_hsi_hoi_grail.json','--stage','retarget',
             '--data',root/'smplx'/cat,'--seq-key',key,'--out',result,'--work-dir',job/'work',
             '--stream-chunk-frames','64','--skip-view'],job/'umr.log')
        import numpy as np
        with np.load(result,allow_pickle=True) as d:
            frames=len(d['qpos']);fps=float(d['fps'].item())
            if frames<2 or fps!=30 or not np.isfinite(d['qpos']).all():raise ValueError('Invalid UMR result')
        if not (job/'scene_pair.json').is_file():raise ValueError('Missing paired scene')
        write(marker,dict(status='completed',robot=robot,category=cat,sequence=key,
                          asset_version=4,
                          source_revision=plan['revision'],frames=frames,fps=fps,
                          result=str(result),visual_qc='pending'))
        shutil.rmtree(job/'work',ignore_errors=True)
    except Exception as exc:
        write(marker,dict(status='failed',robot=robot,category=cat,sequence=key,error=str(exc),source_revision=plan['revision']))
        raise


def barrier_ready(root,plan,phase):
    for node in plan['nodes']:
        file=root/'umr/pipeline'/node/(phase+'.json')
        if not file.exists():return False
        data=read(file)
        if data.get('plan_id')!=plan['plan_id'] or data.get('status')!='complete':return False
        if phase=='smplx' and data.get('asset_version')!=4:return False
    return True


def execute_phase(root,plan,node,phase,tasks,workers,operation):
    status=dict(plan_id=plan['plan_id'],node=node,phase=phase,status='running',
                workers=workers,total=len(tasks),completed=0,failed=0,errors=[],pid=os.getpid(),host=socket.gethostname(),asset_version=4)
    path=root/'umr/pipeline'/node/(phase+'.json');guard=threading.Lock();stop=threading.Event()
    def update():
        with guard:status['updated_at']=time.time();write(path,status)
    def heartbeat():
        while not stop.wait(10):update()
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();update()
    pending=tasks
    try:
        while pending:
            failed=[]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(operation,t):t for t in pending}
                for f in as_completed(futures):
                    try:f.result()
                    except Exception as exc:
                        failed.append(futures[f])
                        with guard:
                            status['failed']+=1;status['errors']=(status['errors']+[str(exc)[-500:]])[-10:]
                    else:
                        with guard:status['completed']+=1
                    update()
            pending=failed
            if pending:
                with guard:status['status']='retrying'
                update();time.sleep(60)
                with guard:status['status']='running'
        with guard:status['status']='complete';status['remaining']=0
        update()
    except BaseException:
        with guard:status['status']='interrupted'
        update();raise
    finally:stop.set();thread.join(timeout=1)


def worker(root,plan,node,preprocessor,workers=None):
    import fcntl
    # Locks are local to each pod, avoiding uncertain cross-pod FUSE flock semantics.
    local_lock=Path(os.environ.get('UMR_LOCK_DIR', '/tmp'))/f'.grail-{node}.lock'
    with local_lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        names=list(plan['nodes']);index=names.index(node);count=len(names)
        tasks=[f for i,f in enumerate(plan['files']) if i%count==index]
        previous=root/'umr/pipeline'/node/'download.json'
        done=read(previous) if previous.exists() else {}
        if done.get('plan_id')!=plan['plan_id'] or done.get('status')!='complete':
            execute_phase(root,plan,node,'download',tasks,plan['download_workers'],lambda t:download(root,plan,t))
        while not barrier_ready(root,plan,'download'):time.sleep(10)
        tasks=[s for i,s in enumerate(plan['sequences']) if assigned(i,node,plan['nodes'])]
        execute_phase(root,plan,node,'smplx',tasks,plan['nodes'][node],lambda t:prepare(root,plan,t,preprocessor))
        while not barrier_ready(root,plan,'smplx'):time.sleep(10)
        tasks=[(r,s) for i,(r,s) in enumerate((r,s) for s in plan['sequences'] for r in ROBOTS)
               if assigned(i,node,plan['nodes'])]
        execute_phase(root,plan,node,'retarget',tasks,workers or plan['nodes'][node],lambda t:retarget(root,plan,t))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--revision',default='40e795761302e611c1e7e3a6caefdd010d56c199')
    p.add_argument('--init',action='store_true')
    p.add_argument('--node',choices=list(NODES))
    p.add_argument('--assist-download',choices=list(NODES),help='Help download this node shard, without changing stage ownership')
    p.add_argument('--preprocessor',type=Path,required=True,help='Trusted GRAIL human/object resampler (input.pkl output.pkl), producing synchronized 30 Hz data')
    p.add_argument('--workers',type=int,help='Retarget concurrency only; preserves shard ownership')
    p.add_argument('--worker-threads',type=int,default=2)
    a=p.parse_args();root=a.root.resolve()
    if a.worker_threads<1 or (a.workers is not None and a.workers<1):p.error('Worker counts must be positive')
    os.environ['GRAIL_WORKER_THREADS']=str(a.worker_threads)
    if a.init:initialize(root,a.revision);return
    if not a.node:p.error('--node required for worker')
    plan=read(root/'source/inventory.json')
    if a.assist_download:
        import fcntl
        helpers=[n for n in plan['nodes'] if n!=a.assist_download]
        if a.node not in helpers:p.error('Helper must differ from download owner')
        with (Path(os.environ.get('UMR_LOCK_DIR', '/tmp'))/f'.grail-assist-{a.node}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            names=list(plan['nodes']);owner=names.index(a.assist_download)
            tasks=[f for i,f in enumerate(plan['files']) if i%len(names)==owner]
            tasks=tasks[helpers.index(a.node)::len(helpers)]
            execute_phase(root,plan,a.node,'download_assist',tasks,plan['download_workers'],
                          lambda t:download(root,plan,t,f'.assist-{a.node}.part'))
        return
    worker(root,plan,a.node,a.preprocessor,a.workers)


if __name__=='__main__':main()
