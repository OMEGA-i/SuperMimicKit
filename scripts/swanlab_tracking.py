"""Independent live training logger and per-embodiment held-out evaluator."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2)); temp.replace(path)


def initialize(cfg, role):
    import swanlab
    os.environ['SWANLAB_API_KEY'] = Path(cfg['key_file']).read_text().strip()
    run = swanlab.init(project=cfg.get('project', 'SuperMimicKit'), workspace=cfg.get('workspace'),
        experiment_name=cfg['experiment_name'], id=cfg['experiment_id'],
        parallel='shared', mode='online',
        logdir=str(Path(cfg['state_dir'])/'swanlog'/role),
        config={'embodiment':cfg['robot'], 'train_motions':40,'val_motions':37,
                'val_every_iterations':500,'eval_protocol':'full clip from frame0; terminal remainder zero; equal-weight motions',
                'training_run':cfg['training_run']})
    for group in ['train','train40','val37']:
        swanlab.define_metric(group+'/*',x_axis=group+'/iteration')
    write(Path(cfg['state_dir'])/(role+'_run.json'),{'id':run.id,'url':getattr(run,'url',None),'role':role})
    return swanlab


def train(cfg):
    swan = initialize(cfg,'train')
    cursor = Path(cfg['state_dir'])/'train_cursor.json'
    state = json.loads(cursor.read_text()) if cursor.exists() else {'offset':0,'iteration':-1}
    path = Path(cfg['training_run'])/'metrics.jsonl'
    previous = None
    while True:
        with path.open() as f:
            f.seek(state['offset'])
            while True:
                line=f.readline()
                if not line or not line.endswith('\n'):break
                row=json.loads(line);n=int(row['iteration'])+1
                if n>state['iteration']:
                    values={'train/'+k:v for k,v in row.items() if isinstance(v,(int,float)) and k!='iteration'}
                    values['train/iteration']=n
                    # Explicitly distinguish single-step reward from full-clip return.
                    values['train/step_reward_normalized']=row['sim_reward']/.1
                    if previous and row['elapsed_s']>previous['elapsed_s']:
                        values['train/iterations_per_second']=(row['iteration']-previous['iteration'])/(row['elapsed_s']-previous['elapsed_s'])
                    swan.log(values,step=n);previous=row
                    state['iteration']=n
                state['offset']=f.tell()
            write(cursor,state)
        time.sleep(5)


def split_metrics(result, manifest):
    out={}
    for group,key,count in [('train40','train',40),('val37','test',37)]:
        wanted=set(manifest['splits'][key]);rows=[r for r in result['motions'] if r['motion'] in wanted]
        if len(rows)!=count:raise ValueError(f'{group}: expected {count}, got {len(rows)}')
        out[group]={'iteration':result['checkpoint_iterations'],
            'normalized_return':sum(r['normalized_return'] for r in rows)/count,
            'failed_motions':sum(r['failed'] for r in rows),'motion_count':count}
    return out


def checkpoint_asset_env(cfg, asset_hash):
    """Keep historical checkpoints on their original physics after asset migration."""
    versions = cfg.get('asset_versions', {})
    if asset_hash in versions:
        return versions[asset_hash]
    if asset_hash == cfg['source_asset_sha256']:
        return cfg['asset_env']
    raise ValueError('Evaluation model differs from training checkpoint')


def evaluate(cfg):
    import hashlib
    import torch
    swan=initialize(cfg,'eval')
    base=Path(cfg['state_dir']);queue=Path(cfg['queue']);queue.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(cfg['manifest']).read_text())
    while True:
        for ckpt in sorted(queue.glob('eval_*.pt'),key=lambda p:int(p.stem.split('_')[1])):
            n=int(ckpt.stem.split('_')[1])
            if n < cfg['start_iteration'] or n % 500:continue
            out=base/'results'/str(n)
            if (out/'uploaded.json').exists():continue
            result_path=out/'results.json'
            if not result_path.exists():
                d=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
                if d['iteration']+1!=n:raise ValueError('Checkpoint iteration mismatch')
                asset_hash=d['training_contract']['asset_sha256']
                asset_env=checkpoint_asset_env(cfg,asset_hash)
                asset_path=Path(asset_env[d['config']['asset_env']])
                from supermimic.asset_identity import verify_asset
                verify_asset(asset_path, asset_hash)
                del d
                env=dict(os.environ,**asset_env,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=str(cfg['gpu']))
                # Interrupted attempts retain diagnostics and retry in a new directory.
                if out.exists():out.rename(out.with_name(out.name+'.interrupted-'+str(time.time_ns())))
                log=base/'logs'/f'{n}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('w') as f:
                    subprocess.run([sys.executable,'-u','-m','scripts.evaluate_full_motions',
                        '--checkpoint',str(ckpt),'--motions',cfg['motions'],'--manifest',cfg['manifest'],
                        '--motion-split','all77','--device','cuda:0','--sleep','0','--output',str(out)],
                        env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            result=json.loads(result_path.read_text());groups=split_metrics(result,manifest)
            for group,values in groups.items():swan.log({group+'/'+k:v for k,v in values.items()},step=n)
            write(out/'uploaded.json',{'iteration':n,'metrics':groups,'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()})
            print(json.dumps({'event':'evaluated','robot':cfg['robot'],'iteration':n,'metrics':groups}),flush=True)
        time.sleep(5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--role',choices=['train','eval'],required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());state=Path(cfg['state_dir']);state.mkdir(parents=True,exist_ok=True)
    with (state/(a.role+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                (train if a.role=='train' else evaluate)(cfg)
            except Exception as e:
                # Credentials never appear in error text we persist.
                write(state/(a.role+'_error.json'),{'type':type(e).__name__,'time':time.time()})
                print(json.dumps({'event':'retry','type':type(e).__name__,'role':a.role}),flush=True)
                try:
                    import swanlab;swanlab.finish()
                except Exception:pass
                time.sleep(30)

if __name__=='__main__':main()
