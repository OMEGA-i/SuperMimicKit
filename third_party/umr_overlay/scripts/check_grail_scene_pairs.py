"""Short CPU-only real-data pairing check, separate from production outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from grail_cluster import ROBOTS, ROOT, read, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--robot',choices=ROBOTS,required=True)
    args=parser.parse_args();root=args.root;robot=args.robot
    key='pickup_table__beer_8__002';category='pickup_table'
    job=root/'umr/pair_validation'/robot;job.mkdir(parents=True,exist_ok=True)
    cfg_path=ROOT/f'robot_configs/humanoid_retarget_{ROBOTS[robot]}_example.json'
    cfg=read(cfg_path);xml=cfg['robot'].get('xml')
    cfg['robot']['xml']=str((cfg_path.parent/xml).resolve()) if xml else str(root/'umr/assets'/robot/'retarget.floating_mjcf.xml')
    cfg['smplx_model_dir']=str(ROOT/'sample_data/grail/g1_smplx_model')
    cfg['view']={'enabled':False}
    cfg['hsi_hoi']={'object':{'grail_mjcf_dir':str(root/'scenes'/category),'grail_auto_prepare_mjcf':False}}
    cfg.setdefault('correspondence',{})['slots']=str(root/'umr'/robot/'correspondence_slots_final.npz')
    cfg['correspondence'].setdefault('train',{})['device']='cpu'
    config=job/'robot.json';write(config,cfg)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (job/'retarget.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/humanoid_retarget_pipeline_hsi_hoi.py'),
            '--config',str(config),'--defaults',str(ROOT/'humanoid_retarget_defaults_hsi_hoi_grail.json'),
            '--stage','retarget','--data',str(root/'smplx'/category),'--seq-key',key,
            '--out',str(job/'retarget.npz'),'--work-dir',str(job/'work'),
            '--max-frames','8','--stream-chunk-frames','8','--skip-view'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'robot':robot,'pair':str(job/'scene_pair.json')}),flush=True)

if __name__=='__main__':main()
