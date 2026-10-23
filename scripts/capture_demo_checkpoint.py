"""Preserve policy + world model from an atomic checkpoint using CPU only."""
import argparse
import json
import os
from pathlib import Path
import torch


def capture(source, output):
    torch.set_num_threads(2)
    data=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    if data.get('learner_version')!=2 or not {'actor','world'}.issubset(data):
        raise ValueError('A complete v2 policy/world checkpoint is required')
    iteration=int(data['iteration'])+1
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/f'demo_{iteration}.pt'
    if target.exists():raise FileExistsError(target)
    saved={k:data[k] for k in ['config','joint_names','body_names','args','iteration','learner_version','training_contract']}
    for name in ['actor','world']:
        saved[name]={}
        for key,value in data[name].items():
            value=value.detach().cpu().clone()
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite {name}.{key}')
            saved[name][key]=value
    temp=target.with_suffix(f'.{os.getpid()}.tmp')
    try:
        torch.save(saved,temp)
        os.replace(temp,target)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({'iterations':iteration,'snapshot':str(target),'bytes':target.stat().st_size}),flush=True)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True,help='Trusted local full checkpoint')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();capture(a.checkpoint,a.output)
