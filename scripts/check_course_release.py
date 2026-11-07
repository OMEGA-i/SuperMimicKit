"""Validate all 16 course bundles against their matching local robot XMLs."""
import argparse, hashlib, io, json, re, sys, tempfile, zipfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretrained/courses-20261107'))
from runtime.load import load

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundles',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--expected-count',type=int,default=16);a=p.parse_args()
 torch.set_num_threads(2); reports=[]
 for archive in sorted(a.bundles.glob('*.zip')):
  if archive.name=='inference-runtime.zip':continue
  with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as z:
   assert z.testzip() is None
   for n in z.namelist():
    assert not Path(n).is_absolute() and '..' not in Path(n).parts
    b=z.read(n)
    if n.endswith('.json') or n.endswith('.py'):
     assert not re.search(rb'/home/|/Users/|/opt/Omega|autel_|hf_[A-Za-z0-9]{20,}|glpat-|ghp_|ranger_r1',b),n
   for n in z.namelist():
    if n.endswith('.pt'):
     with zipfile.ZipFile(io.BytesIO(z.read(n))) as tensors:
      for member in tensors.namelist():
       if member.endswith('.pkl'):
        assert not re.search(rb'/home/|/Users/|autel_|ranger_r1|glpat-|ghp_', tensors.read(member)), member
   z.extractall(tmp)
   bundle=Path(tmp)/archive.stem
   m=json.loads((bundle/'metadata.json').read_text());assert m['robot'] in ('g1','h2','a3','sih2300')
   actor,world,m=load(bundle,a.assets/m['robot']/'robot.xml')
   assert all(t.dtype == torch.float32 for module in (actor, world) for t in module.state_dict().values() if t.is_floating_point())
   for seed in range(3):
    torch.manual_seed(seed)
    h=torch.randn(2,10,actor.layout.obs_dim)*.1;g=torch.randn(2,actor.layout.goal_dim)*.1
    with torch.no_grad():
     y=actor.decoder(torch.cat((torch.tanh(actor.encoder(actor.goal_norm(g))),actor.obs_norm(h).flatten(-2)),-1))
     d=world.dynamics(world.in_norm(torch.randn(2,world.in_norm.mean.numel())*.1))*world.out_norm.std()+world.out_norm.mean
    assert torch.isfinite(y).all() and torch.isfinite(d).all()
   reports.append({'bundle':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'strict_actor_world_load':True,'finite_forward_batches':3,'archive_scan':True})
   print(archive.name,'OK',flush=True)
 assert len(reports)==a.expected_count
 (a.bundles/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
