"""Keep normalized GRAIL recon files associated with their source USD assets."""
import argparse
from pathlib import Path


def link(root, category, key):
    source=root/'source'/category/'object_usd'/(key+'.usd')
    target=root/'smplx'/category/'object_usd'/(key+'.usd')
    if not source.is_file():raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        try:target.symlink_to(source)
        except FileExistsError:pass
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    count=0
    for recon in (a.root/'source').glob('*/object_usd/*.usd'):
        link(a.root,recon.parent.parent.name,recon.stem);count+=1
    print(f'Linked source USD references for {count} normalized sequences')

if __name__=='__main__':main()
