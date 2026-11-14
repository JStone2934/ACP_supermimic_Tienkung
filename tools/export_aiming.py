"""Export just the shared actor and its runtime feature/action convention."""
import argparse
import torch
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    runtime={k:ck['config'][k] for k in ('version','terminal_speed')}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(pi=ck['pi'],config=runtime,task_config=ck['task_config'],stage='actor_export',source_stage=ck['stage'],report=ck.get('report')),out)

if __name__=='__main__':main()
