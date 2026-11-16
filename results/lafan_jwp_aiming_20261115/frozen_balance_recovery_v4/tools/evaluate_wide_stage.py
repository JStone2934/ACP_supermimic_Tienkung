"""Evaluate one training stage; this never replaces full final acceptance."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.wide_curriculum import STAGES
from tools.evaluate_wide_aiming import evaluate


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--level',required=True,type=int,choices=range(len(STAGES)))
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    if ck.get('smoke') or ck['config'].get('curriculum_level',0)!=args.level:
        raise ValueError('Stage differs from checkpoint, or checkpoint is smoke-only')
    task=WideAimingTask(args.device,ck['config']);validate_checkpoint(ck,task)
    pi=FullbodyPolicy(task.robot).to(args.device);pi.load_state_dict(ck['pi'])
    report=evaluate(pi,task,args.out,seed=args.seed,checkpoint=args.checkpoint,stage=args.level)
    print(json.dumps({k:report[k] for k in ('level','passed','random_success_rate')}))


if __name__=='__main__':main()
