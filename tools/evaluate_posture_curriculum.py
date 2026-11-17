"""Original physical task acceptance with an explicit reset condition."""
import argparse,json
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import json_save
from tools.evaluate_wide_aiming import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reset-pose',choices=('old','reference'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--final',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if ck.get('smoke'):raise ValueError('Smoke cannot certify a stage')
    task=WideAimingTask(a.device,ck['config']);validate_checkpoint(ck,task)
    task.evaluation_reset_pose=a.reset_pose
    pi=FullbodyPolicy(task.robot).to(a.device);pi.load_state_dict(ck['pi'])
    report=evaluate(pi,task,a.out,seed=a.seed,checkpoint=a.checkpoint,
                    stage=None if a.final else ck['config']['curriculum_level'])
    report['reset_pose_condition']=a.reset_pose
    json_save(a.out,report)
    print(json.dumps({k:report[k] for k in ('passed','level','random_success_rate','reset_pose_condition')}))


if __name__=='__main__':main()
