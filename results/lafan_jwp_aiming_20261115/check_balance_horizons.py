"""Read-only longer-horizon dynamics check of the exact pre-actor balance checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_fullbody_validation import validate_world_model
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.wide_curriculum import sample
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(731)
    digest=sha256(a.checkpoint);ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    task=WideAimingTask(a.device,ck['config']);validate_checkpoint(ck,task)
    pi=FullbodyPolicy(task.robot).to(a.device);pi.load_state_dict(ck['pi']);pi.eval()
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(a.device);wm.load_state_dict(ck['wm']);wm.eval()
    freeze_normalizers(pi,wm)
    cfg=dict(ck['config'],horizon=32,validation_horizons=[1,8,16,32],wm_validation_envs=32,
             wm_validation_steps=256,wm_gate_root_m=.05,wm_gate_feet_m=.025,wm_gate_tip_m=.1)
    report=validate_world_model(pi,wm,task,lambda n:sample(0,n,a.device),cfg,a.out)
    assert sha256(a.checkpoint)==digest
    report.update(checkpoint_sha256=digest,scope=__doc__,weights_changed=False,policy_acceptance=False)
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
