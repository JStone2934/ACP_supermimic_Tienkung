"""Fresh post-update check; the old pre-update gate is not reused."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'runner'))
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.wide_curriculum import sample
from supermimic.acp.aiming_fullbody_validation import validate_world_model
if __name__=='__main__':
 torch.set_num_threads(2);torch.manual_seed(2026);dev='cuda:0'
 ck=torch.load(P/'train/ckpt_000101.pt',map_location=dev,weights_only=False);cfg=ck['config']
 task=WideAimingTask(dev,cfg);pi=FullbodyPolicy(task.robot).to(dev);pi.load_state_dict(ck['pi']);wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(dev);wm.load_state_dict(ck['wm']);freeze_normalizers(pi,wm)
 validate_world_model(pi,wm,task,lambda n:sample(4,n,dev),cfg,P/'post_update_gate')
