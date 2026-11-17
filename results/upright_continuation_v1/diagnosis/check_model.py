import sys
from pathlib import Path
P=Path('/home/js/robot/locomotion/standing-aiming/results/upright_continuation_v1');sys.path.insert(0,str(P/'runner'))
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.wide_curriculum import sample
from supermimic.acp.aiming_fullbody_validation import validate_world_model
iteration=int(sys.argv[1]);dev=sys.argv[2];torch.set_num_threads(2);torch.manual_seed(17)
c=torch.load(P/'train/ckpt_000100.pt',map_location=dev,weights_only=False);n=torch.load(P/f'train/ckpt_{iteration:06d}.pt',map_location=dev,weights_only=False)
cfg=dict(c['config'],wm_validation_steps=512);task=WideAimingTask(dev,cfg);pi=FullbodyPolicy(task.robot).to(dev);pi.load_state_dict(n['pi']);wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(dev);wm.load_state_dict(c['wm']);freeze_normalizers(pi,wm)
validate_world_model(pi,wm,task,lambda n:sample(4,n,dev),cfg,P/f'diagnosis/model_policy{iteration}')
