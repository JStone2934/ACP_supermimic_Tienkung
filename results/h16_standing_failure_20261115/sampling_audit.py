"""Collect one 4096-environment diagnostic batch, with no optimizer updates."""
import sys
from pathlib import Path
ROOT=Path('/home/js/robot/locomotion/standing-aiming')
RUN=ROOT/'logs/tienkung3_29dof/standing_aiming_fullbody_v2/dual_h16_critic4096_reward_v2_20261114'
sys.path.insert(0,str(RUN/'runner'))
import json
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,FullbodyTask
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_curriculum import Curriculum
from supermimic.acp.aiming_physical import execution_setup,rollout,to_device
from supermimic.acp.networks import Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
OUT=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.cuda.set_device('cuda:1')
import warp as wp
wp.config.quiet=True
ck=torch.load(OUT/'checkpoint.pt',map_location='cpu',weights_only=False)
cfg=ck['config'];execution_setup(cfg)
task=FullbodyTask('cuda:1',cfg);pi=FullbodyPolicy(task.robot).to('cuda:1');pi.load_state_dict(ck['pi'])
critic=Critic(task.critic_dim).to('cuda:1');critic.load_state_dict(ck['critic']);freeze_normalizers(pi,critic)
cur=Curriculum(task.alpha.device)
env=FullbodySim(task,4096,lambda n:cur.sample(n,task.weights['aim']),eval_mode=True)
with torch.no_grad():
    data,summary=rollout(pi,task,env,cfg,seed=cfg['seed']+10000+ck['physical_state']['collection_round'],keep_data=True,prefixes=True)
    data=to_device(data,'cuda:1')
    train=data['env_id']<3072;valid=~train
    pred=torch.cat([critic(x) for x in data['x'].split(1024)])
    pred=bound_bootstrap(pred,data['t'],task,cfg)
    lookup_mean=torch.zeros_like(pred);lookup_median=torch.zeros_like(pred)
    groups=[]
    for t in data['t'].unique(sorted=True):
        same=data['t']==t;ys=data['y'][same&train]
        lookup_mean[same]=ys.mean();lookup_median[same]=ys.median()
        groups.append(dict(step=int(t),train_count=len(ys),valid_count=int((same&valid).sum()),
            train_return_mean=float(ys.mean()),train_return_std=float(ys.std(unbiased=False)),
            previous_action_joint_std_max_rad=float((data['ap'][same&train]*task.alpha).std(0,unbiased=False).max())))
    mae=lambda x:float((x[valid]-data['y'][valid]).abs().mean())
    mse=lambda x:float((x[valid]-data['y'][valid]).square().mean())
    report=dict(checkpoint_iteration=ck['iteration'],environments=4096,train_environments=3072,validation_environments=1024,
        valid_rows=len(pred),train_rows=int(train.sum()),validation_rows=int(valid.sum()),
        physical={k:v for k,v in summary.items() if k!='per_environment'},
        critic_mae=mae(pred),time_only_mean_mae=mae(lookup_mean),time_only_median_mae=mae(lookup_median),
        critic_mse=mse(pred),time_only_mean_mse=mse(lookup_mean),
        validation_target_mean=float(data['y'][valid].mean()),validation_target_std=float(data['y'][valid].std()),
        time_only_explained_variance=1-mse(lookup_mean)/float(data['y'][valid].var(unbiased=False)),
        by_time=groups,all_aim_weights_zero=bool((data['ctx']['aim_weight']==0).all()),
        xi_constant=bool((data['xi'].std(0)==0).all()),
        limits='Time-only lookup fitted on training environment IDs, evaluated on held-out IDs; single fresh collection.')
    (OUT/'sampling_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['physical','by_time']},indent=2),flush=True)
