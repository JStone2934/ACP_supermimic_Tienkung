"""Finite continuation of the physical-return pilot, with fresh data per policy block."""
import argparse
import hashlib
import time
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.standing import freeze_normalizers
from tools.diagnose_aiming_fullbody_update import imagined
from tools.experiment_aiming_critic_isolation import take, unpack
from tools.pilot_aiming_physical_critic import collect, calibration


def degradation(current,baseline):
    # A coarse early-stop rule for this pilot, not a replacement for standing acceptance.
    reasons=[]
    for name,allowed in [('fall_fraction',.10),('mean_root_max_m',.10),
                         ('mean_feet_max_m',.10),('mean_tilt_max_deg',5.)]:
        if current[name]>baseline[name]+allowed:reasons.append(name)
    return reasons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--extra-actor-updates',type=int,default=100)
    p.add_argument('--block-size',type=int,default=10);p.add_argument('--critic-updates',type=int,default=2000)
    a=p.parse_args();source=Path(a.pilot);out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Experiment identity already exists')
    if a.extra_actor_updates%a.block_size:raise ValueError('Actor budget must contain whole blocks')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261114)
    torch.backends.cuda.matmul.allow_tf32=True
    import warp as wp
    wp.config.quiet=True
    candidate=torch.load(source/'candidate.pt',map_location='cpu',weights_only=False)
    if candidate['stage']!='physical_critic_pilot':raise ValueError('Expected the finite pilot candidate')
    original=torch.load(candidate['metadata']['source'],map_location='cpu',weights_only=False)
    cfg=candidate['config'];task=FullbodyTask('cuda:0',cfg);validate_checkpoint(original,task)
    if cfg['horizon']!=8:raise ValueError('H8 is required')
    pi=FullbodyPolicy().cuda();pi.load_state_dict(candidate['pi']);pi.requires_grad_(False)
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).cuda();wm.load_state_dict(candidate['wm']);wm.requires_grad_(False)
    critic=Critic(task.critic_dim).cuda();critic.load_state_dict(candidate['critic'])
    freeze_normalizers(pi,wm,critic)
    opt_pi=torch.optim.Adam(pi.parameters(),lr=cfg['actor_lr'],betas=tuple(cfg['actor_betas']))
    opt_pi.load_state_dict(candidate['actor_optimizer'])
    opt_v=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr'])
    opt_v.load_state_dict(torch.load(source/'physical_critic.pt',map_location='cpu',weights_only=False)['optimizer'])
    prior_updates=candidate['metadata']['actor_updates']
    meta=dict(pilot=str(source.resolve()),candidate_sha256=hashlib.sha256((source/'candidate.pt').read_bytes()).hexdigest(),
              seed=20261114,horizon=8,world_model_frozen=True,exploration=False,
              initial_actor_updates=prior_updates,extra_actor_budget=a.extra_actor_updates,
              block_size=a.block_size,critic_updates_per_block=a.critic_updates,
              source='new complete physical episodes of a frozen policy for each block',
              target='physical Monte Carlo return; no bootstrap',num_envs=64,train_envs=[0,47],calibration_envs=[48,63],
              early_stop_additive_limits={'fall_fraction':.10,'mean_root_max_m':.10,'mean_feet_max_m':.10,'mean_tilt_max_deg':5.},
              formal_training_resumed=False)
    json_save(out/'experiment.json',meta)
    env=FullbodySim(task,64,lambda n:(task.alpha.new_tensor([[3.,0.,0.,.2]]).repeat(n,1),0.),eval_mode=True)
    began=time.monotonic();records=[];fitting=[];actor_log=[];reason='finite_budget_completed';stopped=[]
    baseline=None
    for extra in range(0,a.extra_actor_updates+1,a.block_size):
        total=prior_updates+extra;pi.requires_grad_(False)
        data,physical=collect(pi,task,cfg,env,out/f'policy_{total:04d}',extra<a.extra_actor_updates,save_data=False)
        if baseline is None:baseline=physical
        record=dict(total_actor_updates=total,extra_actor_updates=extra,physical=physical)
        records.append(record)
        artifact=dict(task=original['task'],stage='physical_critic_pilot',task_config=original['task_config'],
                      robot_metadata=original['robot_metadata'],config=cfg,pi=pi.state_dict(),wm=wm.state_dict(),
                      critic=critic.state_dict(),actor_optimizer=opt_pi.state_dict(),critic_optimizer=opt_v.state_dict(),
                      total_actor_updates=total,metadata=meta)
        torch.save(artifact,out/f'policy_{total:04d}.pt')
        json_save(out/'progress.json',dict(records=records,critic_fitting=fitting,actor_updates=actor_log))
        print('PHYSICAL',total,{k:v for k,v in physical.items() if k!='per_environment'},flush=True)
        stopped=degradation(physical,baseline)
        if stopped:
            reason='coarse_physical_degradation';break
        if extra==a.extra_actor_updates:break
        train=(data['env_id']<48).nonzero().flatten();test=(data['env_id']>=48).nonzero().flatten()
        before=calibration(critic,task,cfg,data,test);critic.requires_grad_(True)
        for update in range(a.critic_updates):
            ids=train[torch.randint(len(train),(512,),device=task.alpha.device)]
            opt_v.zero_grad();loss=(critic(data['x'][ids])-data['y'][ids]).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);opt_v.step()
        after=calibration(critic,task,cfg,data,test);critic.requires_grad_(False)
        fit=dict(policy_total_actor_updates=total,before=before,after=after,last_training_loss=float(loss.detach()))
        fitting.append(fit);print('CRITIC',total,fit,flush=True)
        pi.requires_grad_(True)
        for update in range(a.block_size):
            ids=train[torch.randint(len(train),(cfg['pi_batch'],),device=task.alpha.device)]
            opt_pi.zero_grad();parts,_,_,_=imagined(pi,wm,critic,task,cfg,unpack(take(data,ids)))
            (-sum(parts.values()).mean()).backward()
            norm=torch.nn.utils.clip_grad_norm_(pi.parameters(),1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(pi.parameters(),cfg['actor_grad_clip_value']);opt_pi.step()
            actor_log.append(dict(total_actor_updates=total+update+1,gradient_norm=float(norm),
                                  objectives={k:float(v.detach().mean()) for k,v in parts.items()}))
        del data
    unchanged=all(torch.equal(v.cpu(),candidate['wm'][k]) for k,v in wm.state_dict().items());assert unchanged
    report=dict(experiment=meta,records=records,critic_fitting=fitting,actor_updates=actor_log,
                reason=reason,degradation_metrics=stopped,total_actor_updates=records[-1]['total_actor_updates'],
                physical_transitions=sum(r['physical']['physical_transitions'] for r in records),
                wall_seconds=time.monotonic()-began,world_model_unchanged=unchanged,
                limitation='Single seed, fixed WM, no repeated no-op controls; coarse physical judgment rather than causal proof')
    json_save(out/'report.json',report)
    print('COMPLETE',reason,records[-1]['total_actor_updates'],stopped,flush=True)


if __name__=='__main__':main()
