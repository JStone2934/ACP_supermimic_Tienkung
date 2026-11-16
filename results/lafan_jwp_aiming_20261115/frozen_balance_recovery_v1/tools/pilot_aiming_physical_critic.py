"""Single-seed finite pilot: physical Monte Carlo critic, then ten H8 actor updates.

Uses new deterministic trajectories of one frozen policy. Does not resume the
formal trainer or overwrite its checkpoints. Full episodes avoid bootstrap.
"""
import argparse
import copy
import hashlib
import time
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.mdp import flatten_state
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from tools.diagnose_aiming_fullbody_update import imagined
from tools.experiment_aiming_critic_isolation import take, unpack


def physical_mc(rewards, done, gamma):
    """Terminal reward is included; neither failure nor task timeout bootstraps."""
    result=torch.empty_like(rewards);value=torch.zeros_like(rewards[-1])
    for j in reversed(range(len(rewards))):
        value=rewards[j]+gamma*(~done[j])*value
        result[j]=value
    return result


@torch.no_grad()
def collect(pi,task,cfg,env,out,keep_data,save_data=True):
    torch.manual_seed(20261114)
    env.reset(torch.arange(env.N,device=task.alpha.device))
    hist=env.observe()[:,None].repeat(1,10,1)
    alive=torch.ones(env.N,dtype=torch.bool,device=task.alpha.device)
    root=torch.zeros(env.N,device=task.alpha.device);feet=torch.zeros_like(root);tilt=torch.zeros_like(root)
    falls=torch.zeros_like(alive);first_done=torch.full_like(env.ep_step,-1)
    rewards=[];dones=[];rows=[];features=[];valid=[];times=[]
    began=time.monotonic()
    for j in range(task.episode_steps):
        if keep_data and j%8==0:
            S=env.state()
            rows.append(dict(s=flatten_state(S),history=hist.clone(),ap=env.a_prev.clone(),
                             xi=env.xi.clone(),t=env.ep_step.clone(),ctx={k:v.clone() for k,v in env.context.items()}))
            features.append(task.critic_features(S,hist,env.a_prev,env.xi,env.ep_step,env.context))
            valid.append(alive.clone());times.append(j)
        raw=pi(hist,task.command(env.state(),env.context));tr=env.step(raw)
        if not torch.isfinite(tr['r'][alive]).all():raise FloatingPointError('Nonfinite physical reward')
        S=env.state();foot_pos=task.kin.forward(*S)[0][:,task.feet]
        drift=(S[0][:,:2]-tr['root_anchor_w'][:,:2]).norm(dim=-1)
        foot_drift=(foot_pos[...,:2]-tr['feet_anchor_w'][...,:2]).norm(dim=-1).amax(-1)
        angle=torch.rad2deg(torch.acos(S[1][:,2,2].clamp(-1,1)))
        root=torch.maximum(root,torch.where(alive,drift,torch.zeros_like(root)))
        feet=torch.maximum(feet,torch.where(alive,foot_drift,torch.zeros_like(feet)))
        tilt=torch.maximum(tilt,torch.where(alive,angle,torch.zeros_like(tilt)))
        falls |= alive&tr['terminated'];first_done[alive&tr['done']]=j+1
        rewards.append(torch.where(alive,tr['r'],torch.zeros_like(tr['r'])))
        dones.append(tr['done']|~alive);alive &= ~tr['done']
        hist=torch.cat((hist[:,1:],env.observe()[:,None]),1)
        finished=tr['done'].nonzero().flatten()
        if len(finished):
            env.reset(finished);hist[finished]=env.observe()[finished,None].expand(-1,10,-1)
        if (j+1)%500==0:print(f'{out.name}: physical {j+1}/1500',flush=True)
    if alive.any() or (first_done<0).any():raise AssertionError('A physical episode is incomplete')
    R=torch.stack(rewards);D=torch.stack(dones);G=physical_mc(R,D,cfg['gamma'])
    passed=(~falls)&(root<=.10)&(feet<=.05)&(tilt<=15.)
    summary=dict(physical_transitions=env.N*task.episode_steps,wall_seconds=time.monotonic()-began,
                 environments=env.N,mean_return=float(G[0].mean()),fall_fraction=float(falls.float().mean()),
                 standing_pass_fraction=float(passed.float().mean()),
                 mean_root_max_m=float(root.mean()),mean_feet_max_m=float(feet.mean()),mean_tilt_max_deg=float(tilt.mean()),
                 worst_root_m=float(root.max()),worst_feet_m=float(feet.max()),worst_tilt_deg=float(tilt.max()),
                 per_environment=dict(return_=G[0].tolist(),root_max_m=root.tolist(),feet_max_m=feet.tolist(),
                                      tilt_max_deg=tilt.tolist(),fell=falls.tolist(),first_done=first_done.tolist()))
    json_save(out.with_suffix('.json'),summary)
    if not keep_data:return None,summary
    data={k:({name:torch.stack([r[k][name] for r in rows]).flatten(0,1) for name in rows[0][k]} if k=='ctx'
             else torch.stack([r[k] for r in rows]).flatten(0,1)) for k in rows[0]}
    data.update(x=torch.stack(features).flatten(0,1),y=G[times].flatten(),valid=torch.stack(valid).flatten(),
                env_id=torch.arange(env.N,device=task.alpha.device).repeat(len(times)))
    data=take(data,data['valid'].nonzero().flatten())
    if save_data:torch.save(dict(samples=data,rewards=R,done=D,sample_times=times),out.with_suffix('.pt'))
    return data,summary


@torch.no_grad()
def calibration(critic,task,cfg,data,ids):
    prediction=critic(data['x'][ids]);bounded=bound_bootstrap(prediction,data['t'][ids],task,cfg)
    truth=data['y'][ids]
    def metrics(v):
        error=v-truth
        return dict(mae=float(error.abs().mean()),rmse=float(error.square().mean().sqrt()),bias=float(error.mean()))
    return dict(raw=metrics(prediction),bounded=metrics(bounded),samples=len(ids),true_mean=float(truth.mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True)
    p.add_argument('--critic-updates',type=int,default=5000);p.add_argument('--actor-updates',type=int,default=10)
    a=p.parse_args();out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Experiment output already exists')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261114)
    torch.backends.cuda.matmul.allow_tf32=True
    import warp as wp
    wp.config.quiet=True
    started=time.monotonic();ck=torch.load(a.ckpt,map_location='cpu',weights_only=False);cfg=ck['config']
    if cfg['horizon']!=8:raise ValueError('This pilot requires H8')
    task=FullbodyTask('cuda:0',cfg);validate_checkpoint(ck,task)
    pi=FullbodyPolicy().cuda();pi.load_state_dict(ck['pi']);pi.requires_grad_(False)
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).cuda();wm.load_state_dict(ck['wm']);wm.requires_grad_(False)
    critic=Critic(task.critic_dim).cuda();critic.load_state_dict(ck['critic'])
    freeze_normalizers(pi,wm,critic)
    meta=dict(source=str(Path(a.ckpt).resolve()),source_sha256=hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest(),
              policy_version='source_checkpoint_500_fixed_during_all_critic_data_collection',
              target='full physical discounted return; no bootstrap; no imagined critic targets',
              seed=20261114,num_envs=64,training_envs=[0,47],heldout_envs=[48,63],exploration=False,
              critic_updates=a.critic_updates,critic_batch=512,actor_updates=a.actor_updates,actor_batch=256,horizon=8,
              critic_optimizer='fresh Adam; source critic weights and frozen normalization',
              actor_optimizer='source checkpoint Adam moments; unchanged learning rate and clipping',
              stage='A',formal_training_resumed=False)
    json_save(out/'experiment.json',meta)
    env=FullbodySim(task,64,lambda n:(task.alpha.new_tensor([[3.,0.,0.,.2]]).repeat(n,1),0.),eval_mode=True)
    data,before=collect(pi,task,cfg,env,out/'before',True)
    train=(data['env_id']<48).nonzero().flatten();test=(data['env_id']>=48).nonzero().flatten()
    initial=calibration(critic,task,cfg,data,test);print('CALIBRATION before',initial,flush=True)
    opt_v=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr']);curve=[]
    for update in range(1,a.critic_updates+1):
        ids=train[torch.randint(len(train),(512,),device=task.alpha.device)]
        opt_v.zero_grad();loss=(critic(data['x'][ids])-data['y'][ids]).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);opt_v.step()
        if update%1000==0 or update==a.critic_updates:
            entry=dict(update=update,last_training_loss=float(loss.detach()),heldout=calibration(critic,task,cfg,data,test))
            curve.append(entry);print('CRITIC',entry,flush=True)
    final=calibration(critic,task,cfg,data,test)
    assert all(torch.equal(v.cpu(),ck['pi'][k]) for k,v in pi.state_dict().items())
    torch.save(dict(critic=critic.state_dict(),optimizer=opt_v.state_dict(),metadata=meta),out/'physical_critic.pt')
    if not final['bounded']['mae']<initial['bounded']['mae']:
        json_save(out/'report.json',dict(experiment=meta,before=before,calibration_before=initial,calibration_after=final,
                                      critic_curve=curve,actor_updates=0,result='critic calibration did not improve; actor probe skipped'))
        print('STOP: physical critic did not improve held-out calibration',flush=True);return
    pi.requires_grad_(True);critic.requires_grad_(False)
    opt_pi=torch.optim.Adam(pi.parameters(),lr=cfg['actor_lr'],betas=tuple(cfg['actor_betas']))
    opt_pi.load_state_dict(copy.deepcopy(ck['optimizers'][0]));updates=[]
    for update in range(1,a.actor_updates+1):
        ids=train[torch.randint(len(train),(cfg['pi_batch'],),device=task.alpha.device)]
        b=unpack(take(data,ids));opt_pi.zero_grad()
        parts,_,_,_=imagined(pi,wm,critic,task,cfg,b)
        objective=sum(parts.values()).mean();(-objective).backward()
        grad=torch.nn.utils.clip_grad_norm_(pi.parameters(),1.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_value_(pi.parameters(),cfg['actor_grad_clip_value']);opt_pi.step()
        row=dict(update=update,gradient_norm=float(grad),objectives={k:float(v.detach().mean()) for k,v in parts.items()})
        updates.append(row);print('ACTOR',row,flush=True)
    pi.requires_grad_(False)
    _,after=collect(pi,task,cfg,env,out/'after',False)
    unchanged_wm=all(torch.equal(v.cpu(),ck['wm'][k]) for k,v in wm.state_dict().items());assert unchanged_wm
    # Diagnostic artifact deliberately cannot be resumed by the formal trainer.
    torch.save(dict(task=ck['task'],stage='physical_critic_pilot',config=cfg,pi=pi.state_dict(),wm=wm.state_dict(),
                    critic=critic.state_dict(),actor_optimizer=opt_pi.state_dict(),metadata=meta),out/'candidate.pt')
    changes={k:after[k]-before[k] for k in ['mean_return','fall_fraction','standing_pass_fraction',
                                         'mean_root_max_m','mean_feet_max_m','mean_tilt_max_deg']}
    json_save(out/'report.json',dict(experiment=meta,calibration_before=initial,calibration_after=final,
                                  critic_curve=curve,before=before,after=after,changes=changes,
                                  actor_updates=updates,world_model_unchanged=unchanged_wm,
                                  total_physical_transitions=before['physical_transitions']+after['physical_transitions'],
                                  wall_seconds=time.monotonic()-started,
                                  limitation='Single seed and short update budget; before/after may include simulator variability; no causal or convergence guarantee'))
    print('COMPLETE',changes,flush=True)


if __name__=='__main__':main()
