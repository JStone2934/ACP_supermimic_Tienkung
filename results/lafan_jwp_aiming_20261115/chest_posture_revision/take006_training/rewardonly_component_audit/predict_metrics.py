"""Read-only ACP reward-gradient decomposition on recorded physical states."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from supermimic.acp.aiming_fullbody import FullbodyPolicy, flat_boxes
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_adapters import configure_policy_training
from supermimic.acp.mdp import unflatten_state, normalize_xi
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','trained','trajectory','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--split-stability',action='store_true')
    p.add_argument('--include-bootstrap',action='store_true');a=p.parse_args()
    assert not a.out.exists();torch.set_num_threads(2)
    ck=torch.load(a.source,map_location=a.device,weights_only=False)
    learned=torch.load(a.trained,map_location=a.device,weights_only=False)
    cfg=dict(ck['config'],horizon=32)
    task=WideAimingTask(a.device,cfg)
    pi=FullbodyPolicy(task.robot).to(a.device);pi.load_state_dict(ck['pi'])
    configure_policy_training(pi,dict(cfg,policy_training_scope='full'))
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(a.device);wm.load_state_dict(ck['wm'])
    wm.requires_grad_(False);freeze_normalizers(pi,wm)
    critic=None
    if a.include_bootstrap:
        critic=Critic(task.critic_dim).to(a.device);critic.load_state_dict(ck['critic']);critic.requires_grad_(False)
    with np.load(a.trajectory) as z:d={k:torch.as_tensor(z[k],device=a.device) for k in z.files}
    assert not d['done'][:128].any() and not d['ep_step'][0].any()
    n=d['s'].shape[1];S0=unflatten_state(d['s'][0])
    ctx=task.context(S0,task.alpha.new_tensor([[3.,0.,0.,.1]]).expand(n,-1),d['aim_weight'][0])
    chest=ctx['aim_target_w']-3*ctx['aim_normal_w']
    ctx.update(aim_target_w=d['aim_target_w'][0],aim_tolerance=d['aim_tolerance'][0],
               aim_normal_w=torch.nn.functional.normalize(d['aim_target_w'][0]-chest,dim=-1))
    contexts=[];times=(64,112)
    with torch.no_grad():
        for t in range(max(times)+1):
            ctx=task.observe_context(unflatten_state(d['s'][t]),ctx)
            if t in times:contexts.append({k:v.clone() for k,v in ctx.items()})
            ctx=task.advance_context(ctx,1)
    context={k:torch.cat([c[k] for c in contexts]) for k in ctx}
    states=unflatten_state(torch.cat([d['s'][t] for t in times]))
    histories=[]
    for t in times:
        observations=[task.mdp.obs(unflatten_state(d['s'][j]),d['a'][j-1],d['xi'][j]) for j in range(t-9,t+1)]
        histories.append(torch.stack(observations,1))
    history=torch.cat(histories);ap=torch.cat([d['a'][t-1] for t in times])
    xi=torch.cat([d['xi'][t] for t in times]);clock=torch.cat([d['ep_step'][t] for t in times]).long()
    regularizers={'request_limit','action_rate','joint_limit'}
    group_names=(('aim','chest_upright','chest_excess','upright','height','feet','foot_motion','foot_velocity','velocity',
                  'angular_velocity','root','heading','foot_boundary','regularization')
                 if a.split_stability else ('aim','stability','regularization'))
    if a.include_bootstrap:group_names=group_names+('bootstrap',)

    def step(S,h,previous,t):
        current=task.advance_context(context,t-clock);following=task.advance_context(context,t+1-clock)
        raw=pi(h,task.command(S,current));action,_=task.compose_action(raw,xi)
        nxt=wm.step(S,action,normalize_xi(xi),task.mdp.scan(S,flat_boxes(S)))
        _,terms=task.reward_terms(nxt,action,previous,following,raw=raw,xi=xi)
        if a.split_stability:
            groups=torch.stack([sum(v for k,v in terms.items() if k in regularizers)
                                if name=='regularization' else terms.get(name,terms['aim']*0)
                                for name in group_names],-1)
        else:
            groups=torch.stack((terms['aim'],sum(v for k,v in terms.items() if k!='aim' and k not in regularizers),
                                sum(v for k,v in terms.items() if k in regularizers)),-1)
            if a.include_bootstrap:groups=torch.cat((groups,groups[:,:1]*0),-1)
        done=task.soft_termination(nxt,t+1,following)
        h=torch.cat((h[:,1:],task.mdp.obs(nxt,action,xi)[:,None]),1)
        return nxt,h,action,groups,done

    metric_rows=[]
    def objective():
        S,h,previous,t=states,history,ap,clock
        alive=torch.ones(len(clock),device=a.device);total=torch.zeros(len(clock),len(group_names),device=a.device)
        for j in range(cfg['horizon']):
            S,h,previous,groups,done=checkpoint(step,S,h,previous,t,use_reentrant=False)
            total=total+cfg['gamma']**j*alive[:,None]*groups
            alive=alive*(1-done);t=t+1
        if critic is not None:
            features=task.critic_features(S,h,previous,xi,t,task.advance_context(context,t-clock))
            tail=cfg['gamma']**cfg['horizon']*alive*bound_bootstrap(critic(features),t,task,cfg)
            total=torch.cat((total[:,:-1],total[:,-1:]+tail[:,None]),-1)
        with torch.no_grad():
            _,br,_,_=task.kin.forward(*S);R=br[:,task.torso];tilt=torch.rad2deg(torch.atan2(torch.linalg.vector_norm(R[:,:2,2],dim=-1),R[:,2,2]));pelvis=torch.rad2deg(torch.atan2(torch.linalg.vector_norm(S[1][:,:2,2],dim=-1),S[1][:,2,2]));metric_rows.append(dict(chest_mean_deg=float(tilt.mean()),chest_max_deg=float(tilt.max()),pelvis_mean_deg=float(pelvis.mean()),root_z_mean_m=float(S[0][:,2].mean()),alive_mean=float(alive.mean())))
        return total.mean(0)

    parameters=[p for p in pi.parameters() if p.requires_grad]
    names=[n for n,p in pi.named_parameters() if p.requires_grad]
    delta=[learned['pi'][n]-ck['pi'][n] for n in names]
    norm=lambda g:sum(v.square().sum() for v in g).sqrt()
    torch.set_grad_enabled(False);values=objective();rows={}
    with torch.no_grad():
        before=values.detach().cpu().tolist();pi.load_state_dict(learned['pi']);after=objective().cpu().tolist()
    report=dict(scope=__doc__+' Not physical acceptance.',source_sha256=sha256(a.source),
                trained_sha256=sha256(a.trained),trajectory_sha256=sha256(a.trajectory),
                predicted_endpoint_metrics=metric_rows,groups=rows,group_order=list(group_names),source_predicted_rewards=before,trained_predicted_rewards=after,
                fixed_source_world_model=True,recorded_start_times=list(times),sample_count=len(clock),
                horizon=cfg['horizon'],physical_policy_weights_modified=False)
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
