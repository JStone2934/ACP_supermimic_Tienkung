import sys
sys.path.insert(0, '/home/js/robot/locomotion/standing-aiming/logs/tienkung3_29dof/standing_aiming_fullbody_v2/dual_h16_critic4096_reward_v2_20261114/runner')
"""Finite H16 action-direction checks with complete physical tails, on saved copies."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyPolicy, FullbodyTask, flat_boxes
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_physical import execution_setup, to_device, take
from supermimic.acp.mdp import flatten_state, unflatten_state, normalize_xi
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from supermimic.rotations import mat_to_quat


def imagined(actions, b, wm, critic, task, cfg):
    S, hist, ap, t = unflatten_state(b['s']), b['history'], b['ap'], b['t']
    alive = torch.ones_like(t, dtype=torch.float32)
    reward = torch.zeros_like(alive)
    for j, raw in enumerate(actions):
        a, _ = task.compose_action(raw, b['xi'])
        S = wm.step(S, a, normalize_xi(b['xi']), task.mdp.scan(S, flat_boxes(S)))
        reward = reward + cfg['gamma']**j * alive * task.reward(S,a,ap,b['ctx'],raw=raw,xi=b['xi'])
        alive = alive*(1-task.soft_termination(S,t+1,b['ctx']))
        hist = torch.cat((hist[:,1:],task.mdp.obs(S,a,b['xi'])[:,None]),1)
        ap,t=a,t+1
    tail=cfg['gamma']**cfg['horizon']*alive*bound_bootstrap(critic(task.critic_features(S,hist,ap,b['xi'],t,b['ctx'])),t,task,cfg)
    return dict(reward=reward,tail=tail,total=reward+tail)


@torch.no_grad()
def restore(env, b):
    ids=torch.arange(env.N,device=env.device)
    env.sim.reset(ids);env.scene.reset(ids)
    S=unflatten_state(b['s']);p,R,v,w,q,qd=S
    env.robot.write_root_link_pose_to_sim(torch.cat((p+env.scene.env_origins,mat_to_quat(R)),-1),env_ids=ids)
    env.robot.write_root_link_velocity_to_sim(torch.cat((v,w),-1),env_ids=ids)
    env.robot.write_joint_position_to_sim(q[:,env.to_internal],env_ids=ids)
    env.robot.write_joint_velocity_to_sim(qd[:,env.to_internal],env_ids=ids)
    env.robot.set_joint_position_target(q[:,env.to_internal]);env.sim.forward()
    env.xi=b['xi'].clone();env.a_prev=b['ap'].clone();env.ep_step=b['t'].clone()
    env.context={k:v.clone() for k,v in b['ctx'].items()}
    error=float((flatten_state(env.state())-b['s']).abs().max())
    assert error<1e-4,error
    return error


@torch.no_grad()
def physical(env,b,pi,critic,task,cfg,actions=None,steps=1500):
    error=restore(env,b); hist=b['history'].clone();alive=torch.ones(env.N,dtype=torch.bool,device=env.device)
    total=torch.zeros(env.N,device=env.device);short=total.clone();tail=total.clone(); length=total.clone()
    raw_sequence=[];states=[]
    predicted_real_tail=None
    for j in range(steps):
        raw=actions[j] if actions is not None and j<cfg['horizon'] else pi(hist,task.command(env.state(),env.context))
        raw_sequence.append(raw.clone())
        tr=env.step(raw);r=cfg['gamma']**j*torch.where(alive,tr['r'],0.)
        total+=r;length+=alive
        if j<cfg['horizon']:short+=r
        else:tail+=r
        alive &= ~tr['done']
        hist=torch.cat((hist[:,1:],env.observe()[:,None]),1)
        if j<cfg['horizon']:states.append(tr['s1'].clone())
        if j==cfg['horizon']-1:
            predicted_real_tail=cfg['gamma']**cfg['horizon']*alive*bound_bootstrap(critic(task.critic_features(env.state(),hist,env.a_prev,env.xi,env.ep_step,env.context)),env.ep_step,task,cfg)
        if not alive.any():break
    assert steps==cfg['horizon'] or not alive.any(), 'Incomplete physical return'
    return dict(total=total,reward=short,tail=tail,length=length,predicted_real_tail=predicted_real_tail,
                restore_error=error,actions=torch.stack(raw_sequence[:cfg['horizon']]),states=torch.stack(states))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args();out=Path(args.directory)
    torch.set_num_threads(4);dev='cuda:1';torch.cuda.set_device(dev)
    import warp as wp
    wp.config.quiet=True
    ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);cfg=ck['config'];execution_setup(cfg)
    task=FullbodyTask(dev,cfg);pi=FullbodyPolicy(task.robot).to(dev);pi.load_state_dict(ck['pi'])
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(dev);wm.load_state_dict(ck['wm'])
    critic=Critic(task.critic_dim).to(dev);critic.load_state_dict(ck['critic'])
    freeze_normalizers(pi,wm,critic)
    for model in [pi,wm,critic]:model.requires_grad_(False)
    art=torch.load(out/'policy_update/counterfactuals.pt',map_location='cpu',weights_only=False)
    d=to_device(art['diagnostic'],dev);ids=((d['t']>=4)&(d['t']<=36)).nonzero().flatten()[:16];b=take(d,ids)
    n=len(ids); repeats=4
    b=take(b,torch.arange(n,device=dev).repeat_interleave(repeats))
    env=FullbodySim(task,n*repeats,lambda count:(task.alpha.new_tensor([[3.,0.,0.,.2]]).repeat(count,1),0.),eval_mode=True)
    nominal=physical(env,b,pi,critic,task,cfg,steps=cfg['horizon'])
    actions=nominal['actions'].detach().requires_grad_()
    pred=imagined(actions,b,wm,critic,task,cfg)
    directions={}
    for kind in ['total','reward','tail']:
        g=torch.autograd.grad(pred[kind].mean(),actions,retain_graph=True)[0]
        joint_g=g/task.alpha
        directions[kind]=(joint_g/joint_g.abs().amax((0,2),keepdim=True).clamp_min(1e-12)/task.alpha).detach()
    actions=actions.detach()
    report=dict(checkpoint_iteration=ck['iteration'],states=n,replicas_per_state=repeats,
                start_steps=b['t'].reshape(n,repeats)[:,0].tolist(),epsilon_joint_rad=.005,
                note='Open-loop first H16 actions, then the same original policy to actual termination; separate from live training.',cases={})
    for label,adjust in [('baseline_0',torch.zeros_like(actions)),('baseline_1',torch.zeros_like(actions))]+[(kind+'_'+sign,(1 if sign=='plus' else -1)*.005*direction) for kind,direction in directions.items() for sign in ['plus','minus']]+[('baseline_final',torch.zeros_like(actions))]:
        raw=actions+adjust
        with torch.no_grad():predict=imagined(raw,b,wm,critic,task,cfg)
        actual=physical(env,b,pi,critic,task,cfg,raw)
        case={}
        for prefix,values in [('predicted',predict),('physical',actual)]:
            for key in ['total','reward','tail','length','predicted_real_tail']:
                if key in values and values[key] is not None:
                    v=values[key];case[prefix+'_'+key]=float(v.mean());case[prefix+'_'+key+'_by_state']=v.reshape(n,repeats).mean(1).cpu().tolist()
        case['restore_error']=actual['restore_error'];report['cases'][label]=case
        (out/'h16_directions.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,{k:round(v,6) for k,v in case.items() if not isinstance(v,list)},flush=True)
    comparisons={}
    for kind in directions:
        plus,minus=report['cases'][kind+'_plus'],report['cases'][kind+'_minus'];c={}
        for name in ['predicted_total','predicted_tail','predicted_reward','physical_total','physical_tail','physical_reward','physical_predicted_real_tail']:
            diff=torch.tensor(plus[name+'_by_state'])-torch.tensor(minus[name+'_by_state'])
            c[name]=dict(mean=float(diff.mean()),state_sem=float(diff.std()/n**.5))
        comparisons[kind]=c
    report['plus_minus']=comparisons
    (out/'h16_directions.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
