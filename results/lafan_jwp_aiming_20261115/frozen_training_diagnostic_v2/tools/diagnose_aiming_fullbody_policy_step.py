"""Finite counterfactual Adam-step probes; modifies disposable copies, never a training run."""
import argparse
import copy
import json
from pathlib import Path
import torch
from tools.diagnose_aiming_fullbody_update import (
    FullbodyTask, FullbodyPolicy, WorldModel, Critic, FullbodySim, samples, imagined,
    freeze_normalizers, mat_to_quat, flatten_state, json_save)


@torch.no_grad()
def physical(models, labels, task, cfg, b):
    n, groups = len(b['t']), len(models)
    rep = lambda x: torch.cat([x] * groups)
    S = tuple(rep(x) for x in b['S']); ctx = {k: rep(v) for k,v in b['ctx'].items()}
    env = FullbodySim(task, n*groups, lambda count:(task.alpha.new_tensor([[3.,0.,0.,.2]]).repeat(count,1),0.), eval_mode=True)
    ids = torch.arange(env.N,device=task.alpha.device)
    env.sim.reset(ids); env.scene.reset(ids)
    p,R,v,w,q,qd=S
    env.robot.write_root_link_pose_to_sim(torch.cat((p+env.scene.env_origins,mat_to_quat(R)),-1),env_ids=ids)
    env.robot.write_root_link_velocity_to_sim(torch.cat((v,w),-1),env_ids=ids)
    env.robot.write_joint_position_to_sim(q[:,env.to_internal],env_ids=ids)
    env.robot.write_joint_velocity_to_sim(qd[:,env.to_internal],env_ids=ids)
    env.robot.set_joint_position_target(q[:,env.to_internal]);env.sim.forward()
    env.a_prev=rep(b['ap']);env.ep_step=rep(b['t']);env.xi=rep(b['xi']);env.context=ctx
    assert (flatten_state(env.state())-flatten_state(S)).abs().max()<1e-4
    hist=rep(b['history']);standing=torch.zeros(env.N,device=task.alpha.device);request=torch.zeros_like(standing)
    for j in range(8):
        cmd=task.command(env.state(),ctx)
        raw=torch.cat([model(hist[i*n:(i+1)*n],cmd[i*n:(i+1)*n]) for i,model in enumerate(models)])
        tr=env.step(raw);terms=tr['reward_contributions']
        standing += cfg['gamma']**j * sum(v for k,v in terms.items() if k!='request_limit')
        request += cfg['gamma']**j * terms['request_limit']
        hist=torch.cat((hist[:,1:],env.observe()[:,None]),1)
    return {label:dict(standing_reward_8=float(standing[i*n:(i+1)*n].mean()),
                       request_reward_8=float(request[i*n:(i+1)*n].mean())) for i,label in enumerate(labels)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--trajectory',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(710)
    import warp as wp
    wp.config.quiet=True
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False);cfg=ck['config'];task=FullbodyTask('cuda:0',cfg)
    pi=FullbodyPolicy().cuda();pi.load_state_dict(ck['pi'])
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).cuda();wm.load_state_dict(ck['wm']);wm.requires_grad_(False)
    critic=Critic(task.critic_dim).cuda();critic.load_state_dict(ck['critic']);critic.requires_grad_(False)
    freeze_normalizers(pi,wm,critic);b=samples(task,a.trajectory,64)
    parts,_,_,_=imagined(pi,wm,critic,task,cfg,b)
    params=list(pi.parameters())
    grads={k:torch.autograd.grad(v.mean(),params,retain_graph=True) for k,v in parts.items()}
    base={k:v.detach().clone() for k,v in pi.state_dict().items()};names=list(dict(pi.named_parameters()))
    deltas={}
    for mode in ['full_saved','full_fresh','no_bootstrap_fresh']:
        candidate=FullbodyPolicy().cuda();candidate.load_state_dict(base)
        opt=torch.optim.Adam(candidate.parameters(),lr=cfg['actor_lr'],betas=tuple(cfg['actor_betas']))
        if mode=='full_saved':opt.load_state_dict(copy.deepcopy(ck['optimizers'][0]))
        keys=['standing','request'] if mode=='no_bootstrap_fresh' else list(grads)
        for i,param in enumerate(candidate.parameters()):param.grad=-sum(grads[k][i] for k in keys)
        torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.)
        torch.nn.utils.clip_grad_value_(candidate.parameters(),cfg['actor_grad_clip_value']);opt.step()
        deltas[mode]={k:(candidate.state_dict()[k]-base[k]).detach().clone() for k in names}
        del opt,candidate
    del parts,grads
    b=samples(task,a.trajectory,16)
    result={'checkpoint':a.ckpt,'batch':16,'note':'One counterfactual Adam step on a fixed real-state batch; scale 10 is a directional sensitivity test, not ten training steps. No checkpoint or service is updated.','scales':[]}
    for scale in [1.,10.]:
        models=[pi];labels=['baseline']
        for mode,delta in deltas.items():
            for sign in [1.,-1.]:
                model=FullbodyPolicy().cuda();model.load_state_dict(base)
                with torch.no_grad():
                    for name,param in model.named_parameters():param.add_(delta[name],alpha=sign*scale)
                model.requires_grad_(False);models.append(model);labels.append(mode+('_plus' if sign>0 else '_minus'))
        models.extend([pi,pi]);labels.extend(['baseline_replica_1','baseline_replica_2'])
        real=physical(models,labels,task,cfg,b)
        with torch.no_grad():
            for label,model in zip(labels,models):
                terms,_,_,_=imagined(model,wm,critic,task,cfg,b)
                real[label]['model_objectives']={k:float(v.mean()) for k,v in terms.items()}
        result['scales'].append({'scale':scale,'cases':real});json_save(a.out,result)
        print(json.dumps(result['scales'][-1]),flush=True)
        del models


if __name__=='__main__':main()
