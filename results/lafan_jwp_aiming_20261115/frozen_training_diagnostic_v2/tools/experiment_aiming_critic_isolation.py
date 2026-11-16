"""Finite critic-only experiment with fixed actor/WM and held-out physical returns."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, CONTEXT_FIELDS, flat_boxes, validate_checkpoint
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.mdp import flatten_state, unflatten_state, normalize_xi
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from supermimic.rotations import mat_to_quat
from tools.diagnose_aiming_fullbody_update import imagined


def take(data, ids):
    return {k: ({n:x[ids] for n,x in v.items()} if k=='ctx' else v[ids]) for k,v in data.items()}


def unpack(data):
    return dict(data, S=unflatten_state(data['s']))


def sampler(task):
    return lambda n:(task.alpha.new_tensor([[3.,0.,0.,.2]]).repeat(n,1),0.)


@torch.no_grad()
def collect(pi, task, cfg, out):
    env=FullbodySim(task,32,sampler(task));hist=env.observe()[:,None].repeat(1,10,1)
    noise=task.alpha.new_full((29,),cfg['body_noise_rad']);noise[task.arm_ids]=cfg['arm_noise_rad']*.97
    rows=[]
    for j in range(1500):
        obs=env.observe();hist=torch.where((env.ep_step==0)[:,None,None],obs[:,None].expand_as(hist),hist)
        row=dict(s=flatten_state(env.state()),history=hist.clone(),ap=env.a_prev.clone(),xi=env.xi.clone(),t=env.ep_step.clone(),
                 ctx={k:v.clone() for k,v in env.context.items()})
        rows.append(row)
        raw=pi(hist,task.command(env.state(),env.context))+torch.randn_like(env.a_prev)*noise/task.alpha
        tr=env.step(raw);hist=torch.cat((hist[:,1:],env.observe()[:,None]),1)
        if (j+1)%300==0: print(f'collect {j+1}/1500',flush=True)
    data={k:({n:torch.stack([r[k][n] for r in rows]).flatten(0,1) for n in CONTEXT_FIELDS} if k=='ctx'
              else torch.stack([r[k] for r in rows]).flatten(0,1)) for k in rows[0]}
    all_ids=torch.arange(1500*32,device=task.alpha.device)
    training=all_ids[all_ids%32<16];testing=all_ids[all_ids%32>=16]
    train_ids=training[torch.randperm(len(training),device=task.alpha.device)[:4096]]
    # Stratify evaluation over the full recorded time range, with disjoint environments.
    test_ids=testing[torch.linspace(0,len(testing)-1,64,device=task.alpha.device).long()]
    train,test=take(data,train_ids),take(data,test_ids)
    torch.save(dict(train=train,test=test,train_ids=train_ids,test_ids=test_ids),out/'fixed_states.pt')
    return train,test


@torch.no_grad()
def restore(task,b,env=None):
    if env is None:env=FullbodySim(task,len(b['t']),sampler(task),eval_mode=True)
    if env.N!=len(b['t']):raise ValueError('Restore batch size differs')
    ids=torch.arange(env.N,device=task.alpha.device);env.sim.reset(ids);env.scene.reset(ids)
    S=unflatten_state(b['s']);p,R,v,w,q,qd=S
    env.robot.write_root_link_pose_to_sim(torch.cat((p+env.scene.env_origins,mat_to_quat(R)),-1),env_ids=ids)
    env.robot.write_root_link_velocity_to_sim(torch.cat((v,w),-1),env_ids=ids)
    env.robot.write_joint_position_to_sim(q[:,env.to_internal],env_ids=ids)
    env.robot.write_joint_velocity_to_sim(qd[:,env.to_internal],env_ids=ids)
    env.robot.set_joint_position_target(q[:,env.to_internal]);env.sim.forward()
    env.xi=b['xi'].clone();env.a_prev=b['ap'].clone();env.ep_step=b['t'].clone();env.context={k:v.clone() for k,v in b['ctx'].items()}
    error=float((flatten_state(env.state())-b['s']).abs().max())
    if error>1e-4:raise ValueError(f'Restore error {error}')
    return env


def repeat(data,n):
    return {k:({name:torch.cat([x]*n) for name,x in v.items()} if k=='ctx' else torch.cat([v]*n)) for k,v in data.items()}


@torch.no_grad()
def physical_returns(pi,task,cfg,b,out,first_actions=None,env=None,capture_tail=False):
    env=restore(task,b,env);hist=b['history'].clone();alive=torch.ones(env.N,dtype=torch.bool,device=task.alpha.device)
    total=torch.zeros(env.N,device=task.alpha.device);standing=torch.zeros_like(total);eight=torch.zeros_like(total)
    first_done=torch.full((env.N,),-1,device=task.alpha.device,dtype=torch.long);actions=[]
    max_steps=int((task.episode_steps-b['t']).max())
    for j in range(max_steps):
        raw=first_actions[j] if first_actions is not None and j<8 else pi(hist,task.command(env.state(),env.context))
        if j<8:actions.append(raw.clone())
        tr=env.step(raw)
        r=tr['r'];rs=sum(v for k,v in tr['reward_contributions'].items() if k!='request_limit')
        total+=cfg['gamma']**j*torch.where(alive,r,torch.zeros_like(r))
        standing+=cfg['gamma']**j*torch.where(alive,rs,torch.zeros_like(rs))
        first_done[alive&tr['done']]=j+1;alive &= ~tr['done']
        if j==7:eight=total.clone()
        hist=torch.cat((hist[:,1:],env.observe()[:,None]),1)
        if j==7 and capture_tail:
            tail_features=task.critic_features(env.state(),hist,env.a_prev,env.xi,env.ep_step,env.context).clone()
            tail_alive=alive.clone()
        # Prevent post-terminal numerical problems; finished lanes never contribute again.
        done=tr['done'].nonzero().flatten()
        if len(done):
            env.reset(done);hist[done]=env.observe()[done,None].expand(-1,10,-1)
        if (j+1)%300==0:print(f'physical {out.name}: {j+1}/{max_steps}, active={int(alive.sum())}',flush=True)
        if not alive.any():break
    if alive.any():raise AssertionError('Incomplete return rollout')
    result=dict(total=total,standing=standing,first8=eight,actions=torch.stack(actions),first_done=first_done)
    if capture_tail:result.update(tail_features=tail_features,tail_alive=tail_alive)
    torch.save(result,out);return result


@torch.no_grad()
def build_bank(pi,wm,task,cfg,data):
    banks=[]
    for first in range(0,len(data['t']),128):
        b=unpack(take(data,slice(first,first+128)));S=b['S'];hist=b['history'];ap=b['ap'];t=b['t'];alive=torch.ones_like(t,dtype=torch.float32)
        features=[];rewards=[];dones=[];masks=[]
        for j in range(cfg['horizon']):
            features.append(task.critic_features(S,hist,ap,b['xi'],t,b['ctx']))
            raw=pi(hist,task.command(S,b['ctx']));a,_=task.compose_action(raw,b['xi'])
            S=wm.step(S,a,normalize_xi(b['xi']),task.mdp.scan(S,flat_boxes(S)))
            rewards.append(task.reward(S,a,ap,b['ctx'],raw=raw,xi=b['xi']))
            d=task.soft_termination(S,t+1,b['ctx']);dones.append(d);masks.append(alive)
            alive=alive*(1-d);t=t+1;ap=a;hist=torch.cat((hist[:,1:],task.mdp.obs(S,a,b['xi'])[:,None]),1)
        features.append(task.critic_features(S,hist,ap,b['xi'],t,b['ctx']))
        banks.append(dict(x=torch.stack(features,1),r=torch.stack(rewards,1),d=torch.stack(dones,1),mask=torch.stack(masks,1),t=b['t']))
    return {k:torch.cat([b[k] for b in banks]) for k in banks[0]}


@torch.no_grad()
def targets(critic,task,cfg,b):
    steps=b['t'][:,None]+torch.arange(cfg['horizon']+1,device=b['t'].device)[None]
    V=bound_bootstrap(critic(b['x']),steps,task,cfg)
    ret=V[:,-1];ys=[]
    for j in reversed(range(cfg['horizon'])):
        ret=b['r'][:,j]+cfg['gamma']*(1-b['d'][:,j])*((1-cfg['lambda_return'])*V[:,j+1]+cfg['lambda_return']*ret)
        ys.append(ret)
    return torch.stack(ys[::-1],1)


@torch.no_grad()
def calibration(critic,task,cfg,test,truth,bank):
    b=unpack(test);X=task.critic_features(b['S'],b['history'],b['ap'],b['xi'],b['t'],b['ctx'])
    raw=critic(X);capped=bound_bootstrap(raw,b['t'],task,cfg)
    def stats(pred):
        err=pred-truth
        return dict(mae=float(err.abs().mean()),rmse=float(err.square().mean().sqrt()),bias=float(err.mean()),
                    mean_prediction=float(pred.mean()),correlation=float(torch.corrcoef(torch.stack((pred,truth)))[0,1]))
    fixed={k:v[:256] for k,v in bank.items()};Y=targets(critic,task,cfg,fixed)
    mse=((critic(fixed['x'][:,:-1])-Y).square()*fixed['mask']).sum()/fixed['mask'].sum()
    return dict(raw=stats(raw),bounded=stats(capped),true_mean=float(truth.mean()),true_std=float(truth.std()),
                capped_fraction=float((raw>capped+1e-6).float().mean()),fixed_bank_td_mse=float(mse),
                predictions_raw=raw.tolist(),predictions_bounded=capped.tolist(),true_returns=truth.tolist())


def direction(pi,wm,critic,task,cfg,probe,base_actions):
    critic.requires_grad_(False)
    b=unpack(probe);actions=base_actions.clone().requires_grad_(True)
    parts,_,_,_=imagined(pi,wm,critic,task,cfg,b,actions=actions)
    gradient=torch.autograd.grad(parts['bootstrap'].sum(),actions)[0]/task.alpha
    magnitude=gradient.abs().amax(dim=(0,2),keepdim=True)
    result=(gradient/magnitude.clamp_min(1e-12)/task.alpha).detach()
    stats=dict(action_gradient_norm=float(gradient.norm()),active_fraction=float((magnitude.flatten()>1e-8).float().mean()))
    critic.requires_grad_(True)
    return result,stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--resume-cache',action='store_true')
    a=p.parse_args();out=Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume_cache:raise FileExistsError('Use a new experiment identity')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261114)
    torch.backends.cuda.matmul.allow_tf32=True
    import warp as wp
    wp.config.quiet=True
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False);cfg=ck['config'];task=FullbodyTask('cuda:0',cfg);validate_checkpoint(ck,task)
    pi=FullbodyPolicy().cuda();pi.load_state_dict(ck['pi']);pi.requires_grad_(False)
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).cuda();wm.load_state_dict(ck['wm']);wm.requires_grad_(False)
    freeze_normalizers(pi,wm)
    critic=Critic(task.critic_dim).cuda();critic.load_state_dict(ck['critic'])
    optim=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr']);optim.load_state_dict(ck['optimizers'][2])
    meta=dict(source=str(Path(a.ckpt).resolve()),source_sha256=hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest(),
              frozen=['actor','world_model','actor_normalizers','world_model_normalizers'],
              train_env_ids=list(range(16)),test_env_ids=list(range(16,32)),training_starts=4096,test_starts=64,
              milestones=[0,100,500,a.steps],updates_per_iteration=20,seed=20261114,
              target='Unchanged H8 imagined lambda-return with same-critic bootstrap and configured upper bound',
              physical_return='Original total reward including request penalty, gamma=.99, hard termination/timeout, no bootstrap, fixed policy without exploration after sampled start')
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text())!=meta:raise ValueError('Cache identity differs')
    json_save(out/'experiment.json',meta)
    if a.resume_cache and (out/'fixed_states.pt').exists():
        cached=torch.load(out/'fixed_states.pt',weights_only=False);train,test=cached['train'],cached['test']
    else:train,test=collect(pi,task,cfg,out)
    if a.resume_cache and (out/'true_returns.pt').exists():real=torch.load(out/'true_returns.pt',weights_only=False)
    else:real=physical_returns(pi,task,cfg,repeat(test,2),out/'true_returns.pt')
    truth=real['total'].reshape(2,64).mean(0)
    # Exclude near-terminal starts and starts that terminate in the first 8 steps from direction probes.
    valid=((test['t']<1400)&(real['first_done'][:64]>8)).nonzero().flatten()
    ids=valid[torch.linspace(0,len(valid)-1,16,device=valid.device).long()]
    probe=take(test,ids);base_actions=real['actions'][:8,ids]
    print('building frozen imagined bank',flush=True);bank=build_bank(pi,wm,task,cfg,train)
    results=[];directions={};last_loss=None;started=time.monotonic()
    for iteration in range(a.steps+1):
        if iteration:
            ids_train=torch.randint(len(bank['t']),(cfg['pi_batch'],),device=task.alpha.device)
            b={k:v[ids_train] for k,v in bank.items()};X=b['x'][:,:-1].transpose(0,1);critic.norm.update(X)
            Y=targets(critic,task,cfg,b).transpose(0,1);X=X.flatten(0,1);Y=Y.flatten();mask=b['mask'].transpose(0,1).flatten()
            for _ in range(cfg['critic_epochs']):
                for part in torch.randperm(len(X),device=X.device).chunk(cfg['critic_minibatches']):
                    optim.zero_grad();loss=((critic(X[part])-Y[part]).square()*mask[part]).sum()/mask[part].sum().clamp_min(1)
                    loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);optim.step()
            last_loss=float(loss.detach())
        if iteration%100==0:print(f'critic round={iteration}/{a.steps} loss={last_loss} elapsed={time.monotonic()-started:.1f}s',flush=True)
        if iteration in meta['milestones']:
            r=calibration(critic,task,cfg,test,truth,bank);r.update(extra_rounds=iteration,extra_gradient_updates=iteration*20,last_training_loss=last_loss)
            d,stats=direction(pi,wm,critic,task,cfg,probe,base_actions);directions[str(iteration)]=d;r['direction']=stats
            results.append(r);torch.save(dict(critic=critic.state_dict(),optimizer=optim.state_dict(),extra_rounds=iteration),out/f'critic_{iteration:04d}.pt')
            json_save(out/'calibration.json',results);print('CALIBRATION',iteration,r['bounded'],flush=True)
    # Compare all retained directions in one physical batch, including identical baseline controls.
    labels=['baseline_begin'];action_sets=[base_actions];model_deltas={}
    for milestone,d in directions.items():
        c=torch.load(out/f'critic_{int(milestone):04d}.pt',weights_only=False);critic.load_state_dict(c['critic'])
        critic.requires_grad_(False)
        for epsilon in [.005,.02]:
            with torch.no_grad():
                values=[]
                for sign in [1.,-1.]:
                    action=base_actions+sign*epsilon*d;action_sets.append(action)
                    label=f'{milestone}_{epsilon}_{"plus" if sign>0 else "minus"}';labels.append(label)
                    parts,_,_,_=imagined(pi,wm,critic,task,cfg,unpack(probe),actions=action)
                    values.append(parts['bootstrap'].mean())
                model_deltas[f'{milestone}_{epsilon}']=float(values[0]-values[1])
    labels.append('baseline_end');action_sets.append(base_actions)
    # Two replicas per case: 18 cases * 16 states * 2 = 576 physical environments.
    grouped=repeat(probe,2*len(labels));actions=torch.cat([torch.cat([x,x],1) for x in action_sets],1)
    print(f'direction physical batch: {len(grouped["t"])} environments',flush=True)
    physical=physical_returns(pi,task,cfg,grouped,out/'direction_returns.pt',first_actions=actions)
    total=physical['total'].reshape(len(labels),2,16).mean(1);first8=physical['first8'].reshape(len(labels),2,16).mean(1)
    summary=[]
    for milestone in directions:
        for epsilon in [.005,.02]:
            plus=labels.index(f'{milestone}_{epsilon}_plus');minus=labels.index(f'{milestone}_{epsilon}_minus')
            delta=total[plus]-total[minus];short=first8[plus]-first8[minus]
            summary.append(dict(extra_rounds=int(milestone),epsilon_rad=epsilon,physical_return_plus_minus=float(delta.mean()),
                                physical_return_pair_se=float(delta.std()/len(delta)**.5),positive_state_fraction=float((delta>0).float().mean()),
                                physical_reward8_plus_minus=float(short.mean()),physical_tail_plus_minus=float(((delta-short)/cfg['gamma']**8).mean()),
                                model_bootstrap_plus_minus=model_deltas[f'{milestone}_{epsilon}'],per_state_return_differences=delta.tolist()))
    unchanged={name:all(torch.equal(v.cpu(),ck[key][k]) for k,v in model.state_dict().items())
               for name,key,model in [('actor','pi',pi),('world_model','wm',wm)]}
    if not all(unchanged.values()):raise AssertionError('Frozen model changed')
    report=dict(experiment=meta,calibration=results,direction_tests=summary,frozen_models_unchanged=unchanged,
                physical_mc_replica_abs_difference_mean=float((real['total'][:64]-real['total'][64:]).abs().mean()),
                direction_baseline_control_mean_difference=float((total[0]-total[-1]).mean()),
                direction_baseline_control_abs_difference_mean=float((total[0]-total[-1]).abs().mean()),
                probe_test_indices=ids.tolist(),direction_labels=labels)
    json_save(out/'report.json',report);print('COMPLETE',json.dumps({'unchanged':unchanged,'directions':summary}),flush=True)


if __name__=='__main__':main()
