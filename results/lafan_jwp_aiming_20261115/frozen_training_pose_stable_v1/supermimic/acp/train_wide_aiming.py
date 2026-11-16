"""Two-GPU ACP training for one explicitly evaluated wide-aiming stage.

Starts from the gated JWP tracker migration. The first stage distills its fixed
support command into the reference-free interface. Later stages require two
physical promotion reports. This runner never declares final acceptance.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import torch
from .aiming_fullbody import FullbodyPolicy,FullbodyReplay,CONTEXT_FIELDS,REPLAY_FIELDS,flat_boxes,validate_checkpoint
from .aiming_fullbody_env import FullbodySim
from .aiming_fullbody_io import json_save
from .aiming_fullbody_validation import validate_world_model
from .aiming_wide import WideAimingTask
from .distributed import Runtime
from .mdp import unflatten_state
from .networks import WorldModel,Critic,Policy
from .robots import sha256
from .standing import freeze_normalizers
from .tracking_transfer import fixed_support_command
from .train_aiming_fullbody import wm_loss,update_actor
from .wide_curriculum import STAGES,sample


def check_promotion(checkpoint,reports,previous_level):
    digest=sha256(checkpoint)
    if len(reports)!=2 or len({r.get('seed') for r in reports})!=2:
        raise ValueError('Promotion requires two evaluations with different seeds')
    for r in reports:
        if (r.get('checkpoint_sha256')!=digest or r.get('level')!=previous_level or r.get('passed') is not True
                or r.get('fixed_grid_all_pass') is not True or r.get('random_success_rate',0)<.95):
            raise ValueError('Promotion report does not certify this checkpoint and stage')
    return digest


def switch_targets(task,context,ids,level):
    """Keep the original episode frame even after the robot has turned."""
    if not len(ids):return
    initial=task.initial_state(len(ids))
    initial=(context['root_anchor_w'][ids],*initial[1:])
    params,weight=sample(level,len(ids),task.alpha.device)
    new=task.context(initial,params,weight)
    for key in ('aim_target_w','aim_normal_w','aim_tolerance'):
        context[key][ids]=new[key]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--init',type=Path);g.add_argument('--resume',type=Path)
    p.add_argument('--log-dir',required=True,type=Path);p.add_argument('--level',type=int,choices=range(len(STAGES)))
    p.add_argument('--promotion-reports',nargs=2,type=Path);p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();source=args.resume or args.init
    source_digest=sha256(source)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    if sha256(source)!=source_digest:raise ValueError('Source checkpoint changed during load')
    if ck.get('task')!='wide_aiming_v1' or ck.get('smoke'):
        raise ValueError('Requires an identified, non-smoke wide-task initialization or training checkpoint')
    if args.resume and ck.get('stage')!='training':raise ValueError('Resume requires a training checkpoint')
    cfg=dict(ck['config']);level=cfg.get('curriculum_level',0) if args.level is None else args.level
    previous=cfg.get('curriculum_level',0)
    if args.resume and level!=previous:raise ValueError('Use explicit --init when advancing a stage')
    if level!=previous:
        if level!=previous+1 or not args.promotion_reports:raise ValueError('Stages advance one at a time after physical gates')
        check_promotion(source,[json.loads(x.read_text()) for x in args.promotion_reports],previous)
    if ck.get('stage')=='initialization' and level!=0:raise ValueError('The first stage is balance')
    cfg.update(curriculum_level=level,iterations=args.iterations,
               switch_interval_steps=500 if level==len(STAGES)-1 else None)
    if args.iterations<1:raise ValueError('Positive iteration limit required')
    defaults=dict(distill_iters=1000 if ck.get('stage')=='initialization' else 0,
        distill_lr=1e-4,wm_validation_every=100,wm_validation_steps=128,wm_validation_envs=8,
        wm_gate_root_m=.05,wm_gate_feet_m=.025,wm_gate_tip_m=.1,validation_horizons=[1,8],enforce_wm_gate=True,
        verify_sync=True,log_every=25)
    for k,v in defaults.items():cfg.setdefault(k,v)
    if args.init and ck.get('stage')=='training':
        cfg['distill_iters']=0
        cfg['wm_only_iters']=100
    if args.smoke:
        cfg.update(num_envs=32,wm_batch=32,pi_batch=32,buffer_steps=64,initial_collect_steps=16,
            wm_batches=1,wm_only_iters=0,critic_warmup_iters=0,distill_iters=0,critic_epochs=1,
            critic_minibatches=1,iterations=2,ckpt_every=2,log_every=1,wm_validation_steps=16,wm_validation_envs=2)
    args.log_dir.mkdir(parents=True,exist_ok=True)
    lock=None
    if int(os.environ.get('RANK','0'))==0:
        lock=(args.log_dir/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.resume and (args.log_dir/'last.pt').exists():raise FileExistsError('Use --resume for this output')
    execution=argparse.Namespace(**cfg,log_dir=str(args.log_dir),device=None,dist_timeout=1800)
    torch.set_num_threads(2);rt=Runtime(execution)
    local=dict(cfg)
    for k in ('num_envs','wm_batch','pi_batch'):local[k]=getattr(execution,k)
    validate_checkpoint(ck,WideAimingTask('cpu',ck['config']))
    task=WideAimingTask(rt.device,cfg)
    pi=FullbodyPolicy(task.robot).to(rt.device);pi.load_state_dict(ck['pi'])
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(rt.device);wm.load_state_dict(ck['wm']);freeze_normalizers(pi,wm)
    critic=Critic(task.critic_dim).to(rt.device)
    op=torch.optim.Adam(pi.parameters(),lr=cfg['actor_lr']);ow=torch.optim.Adam(wm.parameters(),lr=cfg['wm_lr']);ov=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr'])
    start=0
    if args.resume:
        critic.load_state_dict(ck['critic'])
        for optimizer,state in zip((op,ow,ov),ck['optimizers']):optimizer.load_state_dict(state)
        start=ck['iteration']
    rt.register(dict(pi=pi,wm=wm,critic=critic),dict(pi=op,wm=ow,critic=ov))
    if args.resume:rt.restore(ck)
    teacher=None
    if start<cfg['distill_iters']:
        source_tracker=cfg['source']['checkpoint']
        if sha256(source_tracker)!=cfg['source']['sha256']:raise ValueError('Teacher checkpoint identity changed')
        original=torch.load(source_tracker,map_location=rt.device,weights_only=False)
        if sha256(source_tracker)!=cfg['source']['sha256']:raise ValueError('Teacher changed during load')
        teacher_pose=bool(original['ctc_cfg'].get('reference_pose',False))
        teacher=Policy(task.robot.obs_dim,93+(task.robot.num_joints if teacher_pose else 0),hist=10,robot=task.robot).to(rt.device)
        teacher.load_state_dict(original['pi']);teacher.eval().requires_grad_(False)
    sampler=lambda n:sample(level,n,rt.device)
    env=FullbodySim(task,local['num_envs'],sampler)
    replay=FullbodyReplay(cfg['buffer_steps'],env.N,rt.device,extra=REPLAY_FIELDS,robot=task.robot)
    history=env.observe()[:,None].repeat(1,10,1)
    validation_env=FullbodySim(task,cfg['wm_validation_envs'],sampler) if rt.main else None
    noise=task.alpha.new_full((29,),cfg['body_noise_rad']);noise[task.arm_ids]=cfg['arm_noise_rad']
    began=time.monotonic();model_ready=False;iteration=start

    @torch.no_grad()
    def collect(steps,teacher_fraction):
        nonlocal history
        rewards=[]
        for _ in range(steps):
            if level==len(STAGES)-1:
                ids=((env.ep_step>0)&(env.ep_step%500==0)).nonzero().flatten()
                switch_targets(task,env.context,ids,level)
            obs=env.observe()
            history=torch.where((env.ep_step==0)[:,None,None],obs[:,None].expand_as(history),history)
            raw=pi(history,task.command(env.state(),env.context))
            if teacher is not None and teacher_fraction>0:
                command=fixed_support_command(task,env.state(),env.context)
                if teacher_pose:command=torch.cat((command,task.initial_state(env.N)[4]),-1)
                target=teacher(history,command)
                raw=torch.where((torch.rand(env.N,device=rt.device)<teacher_fraction)[:,None],target,raw)
            fraction=max(cfg.get('noise_min_fraction',.1),1.-iteration/cfg.get('noise_decay_updates',10000))
            raw=raw+torch.randn_like(raw)*noise/task.alpha*fraction
            tr=env.step(raw)
            finite=torch.stack([tr[k].isfinite().all() for k in ('s','s1','a','r')]).all()
            if int(rt.sum((~finite).int())):
                torch.save({k:v.cpu() for k,v in tr.items() if torch.is_tensor(v)},args.log_dir/f'nonfinite_rank{rt.rank}.pt')
                raise FloatingPointError('Nonfinite physical transition')
            replay.add(tr);rewards.append(tr['r'].mean())
            history=torch.cat((history[:,1:],env.observe()[:,None]),1)
        return float(rt.average(torch.stack(rewards).mean()))

    collect(cfg['initial_collect_steps'],1. if teacher is not None else 0.)
    for iteration in range(start,cfg['iterations']):
        distilling=iteration<cfg['distill_iters']
        teacher_fraction=max(0.,1.-iteration/max(1,cfg['distill_iters'])) if distilling else 0.
        reward=collect(cfg['collect_steps'],teacher_fraction)
        for _ in range(cfg['wm_batches']):
            ow.zero_grad(set_to_none=True);loss=wm_loss(wm,task,replay.sample_windows(local['wm_batch'],cfg['wm_window']))
            loss.backward();rt.step(ow,wm,'wm',max_norm=1.)
        metrics={};phase='wm'
        if distilling:
            phase='distill'
            batch,histories=replay.sample_starts(local['pi_batch'],10)
            state=unflatten_state(batch['s']);ctx={k:batch[k] for k in CONTEXT_FIELDS}
            with torch.no_grad():
                command=fixed_support_command(task,state,ctx)
                if teacher_pose:command=torch.cat((command,task.initial_state(len(state[0]))[4]),-1)
                target=teacher(histories['o'],command)
                target,_=task.compose_action(target,batch['xi'])
            op.param_groups[0]['lr']=cfg['distill_lr'];op.zero_grad(set_to_none=True)
            prediction=pi(histories['o'],task.command(state,ctx))
            imitation=((prediction-target)*task.alpha).square().mean();imitation.backward()
            rt.step(op,pi,'distill',max_norm=1.);metrics['distill_joint_target_mse']=float(rt.average(imitation.detach()))
        else:
            op.param_groups[0]['lr']=cfg['actor_lr']
            if iteration>=cfg['distill_iters']+cfg['wm_only_iters']:
                if iteration%cfg['wm_validation_every']==0 or not model_ready and iteration==start:
                    ready=False
                    if rt.main:
                        with torch.random.fork_rng(devices=[task.alpha.device.index]):
                            report=validate_world_model(pi,wm,task,sampler,cfg,args.log_dir/'wm_validation'/f'{iteration:06d}',env=validation_env)
                        ready=report['passed']
                    model_ready=bool(rt.sum(int(ready)))
                if model_ready or args.smoke:
                    actor=iteration>=cfg['distill_iters']+cfg['wm_only_iters']+cfg['critic_warmup_iters']
                    phase='joint' if actor else 'critic'
                    metrics=update_actor(pi,wm,critic,task,replay,op,ov,local,train_actor=actor,runtime=rt)
                else:phase='wm_gate'
        if rt.skipped:raise FloatingPointError(f'Nonfinite optimizer update: {rt.skipped}')
        if (iteration+1)%cfg['log_every']==0:
            row=dict(iteration=iteration+1,phase=phase,level=level,stage=STAGES[level]['name'],
                sim_reward=reward,wm_loss=float(rt.average(loss.detach())),model_ready=model_ready,
                updates=rt.updates,elapsed_s=time.monotonic()-began,**metrics)
            if rt.main:
                json_save(args.log_dir/'status.json',row)
                with (args.log_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            rt.print(json.dumps(row),flush=True)
        if (iteration+1)%cfg['ckpt_every']==0 or iteration+1==cfg['iterations']:
            artifact=dict(task='wide_aiming_v1',stage='training',smoke=args.smoke,
                robot_metadata=task.robot.metadata(),task_config=task.metadata(),config=cfg,
                pi=pi.state_dict(),wm=wm.state_dict(),critic=critic.state_dict(),
                optimizers=[x.state_dict() for x in (op,ow,ov)],iteration=iteration+1,
                model_ready=model_ready,source_checkpoint=dict(path=str(source.resolve()),sha256=source_digest),
                provenance=ck.get('provenance'),promotion_reports=[str(x) for x in args.promotion_reports or []])
            rt.save(artifact,numbered=f'ckpt_{iteration+1:06d}.pt',full=True)
    rt.close()


if __name__=='__main__':main()
