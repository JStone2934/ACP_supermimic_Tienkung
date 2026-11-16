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


def optimization_phase(iteration,cfg,model_ready,critic_warmup_done):
    if iteration<cfg['distill_iters']:return 'distill'
    if iteration<cfg['distill_iters']+cfg['wm_only_iters']:return 'wm'
    if not model_ready:return 'wm_gate'
    return 'critic' if critic_warmup_done<cfg['critic_warmup_iters'] else 'joint'


def check_promotion(checkpoint,reports,previous_level,*,moving=False):
    digest=sha256(checkpoint)
    from . import moving_curriculum,wide_curriculum
    curriculum=moving_curriculum if moving else wide_curriculum
    fixed_count=len(curriculum.grid(previous_level,'cpu'))
    if len(reports)!=2 or len({r.get('seed') for r in reports})!=2:
        raise ValueError('Promotion requires two evaluations with different seeds')
    for r in reports:
        if (r.get('checkpoint_sha256')!=digest or r.get('level')!=previous_level or r.get('passed') is not True
                or r.get('fixed_grid_all_pass') is not True or r.get('random_success_rate',0)<.95
                or r.get('seconds')!=30. or r.get('continuous_without_reset') is not True
                or r.get('reference_free') is not True or r.get('fixed_count')!=fixed_count
                or r.get('random_count')!=100
                or r.get('scope')!=('Intermediate moving curriculum only' if moving else 'Intermediate curriculum only')):
            raise ValueError('Promotion report does not certify this checkpoint and stage')
        scenarios=r.get('scenarios',[])
        if (len(scenarios)!=fixed_count+100 or not all(s.get('passed') is True for s in scenarios[:fixed_count])
                or sum(s.get('passed') is True for s in scenarios[fixed_count:])<95):
            raise ValueError('Promotion requires actual per-scenario full-duration evidence')
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
    p.add_argument('--log-dir',required=True,type=Path);p.add_argument('--level',type=int)
    p.add_argument('--promotion-reports',nargs=2,type=Path);p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--balance-foot-motion-weight',type=float,
                   help='Explicit balance-recovery reward update before any critic/actor training; unchanged value allowed on later resume')
    p.add_argument('--actor-value-gradient',type=float,
                   help='Controlled ACP value-gradient ablation in [0,1], starting before actor training; physical acceptance unchanged')
    p.add_argument('--horizon',type=int,choices=(8,16,32),help='Explicit pre-actor planning-horizon comparison')
    p.add_argument('--compile-training',action='store_true',help='Use numerically verified compiled WM loss and checkpointed imagination')
    p.add_argument('--soft-termination-mode',choices=('legacy_sigmoid','compact'),
                   help='Explicit imagined-termination correction before critic/actor training; physical termination unchanged')
    args=p.parse_args();source=args.resume or args.init
    source_digest=sha256(source)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    if sha256(source)!=source_digest:raise ValueError('Source checkpoint changed during load')
    if ck.get('task') not in ('wide_aiming_v1','moving_aiming_v1') or ck.get('smoke'):
        raise ValueError('Requires an identified, non-smoke wide-task initialization or training checkpoint')
    moving=ck['task']=='moving_aiming_v1'
    from . import wide_curriculum,moving_curriculum
    from .aiming_moving import MovingAimingTask
    curriculum=moving_curriculum if moving else wide_curriculum
    stages=curriculum.STAGES;task_class=MovingAimingTask if moving else WideAimingTask
    if args.resume and ck.get('stage')!='training':raise ValueError('Resume requires a training checkpoint')
    cfg=dict(ck['config']);level=cfg.get('curriculum_level',0) if args.level is None else args.level
    if args.soft_termination_mode is not None:
        updates=ck.get('distributed',{}).get('updates',{})
        if args.soft_termination_mode!=cfg.get('soft_termination_mode','legacy_sigmoid') and (updates.get('critic',0) or updates.get('pi',0)):
            raise ValueError('Change imagined termination before any critic or actor training')
        cfg['soft_termination_mode']=args.soft_termination_mode
    if args.horizon is not None:
        if args.horizon!=cfg['horizon'] and ck.get('distributed',{}).get('updates',{}).get('pi',0):
            raise ValueError('Start a planning-horizon comparison before actor updates')
        cfg['horizon']=args.horizon
        cfg['validation_horizons']=[h for h in (1,8,16,32) if h<=args.horizon]
    if args.compile_training:cfg['compile_training']=True
    if args.actor_value_gradient is not None:
        import math
        weight=args.actor_value_gradient
        if not args.resume or not math.isfinite(weight) or not 0<=weight<=1:
            raise ValueError('Value-gradient control requires explicit resume and a finite weight in[0,1]')
        if weight!=cfg.get('actor_value_gradient',1.) and ck.get('distributed',{}).get('updates',{}).get('pi',0):
            raise ValueError('Start a value-gradient comparison before actor updates')
        cfg['actor_value_gradient']=weight
    if args.balance_foot_motion_weight is not None:
        import math
        weight=args.balance_foot_motion_weight
        if not args.resume or moving or level!=0 or cfg.get('initialization_mode')!='balance_recovery' or not math.isfinite(weight) or weight<0:
            raise ValueError('Foot-motion reward change requires an explicit finite balance-recovery resume at level0')
        old=cfg.get('reward_weights',{}).get('foot_motion',0.)
        updates=ck.get('distributed',{}).get('updates',{})
        if weight!=old and (updates.get('critic',0) or updates.get('pi',0)):
            raise ValueError('Do not change the reward under an already-trained critic/actor')
        cfg['reward_weights']=dict(cfg.get('reward_weights',{}),foot_motion=weight)
    if level not in range(len(stages)):raise ValueError('Invalid curriculum level')
    previous=cfg.get('curriculum_level',0)
    if args.resume and level!=previous:raise ValueError('Use explicit --init when advancing a stage')
    if level!=previous:
        if level!=previous+1 or not args.promotion_reports:raise ValueError('Stages advance one at a time after physical gates')
        check_promotion(source,[json.loads(x.read_text()) for x in args.promotion_reports],previous,moving=moving)
    if ck.get('stage')=='initialization' and level!=0:raise ValueError('The first stage is balance')
    cfg.update(curriculum_level=level,iterations=args.iterations,
               switch_interval_steps=250 if moving else (500 if level==len(stages)-1 else None))
    if args.iterations<1:raise ValueError('Positive iteration limit required')
    defaults=dict(distill_iters=1000 if ck.get('stage')=='initialization' and not moving else 0,
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
    # Match the independently checked eager/compiled numerical precision.
    cfg.setdefault('matmul_tf32',False)
    torch.backends.cuda.matmul.allow_tf32=cfg['matmul_tf32']
    execution=argparse.Namespace(**cfg,log_dir=str(args.log_dir),device=None,dist_timeout=1800)
    torch.set_num_threads(2);rt=Runtime(execution)
    local=dict(cfg)
    for k in ('num_envs','wm_batch','pi_batch'):local[k]=getattr(execution,k)
    validate_checkpoint(ck,task_class('cpu',ck['config']))
    task=task_class(rt.device,cfg)
    pi=FullbodyPolicy(task.robot).to(rt.device);pi.load_state_dict(ck['pi'])
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(rt.device);wm.load_state_dict(ck['wm']);freeze_normalizers(pi,wm)
    critic=Critic(task.critic_dim).to(rt.device)
    if args.init and ck.get('stage')=='training':
        critic.load_state_dict(ck['critic'])
    op=torch.optim.Adam(pi.parameters(),lr=cfg['actor_lr']);ow=torch.optim.Adam(wm.parameters(),lr=cfg['wm_lr']);ov=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr'])
    start=0
    critic_warmup_done=0
    if args.resume:
        critic.load_state_dict(ck['critic'])
        for optimizer,state in zip((op,ow,ov),ck['optimizers']):optimizer.load_state_dict(state)
        start=ck['iteration']
        critic_warmup_done=ck.get('critic_warmup_done',min(cfg['critic_warmup_iters'],
            ck.get('distributed',{}).get('updates',{}).get('critic',0)//(cfg['critic_epochs']*cfg['critic_minibatches'])))
    rt.register(dict(pi=pi,wm=wm,critic=critic),dict(pi=op,wm=ow,critic=ov))
    if args.resume:rt.restore(ck)
    teacher=None
    if start<cfg['distill_iters']:
        source_tracker=cfg['source']['checkpoint']
        if sha256(source_tracker)!=cfg['source']['sha256']:raise ValueError('Teacher checkpoint identity changed')
        original=torch.load(source_tracker,map_location=rt.device,weights_only=False)
        if sha256(source_tracker)!=cfg['source']['sha256']:raise ValueError('Teacher changed during load')
        teacher_pose=bool(original['ctc_cfg'].get('reference_pose',False))
        teacher=Policy(task.robot.obs_dim,93+(task.robot.num_joints if teacher_pose else 0),hist=10,robot=task.robot,
                       pose_action_skip=original['args'].get('pose_action_skip',False)).to(rt.device)
        teacher.load_state_dict(original['pi']);teacher.eval().requires_grad_(False)
    sampler=lambda n:curriculum.sample(level,n,rt.device)
    env=FullbodySim(task,local['num_envs'],sampler)
    replay=FullbodyReplay(cfg['buffer_steps'],env.N,rt.device,extra=REPLAY_FIELDS,robot=task.robot)
    history=env.observe()[:,None].repeat(1,10,1)
    wm_objective=torch.compile(wm_loss,dynamic=False,options={'max_fusion_size':32}) if cfg.get('compile_training') else wm_loss
    validation_env=FullbodySim(task,cfg['wm_validation_envs'],sampler) if rt.main else None
    noise=task.alpha.new_full((29,),cfg['body_noise_rad']);noise[task.arm_ids]=cfg['arm_noise_rad']
    began=time.monotonic();model_ready=False;iteration=start

    @torch.no_grad()
    def collect(steps,teacher_fraction):
        nonlocal history
        statistics=torch.zeros(6,device=rt.device)
        for _ in range(steps):
            if moving:
                moving_curriculum.apply_commands(env,level)
            elif level==len(stages)-1:
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
            replay.add(tr)
            age=(tr['ep_step']+1).float()*task.dt
            statistics+=torch.stack((tr['r'].mean(),tr['terminated'].float().mean(),
                tr['timeout'].float().mean(),age.mean(),tr['done'].float().sum(),
                (age*tr['done']).sum()))
            history=torch.cat((history[:,1:],env.observe()[:,None]),1)
        totals=rt.sum(statistics);means=totals[:4]/(steps*rt.world_size)
        return dict(sim_reward=float(means[0]),physical_termination_fraction=float(means[1]),
            physical_timeout_fraction=float(means[2]),mean_physical_episode_age_s=float(means[3]),
            completed_physical_episodes=int(totals[4]),
            mean_completed_physical_episode_s=float(totals[5]/totals[4]) if totals[4]>0 else None)

    collect(cfg['initial_collect_steps'],1. if teacher is not None else 0.)
    for iteration in range(start,cfg['iterations']):
        distilling=iteration<cfg['distill_iters']
        teacher_fraction=max(0.,1.-iteration/max(1,cfg['distill_iters'])) if distilling else 0.
        collection=collect(cfg['collect_steps'],teacher_fraction)
        for _ in range(cfg['wm_batches']):
            ow.zero_grad(set_to_none=True);loss=wm_objective(wm,task,replay.sample_windows(local['wm_batch'],cfg['wm_window']))
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
                            callback=(lambda env:moving_curriculum.apply_commands(env,level,validation=True)) if moving else None
                            report=validate_world_model(pi,wm,task,sampler,cfg,args.log_dir/'wm_validation'/f'{iteration:06d}',env=validation_env,command_callback=callback)
                        ready=report['passed']
                    model_ready=bool(rt.sum(int(ready)))
                if model_ready or args.smoke:
                    phase=optimization_phase(iteration,cfg,True,critic_warmup_done)
                    actor=phase=='joint'
                    metrics=update_actor(pi,wm,critic,task,replay,op,ov,local,train_actor=actor,runtime=rt)
                    if not actor:critic_warmup_done+=1
                else:phase='wm_gate'
        if rt.skipped:raise FloatingPointError(f'Nonfinite optimizer update: {rt.skipped}')
        if (iteration+1)%cfg['log_every']==0:
            row=dict(iteration=iteration+1,phase=phase,level=level,stage=stages[level]['name'],
                **collection,wm_loss=float(rt.average(loss.detach())),model_ready=model_ready,
                critic_warmup_done=critic_warmup_done,
                updates=rt.updates,elapsed_s=time.monotonic()-began,**metrics)
            if rt.main:
                json_save(args.log_dir/'status.json',row)
                with (args.log_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            rt.print(json.dumps(row),flush=True)
        if (iteration+1)%cfg['ckpt_every']==0 or iteration+1==cfg['iterations']:
            artifact=dict(task=ck['task'],stage='training',smoke=args.smoke,
                robot_metadata=task.robot.metadata(),task_config=task.metadata(),config=cfg,
                pi=pi.state_dict(),wm=wm.state_dict(),critic=critic.state_dict(),
                optimizers=[x.state_dict() for x in (op,ow,ov)],iteration=iteration+1,
                model_ready=model_ready,source_checkpoint=dict(path=str(source.resolve()),sha256=source_digest),
                critic_warmup_done=critic_warmup_done,
                provenance=ck.get('provenance'),promotion_reports=[str(x) for x in args.promotion_reports or []])
            artifact['training_arguments']=vars(args)
            rt.save(artifact,numbered=f'ckpt_{iteration+1:06d}.pt',full=True)
    rt.close()


if __name__=='__main__':main()
