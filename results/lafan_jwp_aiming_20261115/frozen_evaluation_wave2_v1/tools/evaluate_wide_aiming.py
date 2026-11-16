"""75 fixed targets plus 100 independently sampled target-switch episodes.

No reference motions, no automatic resets, no root supports. This evaluator is
separate from curriculum rewards and never promotes a checkpoint by itself.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_targets import fixed_target_grid
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.wide_aiming_validation import episode_metrics
from supermimic.acp.robots import sha256


@torch.no_grad()
def evaluate(policy, task, out, *, seed=0, checkpoint=None):
    device=task.alpha.device
    checkpoint_hash=sha256(checkpoint) if checkpoint else None
    grid=fixed_target_grid(device)
    generator=torch.Generator(device=device).manual_seed(seed)
    random=torch.rand((3,100,4),generator=generator,device=device)
    random[...,0]=2.+2.*random[...,0]
    random[...,1]=(2.*random[...,1]-1.)*torch.pi/3
    random[...,2]=(2.*random[...,2]-1.)*torch.pi/6
    random[...,3]=.05
    parameters=torch.cat((grid[None].expand(3,-1,-1),random),1)
    n=parameters.shape[1]
    env=FullbodySim(task,n,lambda count:(parameters[0],task.weights['aim']),eval_mode=True)
    initial=tuple(x.clone() for x in env.state())
    failed=torch.zeros(n,dtype=torch.bool,device=device)
    bad_action=torch.zeros_like(failed)
    history=None
    records={k:[] for k in ('root','feet','tilt','error','finite_state','targets')}
    policy.eval()

    def record():
        state=env.state()
        feet=task.kin.forward(*state)[0][:,task.feet]
        error,*_=task.metrics(state,env.context)
        finite=torch.stack([x.flatten(1).isfinite().all(-1) for x in state]).all(0)&~bad_action
        values=dict(root=state[0],feet=feet,tilt=torch.rad2deg(torch.acos(state[1][:,2,2].clamp(-1,1))),
                    error=error,finite_state=finite,targets=env.context['aim_target_w'])
        for key,value in values.items():records[key].append(value.detach().cpu().numpy().copy())

    for step in range(1500):
        if step in (500,1000):
            # Sample each target in the original episode frame. The target must
            # not follow the robot after turning or after an adjustment step.
            ctx=task.context(initial,parameters[step//500],task.weights['aim'])
            for key in ('aim_target_w','aim_normal_w','aim_tolerance'):
                env.context[key].copy_(ctx[key])
        record()
        obs=env.observe()
        history=obs[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        action=policy(history,task.command(env.state(),env.context))
        bad_action |= ~action.isfinite().all(-1)
        action=torch.where(action.isfinite(),action,torch.zeros_like(action))
        transition=env.step(action)
        failed |= transition['terminated']|bad_action
    record()
    arrays={k:np.stack(values) for k,values in records.items()}
    failures=failed.cpu().numpy()
    scenarios=[episode_metrics(*(arrays[k][:,i] for k in ('root','feet','tilt','error','finite_state')),
                               failures[i],switching=i>=len(grid)) for i in range(n)]
    fixed_pass=all(s['passed'] for s in scenarios[:len(grid)])
    random_rate=sum(s['passed'] for s in scenarios[len(grid):])/100
    if checkpoint and sha256(checkpoint)!=checkpoint_hash:
        raise ValueError('Checkpoint changed during physical evaluation')
    result=dict(passed=bool(fixed_pass and random_rate>=.95),fixed_grid_all_pass=fixed_pass,
                random_success_rate=random_rate,fixed_count=len(grid),random_count=100,
                seed=seed,seconds=30.,continuous_without_reset=True,reference_free=True,
                checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=checkpoint_hash,
                task_config=task.metadata(),robot_metadata=task.robot.metadata(),
                target_parameters=parameters.cpu().tolist(),scenarios=scenarios)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    json_save(out,result)
    np.savez_compressed(out.with_suffix('.npz'),**arrays,terminated=failures)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    task=WideAimingTask(args.device,ck['config']);validate_checkpoint(ck,task)
    policy=FullbodyPolicy(task.robot).to(args.device);policy.load_state_dict(ck['pi'])
    report=evaluate(policy,task,args.out,seed=args.seed,checkpoint=args.checkpoint)
    print(json.dumps({k:report[k] for k in ('passed','fixed_grid_all_pass','random_success_rate')}))
