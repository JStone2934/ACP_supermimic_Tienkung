"""Independent30s moving/aiming evaluation with explicit velocity and target switches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_moving import MovingAimingTask
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.moving_aiming_validation import moving_episode_metrics
from supermimic.acp import moving_curriculum as curriculum
from supermimic.acp.robots import sha256
from supermimic.rotations import mat_to_quat
from supermimic.acp.chest_posture import apply_posture_acceptance


@torch.no_grad()
def evaluate(policy,task,out,*,seed=0,checkpoint=None,level=None):
    # Reset perturbations use the global Torch RNG, independently of the
    # target sampler's generator. Seed both so reports reproduce the reset.
    torch.manual_seed(seed)
    stage=len(curriculum.STAGES)-1 if level is None else level;device=task.alpha.device
    digest=sha256(checkpoint) if checkpoint else None
    generator=torch.Generator(device=device).manual_seed(seed)
    grid=curriculum.grid(stage,device)
    random=torch.stack([curriculum.sample(stage,100,device,generator)[0] for _ in range(3)])
    parameters=torch.cat((grid[None].expand(3,-1,-1),random),1);n=parameters.shape[1]
    env=FullbodySim(task,n,lambda count:(parameters[0],4.),eval_mode=True)
    failed=torch.zeros(n,dtype=torch.bool,device=device);bad_action=torch.zeros_like(failed);history=None
    keys=('root','root_quat','q','velocity','command','path_goal','tilt','error','finite_state','targets','target_normals')
    records={k:[] for k in keys};policy.eval();speed_cap=curriculum.STAGES[stage]['speed']
    if task.chest_posture.enabled:
        records.update({k:[] for k in ('chest_tilt_deg','chest_forward_deg','chest_lateral_deg','chest_limit_deg','chest_posture_valid')})

    def record():
        state=env.state();error,*_=task.metrics(state,env.context)
        values=dict(root=state[0],root_quat=mat_to_quat(state[1]),q=state[4],velocity=state[2],
            command=env.context['velocity_command'],path_goal=env.context['root_anchor_w'],
            tilt=torch.rad2deg(torch.acos(state[1][:,2,2].clamp(-1,1))),error=error,
            finite_state=torch.stack([x.flatten(1).isfinite().all(-1) for x in state]).all(0)&~bad_action,
            targets=env.context['aim_target_w'],target_normals=env.context['aim_normal_w'])
        if task.chest_posture.enabled:
            rotations=task.kin.forward(*state)[1]
            posture,_=task.chest_posture.measure(rotations[:,task.torso],env.context['aim_normal_w'])
            values.update(posture)
        for k,x in values.items():records[k].append(x.cpu().numpy().copy())
    for step in range(1500):
        if step%250==0:
            speed=speed_cap*curriculum.SPEED_FRACTIONS[step//250];yaw=env.context['yaw_anchor'][:,0]
            env.context['velocity_command']=torch.stack((speed*yaw.cos(),speed*yaw.sin(),torch.zeros_like(yaw)),-1)
        if step in (500,1000):
            curriculum.retarget(task,env.state(),env.context,parameters[step//500],torch.arange(n,device=device))
        record();obs=env.observe()
        history=obs[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        action=policy(history,task.command(env.state(),env.context));bad_action|=~action.isfinite().all(-1)
        transition=env.step(torch.where(action.isfinite(),action,torch.zeros_like(action)))
        failed|=transition['terminated']|bad_action
    record();arrays={k:np.stack(v) for k,v in records.items()};failures=failed.cpu().numpy()
    scenarios=[moving_episode_metrics(*(arrays[k][:,i] for k in ('root','velocity','command','path_goal','tilt','error','finite_state')),
                                      failures[i],maximum_speed=speed_cap) for i in range(n)]
    apply_posture_acceptance(task,arrays,scenarios,lambda i:True)
    fixed=all(x['passed'] for x in scenarios[:len(grid)]);rate=sum(x['passed'] for x in scenarios[len(grid):])/100
    if checkpoint and sha256(checkpoint)!=digest:raise ValueError('Checkpoint changed during evaluation')
    result=dict(passed=bool(fixed and rate>=.95),fixed_grid_all_pass=fixed,random_success_rate=rate,
        fixed_count=len(grid),random_count=100,seed=seed,seconds=30.,continuous_without_reset=True,reference_free=True,
        checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=digest,task_config=task.metadata(),
        robot_metadata=task.robot.metadata(),scenarios=scenarios,target_parameters=parameters.cpu().tolist(),
        level=level,scope='Full moving aiming acceptance' if level is None else 'Intermediate moving curriculum only',
        hit_radius_m=.05,maximum_speed_m_s=speed_cap,velocity_rmse_limit_m_s=.08,online_target_switches=True,
        velocity_profile=list(curriculum.SPEED_FRACTIONS),velocity_phase_seconds=5.,target_phase_seconds=10.)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);json_save(out,result)
    np.savez_compressed(out.with_suffix('.npz'),**arrays,terminated=failures,control_dt=.02,hit_radius_m=.05,
                        robot_metadata=json.dumps(task.robot.metadata()),checkpoint_sha256=digest)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--level',type=int,choices=range(len(curriculum.STAGES)),help='Intermediate speed gate; omit for full acceptance')
    a=p.parse_args();torch.set_num_threads(2);ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if ck.get('task')!='moving_aiming_v1' or ck.get('smoke'):raise ValueError('Expected a formal moving checkpoint')
    if a.level is not None and ck['config']['curriculum_level']!=a.level:raise ValueError('Stage does not match checkpoint')
    task=MovingAimingTask(a.device,ck['config']);validate_checkpoint(ck,task)
    policy=FullbodyPolicy(task.robot).to(a.device);policy.load_state_dict(ck['pi'])
    result=evaluate(policy,task,a.out,seed=a.seed,checkpoint=a.checkpoint,level=a.level)
    print(json.dumps({k:result[k] for k in ('passed','fixed_grid_all_pass','random_success_rate')}))
