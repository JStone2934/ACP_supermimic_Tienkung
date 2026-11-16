"""Physical plant baseline: fixed PD targets from training poses, not a learned policy."""
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.rotations import quat_to_mat
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.robots import get_robot
from supermimic.acp.wide_curriculum import sample
from supermimic.acp.wide_aiming_validation import episode_metrics


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(214)
    b=Path('results/lafan_jwp_aiming_20261115');device='cuda:1';robot=get_robot('tienkung3_29dof')
    seeds=json.loads((b/'jwp_training_quiet_seeds.json').read_text())['seeds']
    seeds=seeds+[dict(clip='robot_default_pose_baseline',frame=None,initial_q=list(robot.default_q),
                     initial_height=robot.stand_height,initial_quat=[1.,0.,0.,0.])]
    copies=4;n=len(seeds)*copies
    q=torch.tensor([s['initial_q'] for s in seeds],device=device).repeat_interleave(copies,0)
    heights=torch.tensor([s['initial_height'] for s in seeds],device=device).repeat_interleave(copies,0)
    rotations=quat_to_mat(torch.tensor([s['initial_quat'] for s in seeds],device=device)).repeat_interleave(copies,0)
    cfg=dict(seeds[15],physics_substeps=16,reset_position_noise_m=.01,reset_rotation_noise_rad=.015,reset_joint_noise_rad=.03)
    task=WideAimingTask(device,cfg);original=task.initial_state
    def initial(count):
        assert count==n
        p,_,v,w,_,qd=original(count);p[:,2]=heights
        return p,rotations.clone(),v,w,q.clone(),qd
    task.initial_state=initial
    env=FullbodySim(task,n,lambda count:sample(0,count,device),eval_mode=True)
    raw=(q-task.mdp.default_q)/task.alpha
    records={key:[] for key in ['root','feet','tilt','finite_state']};failed=torch.zeros(n,dtype=torch.bool,device=device)
    for t in range(1501):
        state=env.state();bp,_,_,_=task.kin.forward(*state)
        values=dict(root=state[0],feet=bp[:,task.feet],tilt=torch.rad2deg(torch.acos(state[1][:,2,2].clamp(-1,1))),
                    finite_state=torch.stack([x.flatten(1).isfinite().all(-1) for x in state]).all(0))
        for key,value in values.items():records[key].append(value.cpu().numpy().copy())
        if t<1500:failed|=env.step(raw)['terminated']
    arrays={k:np.stack(v) for k,v in records.items()};terminated=failed.cpu().numpy();rows=[]
    for i,seed in enumerate(seeds):
        scenarios=[]
        for j in range(i*copies,(i+1)*copies):
            result=episode_metrics(arrays['root'][:,j],arrays['feet'][:,j],arrays['tilt'][:,j],np.zeros(1501),
                arrays['finite_state'][:,j],terminated[j],aim_required=False)
            result['foot_drift_after3s_m']=float(np.linalg.norm(arrays['feet'][150:,j]-arrays['feet'][150,j],axis=-1).max())
            scenarios.append(result)
        row=dict(index=i,clip=seed['clip'],frame=seed['frame'],full_success_rate=float(np.mean([s['passed'] for s in scenarios])),
                 balance_only_rate=float(np.mean([s['standing_pass'] for s in scenarios])),
                 median_max_tilt_deg=float(np.median([s['max_tilt_deg'] for s in scenarios])),
                 median_foot_drift_m=float(np.median([s['foot_drift_after3s_m'] for s in scenarios])),scenarios=scenarios)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='scenarios'}),flush=True)
    report=dict(scope=__doc__,policy_acceptance=False,formal_training=False,seconds=30,continuous_without_reset=True,
        reset_seed=214,copies_per_pose=copies,training_source_only=True,robot_metadata=robot.metadata(),
        physics_substeps=16,reset_noise=dict(position_m=.01,rotation_rad=.015,joint_rad=.03),rows=rows)
    (b/'static_pose_pd_baseline.json').write_text(json.dumps(report,indent=2));np.savez_compressed(b/'static_pose_pd_baseline.npz',**arrays,terminated=terminated)


if __name__=='__main__':main()
