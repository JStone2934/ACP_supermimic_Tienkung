"""Center-pose diagnostics from an actually balanced rollout, changing distal arm joints only."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.train_wide_aiming import check_promotion
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256
from supermimic.rotations import quat_to_mat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--balance-reports',type=Path,nargs=2,required=True)
    p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    digest=check_promotion(a.source,[json.loads(x.read_text()) for x in a.balance_reports],0)
    ck=torch.load(a.source,map_location='cpu',weights_only=False);assert not ck.get('smoke')
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    report=json.loads(a.trajectory.with_suffix('.json').read_text());assert report['scenarios'][0]['standing_pass']
    with np.load(a.trajectory) as z:
        task.robot.validate_metadata(json.loads(str(z['robot_metadata'])))
        assert str(z['checkpoint_sha256'])==report['checkpoint_sha256'] and z['q'].shape[0]==1501
        t,i=500,0;p0=torch.tensor(z['root'][t,i])[None];quat=torch.tensor(z['root_quat'][t,i])[None]
        q0=torch.tensor(z['q'][t,i])[None];target=torch.tensor(z['targets'][t,i])[None];normal=torch.tensor(z['target_normals'][t,i])[None]
    state=(p0,quat_to_mat(quat),torch.zeros(1,3),torch.zeros(1,3),q0,torch.zeros(1,29))
    context=dict(aim_target_w=target,aim_normal_w=normal)
    arm=task.arm_ids.tolist();distal=[task.robot.joint_names.index(x) for x in ('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')]
    x=q0[:,distal].clone().requires_grad_();optimizer=torch.optim.Adam([x],lr=.01)
    best=float('inf');solution=q0.clone()
    for _ in range(600):
        q=q0.clone();q[:,distal]=x;s=(*state[:4],q,state[5]);h,d,_=task.hand(s)
        delta=target-h;loss=((d-torch.nn.functional.normalize(delta,dim=-1)).square().sum(-1)*delta.square().sum(-1)).mean()+1e-6*(x-q0[:,distal]).square().sum()
        error=float(task.metrics(s,context)[0].detach().item())
        if error<best:best=error;solution=q.detach().clone()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():x.clamp_(task.mdp.soft_lo[distal],task.mdp.soft_hi[distal])
    assert best<.001,('Distal-only center pose did not converge',best)
    unchanged=[j for j in range(29) if j not in distal];assert torch.equal(solution[:,unchanged],q0[:,unchanged])
    model=task.robot.get_spec().compile();data=mujoco.MjData(model);addresses,dof=task.robot.joint_addresses(model);tcp=model.body('right_tcp_link').id
    a.out.mkdir(parents=True,exist_ok=True)
    provenance=dict(trajectory=str(a.trajectory.resolve()),trajectory_sha256=sha256(a.trajectory),trajectory_checkpoint_sha256=report['checkpoint_sha256'],frame=t,scenario=i,balanced_full30s=True)
    for number,(label,pose,gain) in enumerate((('observed',q0,1.),('distal_ik',solution,1.),('distal_ik_half_gravity',solution,.5)),1):
        path=a.out/f'ckpt_{number:06d}.pt'
        if path.exists():raise FileExistsError(path)
        data.qpos[:3]=p0[0].numpy();data.qpos[3:7]=quat[0].numpy();data.qpos[addresses]=pose[0].numpy();mujoco.mj_forward(model,data)
        h=data.xpos[tcp];direction=-data.xmat[tcp].reshape(3,3)[:,2];n=normal[0].numpy();goal=target[0].numpy()
        denominator=n@direction;distance=n@(goal-h)/denominator
        assert denominator>0 and distance>0
        mujoco_error=float(np.linalg.norm(h+distance*direction-goal))
        torch_error=float(task.metrics((*state[:4],pose,state[5]),context)[0].item())
        assert abs(mujoco_error-torch_error)<1e-4
        gravity=data.qfrc_bias[dof].copy();offset=torch.tensor(gravity/np.asarray(task.robot.kp),dtype=torch.float32)
        desired=pose[0].clone();desired[arm]+=gain*offset[arm]
        assert ((desired[arm]>=task.mdp.soft_lo[arm])&(desired[arm]<=task.mdp.soft_hi[arm])).all()
        pi={k:v.detach().clone() for k,v in ck['pi'].items()};pi['dec.8.weight'][arm]=0;pi['dec.8.bias'][arm]=((desired-task.mdp.default_q)/task.alpha)[arm]
        body=[j for j in range(29) if j not in arm]
        for key,value in ck['pi'].items():
            if key in ('dec.8.weight','dec.8.bias'):assert torch.equal(value[body],pi[key][body])
            else:assert torch.equal(value,pi[key])
        policy=FullbodyPolicy(task.robot);policy.load_state_dict(pi)
        with torch.no_grad():
            raw=policy(torch.randn(4,10,99),torch.randn(4,107));_,actual=task.compose_action(raw,torch.zeros(4,task.robot.xi_dim))
            torch.testing.assert_close(actual[:,arm],desired[arm].expand(4,-1),atol=1e-6,rtol=0)
        cfg=dict(ck['config'],curriculum_level=1,initialization_mode='observed_arm_center_calibration',policy_training_scope='right_arm_outputs',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100);cfg.pop('support_actor_lr',None)
        calibration=dict(method=label,gravity_compensation_gain=gain,observed_pose=provenance,arm_joint_order=[task.robot.joint_names[j] for j in arm],arm_pose_rad=pose[0,arm].tolist(),arm_pd_targets_rad=desired[arm].tolist(),distal_joint_names=[task.robot.joint_names[j] for j in distal],torch_ray_error_m=torch_error,mujoco_ray_error_m=mujoco_error,all_non_right_arm_policy_weights_exact=True,reset_pose_unchanged=True,physical_acceptance=False,scope='Center-only constant arm diagnostic; independent full physical evaluation required')
        artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),robot_metadata=ck['robot_metadata'],pi=pi,wm=ck['wm'],source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),provenance=ck.get('provenance'),calibration=calibration)
        save_checkpoint(path,artifact);json_save(path.with_suffix('.json'),dict(checkpoint_sha256=sha256(path),source_sha256=digest,calibration=calibration))
        print(json.dumps(dict(checkpoint=str(path),method=label,kinematic_error_m=torch_error,arm_pose_rad=pose[0,arm].tolist())),flush=True)
    assert sha256(a.source)==digest


if __name__=='__main__':main()
