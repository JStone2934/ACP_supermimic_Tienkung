"""Diagnostic fixed-center arm poses folded into the accepted balance backbone."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--balance-reports',type=Path,nargs=2,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--geometry-source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    digest=check_promotion(a.source,[json.loads(x.read_text()) for x in a.balance_reports],0)
    ck=torch.load(a.source,map_location='cpu',weights_only=False)
    assert not ck.get('smoke') and ck['config']['curriculum_level']==0
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    g=json.loads(a.geometry.read_text());assert g['checkpoint_sha256']==sha256(a.geometry_source)
    original=torch.load(a.geometry_source,map_location='cpu',weights_only=False)
    for key in ('initial_q','initial_quat','initial_height'):assert original['config'][key]==ck['config'][key],key
    arm=task.arm_ids.tolist();assert g['joints_optimized']==[task.robot.joint_names[i] for i in arm] and g['axis']==[0,0,-1]
    candidates=[i for i,x in enumerate(g['parameters']) if abs(x[0]-3)<1e-6 and abs(x[1])+abs(x[2])<1e-6]
    assert len(candidates)==1;index=candidates[0]
    state=task.initial_state(1);q=state[4].clone();q[:,arm]=torch.tensor(g['diagnostic_arm_q'][index])
    ctx=task.context(state,torch.tensor([[3.,0.,0.,.2]]),4.)
    error=task.metrics((*state[:4],q,state[5]),ctx)[0]
    assert error.item()<1e-4 and ((q>=task.mdp.soft_lo)&(q<=task.mdp.soft_hi)).all()
    model=task.robot.get_spec().compile();data=mujoco.MjData(model);addresses,dof=task.robot.joint_addresses(model)
    data.qpos[:3]=state[0][0].numpy();data.qpos[3:7]=task.initial_quat.numpy();data.qpos[addresses]=q[0].numpy()
    mujoco.mj_forward(model,data)
    gravity=data.qfrc_bias[dof].copy();comp=torch.tensor(gravity/np.asarray(task.robot.kp),dtype=torch.float32)
    body=[i for i in range(29) if i not in arm]
    a.out.mkdir(parents=True,exist_ok=True)
    for number,gain in enumerate((0.,1.),1):
        path=a.out/f'ckpt_{number:06d}.pt'
        if path.exists():raise FileExistsError(path)
        target=q[0].clone();target[arm]+=gain*comp[arm]
        if ((target[arm]<task.mdp.soft_lo[arm])|(target[arm]>task.mdp.soft_hi[arm])).any():raise ValueError('Gravity compensated arm target exceeds soft limits')
        pi={k:v.detach().clone() for k,v in ck['pi'].items()}
        pi['dec.8.weight'][arm]=0;pi['dec.8.bias'][arm]=((target-task.mdp.default_q)/task.alpha)[arm]
        for key,value in ck['pi'].items():
            if key in ('dec.8.weight','dec.8.bias'):assert torch.equal(value[body],pi[key][body])
            else:assert torch.equal(value,pi[key])
        policy=FullbodyPolicy(task.robot);policy.load_state_dict(pi)
        with torch.no_grad():
            raw=policy(torch.randn(8,10,99),torch.randn(8,107));xi=torch.zeros(8,task.spec.xi_dim) if hasattr(task,'spec') else torch.zeros(8,task.robot.xi_dim)
            _,actual=task.compose_action(raw,xi)
            torch.testing.assert_close(actual[:,arm],target[arm].expand(8,-1),atol=1e-6,rtol=0)
        cfg=dict(ck['config'],curriculum_level=1,initialization_mode='center_pose_calibration',policy_training_scope='right_arm_outputs',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
        cfg.pop('support_actor_lr',None)
        calibration=dict(method='Constant right-arm decoder targets from audited center IK, optional static gravity torque divided by joint stiffness',
            gravity_compensation_gain=gain,arm_joint_order=g['joints_optimized'],arm_pose_rad=q[0,arm].tolist(),arm_pd_targets_rad=target[arm].tolist(),
            arm_gravity_torque_nm=gravity[arm].tolist(),arm_pd_gravity_offset_rad=comp[arm].tolist(),
            source_kinematic_ray_error_m=float(error.item()),all_non_right_arm_policy_weights_exact=True,
            geometry_sha256=sha256(a.geometry),scope='Center-target physical diagnostic only; no variable-target policy or physical acceptance claim',physical_acceptance=False)
        artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),robot_metadata=ck['robot_metadata'],
            pi=pi,wm=ck['wm'],source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),provenance=ck.get('provenance'),calibration=calibration)
        save_checkpoint(path,artifact);json_save(path.with_suffix('.json'),dict(checkpoint_sha256=sha256(path),source_sha256=digest,calibration=calibration))
        print(json.dumps(dict(checkpoint=str(path),gravity_compensation_gain=gain,arm_pd_targets_rad=target[arm].tolist())))
    assert sha256(a.source)==digest


if __name__=='__main__':main()
