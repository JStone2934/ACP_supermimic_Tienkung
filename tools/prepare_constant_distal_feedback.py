"""Calibrate a constant distal equilibrium plus audited direction feedforward/feedback."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from supermimic.rotations import quat_to_mat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','trajectory','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gains',type=float,nargs='+',default=[1.5,3.])
    a=p.parse_args();torch.set_num_threads(2)
    assert all(np.isfinite(g) and 0<g<=5 for g in a.gains)
    ck=torch.load(a.source,map_location='cpu',weights_only=False);digest=sha256(a.source)
    assert ck['stage'] in ('training','calibration') and not ck.get('smoke') and ck['config']['distal_command_skip']
    report=json.loads(a.trajectory.with_suffix('.json').read_text())
    assert report['checkpoint_sha256']==digest and report['continuous_without_reset'] and report['seconds']==30
    params=np.array(report['target_parameters'][0]);ids0=np.where(np.isclose(params[:report['fixed_count'],0],3)&np.isclose(params[:report['fixed_count'],1],0)&np.isclose(params[:report['fixed_count'],2],0))[0]
    assert len(ids0)==1;index=int(ids0[0]);assert report['scenarios'][index]['passed']
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    ids=[task.robot.joint_names.index(n) for n in ('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')]
    with np.load(a.trajectory) as z:
        assert str(z['checkpoint_sha256'])==digest
        root=torch.tensor(z['root'][750,index])[None];quat=torch.tensor(z['root_quat'][750,index])[None];q=torch.tensor(z['q'][750,index])[None]
    state=(root,quat_to_mat(quat),torch.zeros_like(root),torch.zeros_like(root),q,torch.zeros_like(q))
    def direction(x):
        joints=q.clone();joints[:,ids]=x
        _,d,torso=task.hand((*state[:4],joints,state[5]))
        return (torso.transpose(-1,-2)@d[...,None]).squeeze(-1)[0]
    x=q[0,ids].clone().requires_grad_();reference=direction(x).detach()
    J=torch.autograd.functional.jacobian(direction,x).detach();inverse=torch.linalg.pinv(J,rtol=1e-5)
    torch.testing.assert_close(J@inverse@J,J,atol=1e-5,rtol=1e-5)
    model=task.robot.get_spec().compile();data=mujoco.MjData(model);addresses,dof=task.robot.joint_addresses(model)
    data.qpos[:3]=root[0].numpy();data.qpos[3:7]=quat[0].numpy();data.qpos[addresses]=q[0].numpy();mujoco.mj_forward(model,data)
    tcp=model.body('right_tcp_link').id;torso=model.body(task.robot.torso_body).id
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(model,data,jp,jr,tcp)
    dw=-data.xmat[tcp].reshape(3,3)[:,2];R=data.xmat[torso].reshape(3,3)
    independent=R.T@np.stack([np.cross(jr[:,dof[i]],dw) for i in ids],axis=1)
    error=float(np.max(np.abs(J.numpy()-independent)));assert error<1e-5
    np.testing.assert_allclose(reference.numpy(),R.T@dw,atol=1e-5,rtol=0)
    gravity=torch.tensor(data.qfrc_bias[dof]/np.asarray(task.robot.kp),dtype=torch.float32)
    desired=q[0]+gravity
    base=(desired-task.mdp.default_q)/task.alpha
    a.out.mkdir(parents=True,exist_ok=True)
    for number,gain in enumerate(a.gains,1):
        out=a.out/f'ckpt_{number:06d}.pt'
        if out.exists():raise FileExistsError(out)
        pi={key:value.clone() for key,value in ck['pi'].items()}
        pi['dec.8.weight'][ids]=0;pi['dec.8.bias'][ids]=base[ids]
        W=inverse/task.alpha[ids,None]
        pi['distal_command_skip.weight'].zero_();pi['distal_command_skip.weight'][:,:3]=(gain+1)*W
        pi['distal_command_skip.weight'][:,3:6]=-gain*W;pi['distal_command_skip.bias']=-W@reference
        body=[j for j in range(29) if j not in ids]
        for key in pi:
            if key.startswith('distal_command_skip.'):continue
            if key in ('dec.8.weight','dec.8.bias'):assert torch.equal(pi[key][body],ck['pi'][key][body]),key
            else:assert torch.equal(pi[key],ck['pi'][key]),key
        original=FullbodyPolicy(task.robot);original.load_state_dict(ck['pi'])
        policy=FullbodyPolicy(task.robot);policy.load_state_dict(pi)
        torch.manual_seed(571);history=torch.randn(16,10,99);command=torch.randn(16,107)
        command[0,93:96]=reference;command[0,96:99]=reference
        with torch.no_grad():
            actual=policy(history,command);old=original(history,command)
            target=torch.nn.functional.normalize(command[:,93:96],dim=-1)
            expected=base[ids]+((target-reference)+gain*(target-command[:,96:99]))@W.T
        torch.testing.assert_close(actual[:,body],old[:,body],atol=0,rtol=0)
        torch.testing.assert_close(actual[:,ids],expected,atol=2e-5,rtol=2e-5)
        cfg=dict(ck['config'],initialization_mode='constant_distal_equilibrium_direction_calibration',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
        calibration=dict(method=__doc__,feedback_gain=gain,feedforward_gain=1.,trajectory=str(a.trajectory.resolve()),trajectory_sha256=sha256(a.trajectory),
            frame=750,scenario=index,jacobian=J.tolist(),pseudoinverse=inverse.tolist(),jacobian_mujoco_max_abs_difference=error,
            reference_direction_torso=reference.tolist(),distal_q_rad=q[0,ids].tolist(),distal_pd_target_rad=desired[ids].tolist(),
            all_other_joint_outputs_exact=True,normalizers_exact=True,initial_pose_unchanged=True,physical_acceptance=False,
            online_ik=False,runtime_motion_files=False,limitations='Local linear kinematics and static gravity compensation; requires full physical evaluation')
        artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),
            robot_metadata=ck['robot_metadata'],pi=pi,wm=ck['wm'],provenance=ck.get('provenance'),
            source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=calibration)
        validate_checkpoint(artifact,WideAimingTask('cpu',cfg));save_checkpoint(out,artifact)
        json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration))
        print(json.dumps(dict(output=str(out),gain=gain,jacobian_error=error)),flush=True)
    assert sha256(a.source)==digest


if __name__=='__main__':main()
