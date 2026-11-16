"""Initialize the learned distal command readout from an independently checked local Jacobian."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256
from supermimic.rotations import quat_to_mat


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
 p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--gains',type=float,nargs='+',default=[1.,.5,.25]);a=p.parse_args();torch.set_num_threads(2)
 assert all(np.isfinite(k) and 0<k<=2 for k in a.gains)
 ck=torch.load(a.source,map_location='cpu',weights_only=False);digest=sha256(a.source)
 assert not ck.get('smoke') and ck['stage']=='calibration' and ck['config']['distal_command_skip']
 assert all(torch.count_nonzero(ck['pi'][key])==0 for key in ('distal_command_skip.weight','distal_command_skip.bias'))
 task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
 report=json.loads(a.trajectory.with_suffix('.json').read_text());assert report['scenarios'][0]['passed']
 assert report['checkpoint_sha256']==ck['source_checkpoint']['sha256']
 ids=[task.robot.joint_names.index(n) for n in ('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')]
 with np.load(a.trajectory) as z:
  assert str(z['checkpoint_sha256'])==report['checkpoint_sha256']
  frame=750;index=0;root=torch.tensor(z['root'][frame,index])[None];quat=torch.tensor(z['root_quat'][frame,index])[None];q=torch.tensor(z['q'][frame,index])[None]
 state=(root,quat_to_mat(quat),torch.zeros_like(root),torch.zeros_like(root),q,torch.zeros_like(q))
 def direction(x):
  joints=q.clone();joints[:,ids]=x
  _,d,torso=task.hand((*state[:4],joints,state[5]))
  return (torso.transpose(-1,-2)@d[...,None]).squeeze(-1)[0]
 x=q[0,ids].clone().requires_grad_();J=torch.autograd.functional.jacobian(direction,x).detach()
 model=task.robot.get_spec().compile();data=mujoco.MjData(model);addresses,dof=task.robot.joint_addresses(model)
 data.qpos[:3]=root[0].numpy();data.qpos[3:7]=quat[0].numpy();data.qpos[addresses]=q[0].numpy();mujoco.mj_forward(model,data)
 tcp=model.body('right_tcp_link').id;torso=model.body(task.robot.torso_body).id
 jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(model,data,jp,jr,tcp)
 dw=-data.xmat[tcp].reshape(3,3)[:,2];R=data.xmat[torso].reshape(3,3)
 independent=R.T@np.stack([np.cross(jr[:,dof[i]],dw) for i in ids],axis=1)
 error=float(np.max(np.abs(J.numpy()-independent)));assert error<1e-5,error
 np.testing.assert_allclose(direction(x).detach().numpy(),R.T@dw,atol=1e-5,rtol=0)
 singular=torch.linalg.svdvals(J);assert singular[1]>.1 and singular[2]<1e-5,singular
 inverse=torch.linalg.pinv(J,rtol=1e-5)
 torch.testing.assert_close(J@inverse@J,J,atol=1e-5,rtol=1e-5)
 a.out.mkdir(parents=True,exist_ok=True)
 geometry=dict(trajectory=str(a.trajectory.resolve()),trajectory_sha256=sha256(a.trajectory),trajectory_checkpoint_sha256=report['checkpoint_sha256'],
  frame=frame,scenario=index,scenario_passed_full_center=True,axis=[0,0,-1],frame_of_vectors='torso',joint_names=[task.robot.joint_names[i] for i in ids],
  q_rad=q[0,ids].tolist(),jacobian=J.tolist(),pseudoinverse=inverse.tolist(),singular_values=singular.tolist(),torch_mujoco_jacobian_max_abs_difference=error)
 for number,gain in enumerate(a.gains,1):
  path=a.out/f'ckpt_{number:06d}.pt'
  if path.exists():raise FileExistsError(path)
  pi={k:v.detach().clone() for k,v in ck['pi'].items()};W=gain*inverse/task.alpha[ids,None]
  pi['distal_command_skip.weight'][:,:3]=W;pi['distal_command_skip.weight'][:,3:6]=-W
  for key,value in ck['pi'].items():
   if key!='distal_command_skip.weight':assert torch.equal(value,pi[key]),key
  policy=FullbodyPolicy(task.robot);policy.load_state_dict(pi)
  torch.manual_seed(111);history=torch.randn(8,10,99);command=torch.randn(8,107)
  original=FullbodyPolicy(task.robot);original.load_state_dict(ck['pi'])
  with torch.no_grad():
   expected=(torch.nn.functional.normalize(command[:,93:96],dim=-1,eps=1e-6)-command[:,96:99])@W.T
   delta=policy(history,command)-original(history,command)
   body=[j for j in range(29) if j not in ids];torch.testing.assert_close(delta[:,body],torch.zeros_like(delta[:,body]),atol=0,rtol=0)
   torch.testing.assert_close(delta[:,ids],expected,atol=1e-5,rtol=1e-5)
  cfg=dict(ck['config'],initialization_mode='local_distal_direction_feedback_calibration')
  calibration=dict(method=__doc__,gain=gain,geometry=geometry,all_encoder_decoder_weights_exact=True,
   normalizers_exact=True,runtime_motion_files=False,online_ik=False,feedback='Fixed affine network weights applied to unit_target_direction minus observed_TCP_direction',
   physical_acceptance=False,limitations='Local Jacobian initialization; joint saturation and dynamic stability require full physical evaluation')
  artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=task.metadata(),robot_metadata=ck['robot_metadata'],
   pi=pi,wm=ck['wm'],provenance=ck.get('provenance'),source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=calibration)
  validate_checkpoint(artifact,WideAimingTask('cpu',cfg));save_checkpoint(path,artifact)
  json_save(path.with_suffix('.json'),dict(checkpoint_sha256=sha256(path),calibration=calibration))
  print(json.dumps(dict(output=str(path),gain=gain,jacobian_error=error)),flush=True)
 assert sha256(a.source)==digest


if __name__=='__main__':main()
