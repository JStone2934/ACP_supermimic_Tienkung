"""Offline shared proximal pose with constant arm equilibrium and distal feedback."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np,torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256
from supermimic.rotations import quat_to_mat
from tools.prepare_arm_observation_calibration import folded_observation_state

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('source','trajectory','geometry','out'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--feedback-gain',type=float,default=3.)
 a=p.parse_args();torch.set_num_threads(2);assert 0<a.feedback_gain<=5
 ck=torch.load(a.source,map_location='cpu',weights_only=False);digest=sha256(a.source)
 g=json.loads(a.geometry.read_text());assert g['source_sha256']==digest and g['trajectory_sha256']==sha256(a.trajectory)
 solution=next(c for c in g['cases'] if c['within5cm']==75)
 task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task);arm=task.arm_ids.tolist();ids=arm[4:];prox=arm[:4]
 with np.load(a.trajectory) as z:
  assert str(z['checkpoint_sha256'])==digest
  root=torch.tensor(z['root'][750,37])[None];quat=torch.tensor(z['root_quat'][750,37])[None];oldq=torch.tensor(z['q'][750,37])[None]
 q=oldq.clone();q[0,arm]=torch.tensor(solution['q_rad'][37])[arm]
 model=task.robot.get_spec().compile();data=mujoco.MjData(model);addresses,dof=task.robot.joint_addresses(model)
 def forward(joints):
  data.qpos[:3]=root[0];data.qpos[3:7]=quat[0];data.qpos[addresses]=joints[0];mujoco.mj_forward(model,data)
 forward(oldq);oldtarget=oldq[0]+torch.tensor(data.qfrc_bias[dof]/np.array(task.robot.kp),dtype=torch.float32)
 forward(q);target=q[0]+torch.tensor(data.qfrc_bias[dof]/np.array(task.robot.kp),dtype=torch.float32);base=(target-task.mdp.default_q)/task.alpha
 def direction(x):
  joints=q.clone();joints[:,ids]=x
  _,d,R=task.hand((root,quat_to_mat(quat),torch.zeros_like(root),torch.zeros_like(root),joints,torch.zeros_like(joints)))
  return (R.transpose(-1,-2)@d[...,None]).squeeze(-1)[0]
 x=q[0,ids].clone().requires_grad_();ref=direction(x).detach();J=torch.autograd.functional.jacobian(direction,x).detach();inverse=torch.linalg.pinv(J,rtol=1e-5)
 tcp=model.body('right_tcp_link').id;torso=model.body(task.robot.torso_body).id;jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(model,data,jp,jr,tcp)
 dw=-data.xmat[tcp].reshape(3,3)[:,2];R=data.xmat[torso].reshape(3,3);independent=R.T@np.stack([np.cross(jr[:,dof[i]],dw) for i in ids],1)
 error=float(np.max(np.abs(J.numpy()-independent)));assert error<1e-5
 policy=FullbodyPolicy(task.robot);policy.load_state_dict(ck['pi'])
 columns=prox+[29+i for i in prox]+[64+i for i in prox]
 values=torch.cat((oldq[0,prox]-task.mdp.default_q[prox],torch.zeros(4),ck['pi']['dec.8.bias'][prox]))
 folded,_=folded_observation_state(policy,columns,values)
 a.out.mkdir(parents=True,exist_ok=True)
 for number,(fold,gain) in enumerate([(False,a.feedback_gain),(True,a.feedback_gain)],1):
  out=a.out/f'ckpt_{number:06d}.pt';assert not out.exists()
  pi={k:v.clone() for k,v in (folded if fold else ck['pi']).items()};pi['dec.8.weight'][arm]=0;pi['dec.8.bias'][arm]=base[arm]
  W=inverse/task.alpha[ids,None];pi['distal_command_skip.weight'].zero_();pi['distal_command_skip.weight'][:,:3]=(gain+1)*W;pi['distal_command_skip.weight'][:,3:6]=-gain*W;pi['distal_command_skip.bias']=-W@ref
  cfg=dict(ck['config'],initialization_mode='shared_proximal_offline_calibration',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
  initial=torch.tensor(cfg['initial_q']);initial[arm]=q[0,arm];cfg['initial_q']=initial.tolist()
  calibration=dict(geometry=str(a.geometry.resolve()),geometry_sha256=sha256(a.geometry),shared_arm_pose_rad=q[0,arm].tolist(),proximal_observation_folded=fold,fold_values=values.tolist(),feedback_gain=gain,jacobian_mujoco_max_abs_difference=error,jacobian=J.tolist(),pseudoinverse=inverse.tolist(),reference_direction_torso=ref.tolist(),initial_pose_target_independent=True,body_reset_unchanged=True,physical_acceptance=False,online_ik=False)
  artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),robot_metadata=ck['robot_metadata'],pi=pi,wm=ck['wm'],provenance=ck.get('provenance'),source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=calibration)
  validate_checkpoint(artifact,WideAimingTask('cpu',cfg));save_checkpoint(out,artifact);json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration));print(str(out),flush=True)
 assert sha256(a.source)==digest
if __name__=='__main__':main()
