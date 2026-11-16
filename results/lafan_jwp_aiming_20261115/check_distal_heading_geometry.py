"""Offline full75target reachability assuming an ideal body yaw adjustment; not a turning policy."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.wide_curriculum import grid
from supermimic.acp.robots import sha256
from supermimic.rotations import quat_to_mat,mat_to_quat


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:1');a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(390);torch.backends.cuda.matmul.allow_tf32=False
 b=Path(__file__).resolve().parent;out=b/'distal_heading_geometry.json'
 if out.exists():raise FileExistsError(out)
 source=b/'balance_pose_center_candidates/ckpt_000002.pt';ck=torch.load(source,map_location=a.device,weights_only=False)
 task=WideAimingTask(a.device,ck['config']);ids=[task.robot.joint_names.index(n) for n in ('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')]
 trajectory=b/'evaluation_stance_bias/inward_0.25/seed_17.npz'
 with np.load(trajectory) as z:
  def state(frame):
   root=torch.tensor(z['root'][frame,0],device=a.device)[None].repeat(75,1)
   quat=torch.tensor(z['root_quat'][frame,0],device=a.device)[None].repeat(75,1)
   q=torch.tensor(z['q'][frame,0],device=a.device)[None].repeat(75,1)
   return root,quat_to_mat(quat),torch.zeros_like(root),torch.zeros_like(root),q,torch.zeros_like(q)
  initial=state(0);S=state(500);actual_quat=z['root_quat'][500,0].copy()
 params=grid(5,a.device);assert params.shape==(75,4)
 ctx=task.context(initial,params,4.)
 yaw=params[:,1];rotation=torch.zeros(75,3,3,device=a.device);rotation[:,0,0]=yaw.cos();rotation[:,1,1]=yaw.cos();rotation[:,0,1]=-yaw.sin();rotation[:,1,0]=yaw.sin();rotation[:,2,2]=1
 S=(S[0],rotation@S[1],*S[2:]);actual_quat=mat_to_quat(S[1]).cpu().numpy();nominal=S[4].clone()
 nominal[:,task.arm_ids]=torch.tensor(ck['calibration']['arm_pose_rad'],device=a.device)
 best=torch.full((75,),float('inf'),device=a.device);solution=nominal.clone()
 for restart in range(3):
  x=nominal[:,ids].clone()
  if restart:x+=torch.randn_like(x)*.5
  x=x.clamp(task.mdp.soft_lo[ids],task.mdp.soft_hi[ids]).requires_grad_()
  op=torch.optim.Adam([x],lr=.025)
  for iteration in range(700):
   q=nominal.clone();q[:,ids]=x;state=(*S[:4],q,S[5]);h,d,_=task.hand(state)
   delta=ctx['aim_target_w']-h;loss=((d-torch.nn.functional.normalize(delta,dim=-1)).square().sum(-1)*delta.square().sum(-1)).mean()
   with torch.no_grad():
    error=task.metrics(state,ctx)[0];better=error<best;best=torch.where(better,error,best);solution=torch.where(better[:,None],q.detach(),solution)
   op.zero_grad();loss.backward();op.step()
   with torch.no_grad():x.clamp_(task.mdp.soft_lo[ids],task.mdp.soft_hi[ids])
  print(json.dumps(dict(restart=restart,within5cm=int((best<=.05).sum()),median_error_m=float(best.median()))),flush=True)
 body=[j for j in range(29) if j not in ids];assert torch.equal(solution[:,body],nominal[:,body])
 model=task.robot.get_spec().compile();data=mujoco.MjData(model);addresses,_=task.robot.joint_addresses(model);tcp=model.body('right_tcp_link').id
 rows=[];maximum_difference=0.
 for i in range(75):
  data.qpos[:3]=S[0][i].cpu().numpy();data.qpos[3:7]=actual_quat[i];data.qpos[addresses]=solution[i].cpu().numpy();mujoco.mj_forward(model,data)
  hand=data.xpos[tcp];direction=-data.xmat[tcp].reshape(3,3)[:,2];target=ctx['aim_target_w'][i].cpu().numpy();normal=ctx['aim_normal_w'][i].cpu().numpy()
  den=normal@direction;distance=normal@(target-hand)/den if den>1e-4 else -1
  error=float(np.linalg.norm(hand+distance*direction-target)) if distance>0 else float('inf')
  if np.isfinite(error) and torch.isfinite(best[i]):maximum_difference=max(maximum_difference,abs(error-float(best[i])))
  else:assert np.isfinite(error)==bool(torch.isfinite(best[i]))
  row=dict(distance_m=float(params[i,0]),azimuth_deg=float(torch.rad2deg(params[i,1])),elevation_deg=float(torch.rad2deg(params[i,2])),
   error_m=error if np.isfinite(error) else None,within5cm=error<=.05,distal_q_rad=solution[i,ids].cpu().tolist(),
   minimum_soft_limit_margin_rad=float(torch.minimum(solution[i,ids]-task.mdp.soft_lo[ids],task.mdp.soft_hi[ids]-solution[i,ids]).min()))
  rows.append(row)
 assert maximum_difference<1e-4,maximum_difference
 report=dict(scope=__doc__,checkpoint_sha256=sha256(source),trajectory_sha256=sha256(trajectory),body_frame=500,scenario=0,
  target_frame='Original initial-chest world frame; exact final acceptance grid',rows=rows,within5cm=sum(r['within5cm'] for r in rows),
  torch_mujoco_max_error_difference_m=maximum_difference,robot_metadata=task.robot.metadata(),
  fixed_body_joint_pose_and_proximal_arm=True,assumed_body_yaw_adjustment_rad=params[:,1].cpu().tolist(),physical_policy_acceptance=False,limitation='Assumes ideal instantaneous body yaw adjustment; does not establish a physical turning, balance or aiming policy')
 out.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(dict(within5cm=report['within5cm'],output=str(out))),flush=True)


if __name__=='__main__':main()
