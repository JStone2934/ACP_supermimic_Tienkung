"""Frozen tracker: original/repaired frame452, both support command conventions."""
import sys,json
from pathlib import Path
ROOT=Path('/home/js/robot/locomotion/standing-aiming');sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from supermimic.acp.env import TrackingSim,make_sim
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP
from supermimic.acp.networks import Policy
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.task import ContactCfg,ContactTask,goal_command
from supermimic.rotations import mat_to_quat

def main():
 torch.set_num_threads(2);torch.manual_seed(17);dev='cuda:0';out=ROOT/'results/take006_tracker_validation'
 ckpath=ROOT/'logs/tienkung3_29dof/lafan_jwp_20261115/jwp_finetune_support/train/ckpt_001000.pt'
 digest=sha256(ckpath);ck=torch.load(ckpath,map_location=dev,weights_only=False)
 robot=get_robot('tienkung3_29dof');robot.validate_metadata(ck['robot_metadata']);kin=RobotKinematics(dev,robot)
 m=MotionLib.from_dir(str(out/'dataset/motions'),kin,dev);mdp=MDP(kin,m,torch.tensor(robot.default_q,device=dev))
 cc=dict(ck['ctc_cfg']);cc['contacts_dir']=str(out/'dataset/contacts');ctc=ContactTask(ContactCfg(**cc),mdp,str(out/'dataset/motions'),dev)
 pi=Policy(robot.obs_dim,ctc.cmd_dim,hist=10,robot=robot,pose_action_skip=ck['args'].get('pose_action_skip',False)).to(dev)
 pi.load_state_dict(ck['pi']);pi.eval().requires_grad_(False)
 n=4;env=TrackingSim(n,m,mdp,dev,eval_mode=True,sim_cfg=make_sim(robot=robot,physics_substeps=ck['args'].get('physics_substeps')))
 env.reset_noise=False;ids=torch.arange(n,device=dev);frames=torch.full((n,),452,device=dev,dtype=torch.long);env.reset(ids,frames)
 ready=torch.tensor(np.load(ROOT/'results/lafan_jwp_aiming_20261115/chest_posture_revision/take006_ready/ready_pose.npz')['repaired_pose'],device=dev,dtype=torch.float32)
 rid=ids[2:];pose=ready[:7].repeat(2,1);pose[:,:3]+=env.scene.env_origins[rid]
 env.robot.write_root_link_pose_to_sim(pose,env_ids=rid)
 env.robot.write_joint_position_to_sim(ready[7:][env.to_internal].repeat(2,1),env_ids=rid)
 env.robot.write_root_link_velocity_to_sim(torch.zeros(n,6,device=dev),env_ids=ids)
 env.robot.write_joint_velocity_to_sim(torch.zeros(n,29,device=dev),env_ids=ids)
 env.refresh();initial=tuple(x.clone() for x in env.state());fixed_q=initial[4].clone()
 anchor=initial[0].clone();yaw=torch.atan2(initial[1][:,1,0],initial[1][:,0,0]);targets=ctc.contact_point(initial,m.boxes(frames,env.v)).clone()
 valid=torch.zeros(n,9,dtype=torch.bool,device=dev);valid[[0,2],:2]=True
 feetids=[kin.tracked_names.index(robot.contact_bodies[k][0]) for k in ('left_foot','right_foot')];torso=kin.tracked_names.index(robot.torso_body)
 feet0=kin.forward(*initial)[0][:,feetids];history=None;trace=[];metrics=[];failed=torch.zeros(n,dtype=torch.bool,device=dev)
 first=torch.full((n,),-1.,device=dev)
 for step in range(1500):
  env.f.copy_(frames);state=env.state();obs=env.observe();history=obs[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
  cmd=ctc.command(state,frames,env.v,torch.zeros_like(frames),noise=False,targets_w=targets,valid=valid,goal=goal_command(state,anchor,yaw,.5),reference_q=fixed_q)
  a=pi(history,cmd);transition=env.step(a);state=env.state();bp,br,*_=kin.forward(*state)
  tilt=torch.rad2deg(torch.acos(state[1][:,2,2].clamp(-1,1)));chest=torch.rad2deg(torch.acos(br[:,torso,2,2].clamp(-1,1)))
  drift=(state[0][:,:2]-anchor[:,:2]).norm(dim=-1)
  if step==149:feet0=bp[:,feetids].clone()
  feet=(bp[:,feetids]-feet0).norm(dim=-1).amax(-1) if step>=149 else torch.zeros_like(drift)
  physical_bad=(state[0][:,2]-anchor[:,2]).abs()>.25
  bad=physical_bad|(tilt>15)|(drift>.5)|~a.isfinite().all(-1)
  first=torch.where(bad&~failed,(step+1)*.02,first);failed|=bad
  metrics.append(torch.stack((tilt,chest,drift,feet,(state[4]-fixed_q).square().mean(-1).sqrt()),-1).cpu().numpy())
  if step%5==0:trace.append(torch.cat((state[0],mat_to_quat(state[1]),state[4]),-1).cpu().numpy())
  if step%250==0:print('step',step,flush=True)
 arr=np.stack(metrics);names=['original_fixed_contacts','original_no_upcoming_contacts','repaired_fixed_contacts','repaired_no_upcoming_contacts'];reports=[]
 for i,name in enumerate(names):
  reports.append(dict(name=name,balance_pass=not bool(failed[i]),first_balance_failure_s=float(first[i]),max_pelvis_deg=float(arr[:,i,0].max()),max_chest_deg=float(arr[:,i,1].max()),post_grace_chest_valid_fraction=float((arr[149:,i,1]<=10).mean()),max_root_drift_m=float(arr[:,i,2].max()),post_grace_max_foot_drift_m=float(arr[:,i,3].max()),joint_rmse_rad=float(np.sqrt((arr[:,i,4]**2).mean()))))
 assert sha256(ckpath)==digest
 result=dict(checkpoint=str(ckpath),sha256=digest,source_frame=452,reference_pose=cc.get('reference_pose'),physics_substeps=env.physics_substeps,seconds=30,reset_noise=False,continuous_without_reset=True,scope='Nominal diagnostic, not100-scenario acceptance; fixed reference including q only when supported by original tracker; custom balance thresholds root height25cm, pelvis15deg, root radius50cm; no training',scenarios=reports)
 (out/'hold.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'hold.npz',trajectory=np.stack(trace),sample_dt=.1,metrics=arr)
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 with torch.no_grad():main()
