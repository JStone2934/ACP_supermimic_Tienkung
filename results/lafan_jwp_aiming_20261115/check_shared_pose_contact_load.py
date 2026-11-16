from pathlib import Path
import json,torch,numpy as np,mujoco
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
b=Path(__file__).resolve().parent;paths=[b/'wide_gated_transfer/ckpt_000001.pt',b/'wide_shared_proximal_candidates/ckpt_000002.pt'];states=[]
for path in paths:
 ck=torch.load(path,map_location='cpu',weights_only=False);T=WideAimingTask('cpu',ck['config']);m=T.robot.get_spec().compile();d=mujoco.MjData(m);addr,dof=T.robot.joint_addresses(m);d.qpos[:3]=[0,0,T.initial_height];d.qpos[3:7]=T.initial_quat;d.qpos[addr]=T.initial_q;mujoco.mj_forward(m,d);J=[]
 for name in ('left_foot','right_foot'):
  body=m.body(T.robot.contact_bodies[name][0]).id;jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,body);J.extend([jp,jr])
 J=np.concatenate(J,axis=0);weight=np.tile([1.,1.,1.,.04,.04,.04],2);A=J[:,:6].T;force=(weight[:,None]*np.linalg.pinv(A*weight[None,:]))@d.qfrc_bias[:6];tau=d.qfrc_bias-J.T@force;assert np.linalg.norm(tau[:6])<1e-5
 states.append(dict(sha256=sha256(path),com=d.subtree_com[0].copy(),tau=tau[dof].copy(),wrench=force))
delta=states[1]['tau']-states[0]['tau'];dq=delta/np.array(T.robot.kp)
r=dict(scope='Read-only static contact-load sensitivity; no closed-loop proof',source_sha256=states[0]['sha256'],candidate_sha256=states[1]['sha256'],com_shift_m=(states[1]['com']-states[0]['com']).tolist(),joint_names=T.robot.joint_names,delta_motor_torque_nm=delta.tolist(),delta_pd_target_rad=dq.tolist(),old_contact_wrench=states[0]['wrench'].tolist(),new_contact_wrench=states[1]['wrench'].tolist(),limitations='Two6Dfootwrenches minimum weighted norm; excludes friction constraints and dynamic feedback',physical_acceptance=False)
(b/'shared_pose_contact_load.json').write_text(json.dumps(r,indent=2));print('COM shift',r['com_shift_m'],'leg PD delta',dq[:12].tolist())
