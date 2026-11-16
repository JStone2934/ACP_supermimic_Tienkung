import os,sys
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from supermimic.acp.robots import get_robot
P=Path(__file__).resolve().parent;report=json.load(open(P/'repair.json'));r=get_robot('tienkung3_29dof');spec=r.get_spec();spec.worldbody.add_geom(name='ready_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),rgba=(.8,.83,.86,1),friction=(.7,.005,.0001));spec.worldbody.add_light(pos=(2,-3,5),dir=(-.3,.3,-1));
import mjlab.actuator
from mjlab.utils.spec import create_position_actuator
for name,k,b,e,a in zip(r.joint_names,r.kp,r.kd,r.effort,r.armature):create_position_actuator(spec,name,stiffness=k,damping=b,effort_limit=e,armature=a)
m=spec.compile();m.opt.timestep=r.control_dt/r.decimation;m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;m.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON;m.opt.iterations=100;m.opt.ls_iterations=50;m.opt.tolerance=1e-6;m.geom_friction[:,0]=1;d=mujoco.MjData(m);addr,dof=r.joint_addresses(m);pose=np.array(report['repaired_pose']);m.dof_armature[dof]=r.armature

def setpose(row):
 mujoco.mj_resetData(m,d);d.qpos[:7]=row[:7];d.qpos[addr]=row[7:];mujoco.mj_forward(m,d)
setpose(pose);contacts=[]
for con in d.contact:
 n=[m.body(m.geom_bodyid[g]).name for g in con.geom];contacts.append(dict(bodies=n,distance_m=float(con.dist),point=con.pos.tolist()))
setpose(pose);target=np.array(report['repaired_pose'][7:] if '--nominal' in sys.argv else report['static']['pd_targets_rad']);eff=np.array(r.effort);kp=np.array(r.kp);kd=np.array(r.kd);records=[];maxfrac=0
for step in range(round(30/m.opt.timestep)):
 d.ctrl[:]=target;mujoco.mj_step(m,d);maxfrac=max(maxfrac,float(np.max(abs(d.actuator_force)/eff)))
 if step%4==0:
  R=d.xmat[m.body(r.torso_body).id].reshape(3,3);tilt=float(np.degrees(np.arctan2(np.linalg.norm(R[:2,2]),R[2,2])));records.append(dict(t=float(d.time),root_z=float(d.qpos[2]),root_xy=d.qpos[:2].tolist(),chest_deg=tilt,q=d.qpos[addr].tolist(),root_speed=float(np.linalg.norm(d.qvel[:3])),contacts=d.ncon))
 if d.qpos[2]<.5:break
mapping=[dict(joint=n,qpos=int(addr[i]),dof=int(dof[i]),actuator_joint=m.joint(int(m.actuator_trnid[i,0])).name) for i,n in enumerate(r.joint_names)];assert all(x['joint']==x['actuator_joint'] for x in mapping)
summary=dict(mapping=mapping,target_mode='nominal_pose' if '--nominal' in sys.argv else 'static_lp_compensated',duration_s=float(d.time),controller='Native mjlab create_position_actuator; implicitfast, dt5ms, kp/kd/effort/armature exact RobotSpec; friction1; constant static-LP compensated target, no policy; stop if rootZ<.5m else30s',max_unclipped_torque_effort_ratio=maxfrac,max_chest_deg=max(x['chest_deg'] for x in records),final_root_z=float(d.qpos[2]),initial_root_z=float(pose[2]),final_root_xy_drift_m=float(np.linalg.norm(d.qpos[:2]-pose[:2])),final_root_speed_m_s=float(np.linalg.norm(d.qvel[:3])),native_initial_contacts=contacts,trajectory=records,physical_acceptance=False)
(P/('pd_native_nominal.json' if '--nominal' in sys.argv else 'pd_native_matched.json')).write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['trajectory','native_initial_contacts']},indent=2));print('initial_contact_count',len(contacts))
