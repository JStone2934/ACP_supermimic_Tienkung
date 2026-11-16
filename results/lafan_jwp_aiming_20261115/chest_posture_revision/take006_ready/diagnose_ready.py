import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from supermimic.acp.robots import get_robot
P=Path(__file__).resolve().parent;report=json.load(open(P/'repair.json'));r=get_robot('tienkung3_29dof');spec=r.get_spec();spec.worldbody.add_geom(name='ready_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),rgba=(.8,.83,.86,1),friction=(.7,.005,.0001));spec.worldbody.add_light(pos=(2,-3,5),dir=(-.3,.3,-1));m=spec.compile();m.opt.timestep=.001;d=mujoco.MjData(m);addr,dof=r.joint_addresses(m);pose=np.array(report['repaired_pose']);m.dof_armature[dof]=r.armature

def setpose(row):
 mujoco.mj_resetData(m,d);d.qpos[:7]=row[:7];d.qpos[addr]=row[7:];mujoco.mj_forward(m,d)
setpose(pose);contacts=[]
for con in d.contact:
 n=[m.body(m.geom_bodyid[g]).name for g in con.geom];contacts.append(dict(bodies=n,distance_m=float(con.dist),point=con.pos.tolist()))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);camera=mujoco.MjvCamera();camera.lookat[:]=(0,0,.9);camera.distance=3.25;camera.elevation=-10;m.vis.global_.offwidth=640;m.vis.global_.offheight=640
sheet=Image.new('RGB',(1920,1280),'white')
with mujoco.Renderer(m,640,640) as render:
 for i,(name,row) in enumerate([('Original selected',report['initial_pose']),('Legs coordinated',pose)]):
  setpose(np.array(row))
  for k,az in enumerate([35,0,90]):
   camera.azimuth=az;render.update_scene(d,camera);im=Image.fromarray(render.render());ImageDraw.Draw(im).text((10,10),name,fill='black',font=font);im.save(P/f'{name.split()[0].lower()}_{k}.jpg');sheet.paste(im,(640*k,640*i))
sheet.save(P/'comparison.jpg')
setpose(pose);target=np.array(report['static']['pd_targets_rad']);eff=np.array(r.effort);kp=np.array(r.kp);kd=np.array(r.kd);records=[];maxfrac=0
for step in range(2000):
 tau=kp*(target-d.qpos[addr])-kd*d.qvel[dof];maxfrac=max(maxfrac,float(np.max(abs(tau)/eff)));d.qfrc_applied[:]=0;d.qfrc_applied[dof]=np.clip(tau,-eff,eff);mujoco.mj_step(m,d)
 if step%20==0:
  R=d.xmat[m.body(r.torso_body).id].reshape(3,3);tilt=float(np.degrees(np.arctan2(np.linalg.norm(R[:2,2]),R[2,2])));records.append(dict(t=float(d.time),root_z=float(d.qpos[2]),root_xy=d.qpos[:2].tolist(),chest_deg=tilt,q=d.qpos[addr].tolist(),root_speed=float(np.linalg.norm(d.qvel[:3])),contacts=d.ncon))
summary=dict(duration_s=float(d.time),controller='Fixed gravity-compensated PD targets from feasible static wrench solve; effort clipped, normal model kp/kd/armature, 1ms MuJoCo integration; no policy',max_unclipped_torque_effort_ratio=maxfrac,max_chest_deg=max(x['chest_deg'] for x in records),final_root_z=float(d.qpos[2]),initial_root_z=float(pose[2]),final_root_xy_drift_m=float(np.linalg.norm(d.qpos[:2]-pose[:2])),final_root_speed_m_s=float(np.linalg.norm(d.qvel[:3])),native_initial_contacts=contacts,trajectory=records,physical_acceptance=False)
(P/'pd_diagnostic.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['trajectory','native_initial_contacts']},indent=2));print('initial_contact_count',len(contacts))
