import os,sys
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
ROOT=Path('/home/js/robot/locomotion/standing-aiming');sys.path.insert(0,str(ROOT))
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from supermimic.acp.robots import get_robot
r=get_robot('tienkung3_29dof');spec=r.get_spec();spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.05],rgba=[.35,.4,.45,1],group=2);spec.worldbody.add_light(pos=[0,-3,5],dir=[0,0,-1])
m=spec.compile();m.vis.global_.offwidth=640;m.vis.global_.offheight=480;d=mujoco.MjData(m);qa,_=r.joint_addresses(m)
cam=mujoco.MjvCamera();cam.distance=3;cam.azimuth=40;cam.elevation=-12;opt=mujoco.MjvOption();opt.geomgroup[0]=0
out=ROOT/'results/take006_tracker_validation';z=np.load(out/'full_clip.npz');ref=np.load(out/'dataset/motions/jwp/take006_chr00_motion.npz')['joint_pos']
with mujoco.Renderer(m,480,640) as ren:
 def frame(q,text):
  d.qpos[:7]=q[:7];d.qpos[qa]=q[7:];mujoco.mj_forward(m,d);cam.lookat[:]=q[:3];cam.lookat[2]=.85
  cam.azimuth=np.degrees(np.arctan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]**2+q[6]**2)))+110;ren.update_scene(d,cam,scene_option=opt);im=Image.fromarray(ren.render());ImageDraw.Draw(im).text((10,12),text,fill='white');return im
 with imageio.get_writer(out/'full_clip_comparison.mp4',fps=10,codec='libx264') as writer:
  for t in range(0,len(z['q']),5):
   q=np.r_[z['root_p'][t,0],z['root_quat'][t,0],z['q'][t,0]];im=Image.new('RGB',(1280,480))
   im.paste(frame(ref[int(z['f'][t,0])],f'Reference | {t*.02:.2f}s'),(0,0));im.paste(frame(q,f'Frozen tracker physics | {t*.02:.2f}s'),(640,0));writer.append_data(np.asarray(im))
 if (out/'hold.npz').exists():
  h=np.load(out/'hold.npz')['trajectory'];names=['Original + fixed contacts','Original + no upcoming contacts','Repaired + fixed contacts','Repaired + no upcoming contacts']
  with imageio.get_writer(out/'hold_comparison.mp4',fps=10,codec='libx264') as writer:
   for t in range(len(h)):
    im=Image.new('RGB',(1280,960))
    for j in range(4):im.paste(frame(h[t,j],f'{names[j]} | {t*.1+.02:.2f}s'),((j%2)*640,(j//2)*480))
    writer.append_data(np.asarray(im))
    if t==100:im.save(out/'hold_10s.jpg')
