"""Render recorded actual dynamics with target circle, pointing ray and hit point."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,subprocess
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from supermimic.acp.robots import get_robot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('trajectory');ap.add_argument('out');args=ap.parse_args()
    z=np.load(args.trajectory);s=z['s'][:,0];goals=z['goal'][:,0];robot=get_robot('tienkung3_29dof')
    spec=robot.get_spec();spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),rgba=(.25,.28,.32,1));spec.worldbody.add_light(pos=(0,-3,5),dir=(0,0,-1))
    m=spec.compile();m.vis.global_.offwidth=960;m.vis.global_.offheight=540;d=mujoco.MjData(m);adr,_=robot.joint_addresses(m);tcp=m.body('right_tcp_link').id
    cam=mujoco.MjvCamera();cam.lookat[:]=(1.1,0,1.);cam.distance=5.;cam.azimuth=110;cam.elevation=-12
    n=z['n0'][0];e1=z['e1'][0];e2=z['e2'][0];Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    pipe=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','25','-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p',args.out],stdin=subprocess.PIPE)
    def line(scene,a,b,col,width=.006):
        geom=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(col))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b)
    with mujoco.Renderer(m,540,960) as render:
        for t in range(0,min(len(s)-1,int(z['end'][0])),2):
            d.qpos[:7]=s[t,:7];d.qpos[adr]=s[t,13:42];mujoco.mj_forward(m,d);render.update_scene(d,cam)
            h=d.xpos[tcp];direction=-d.xmat[tcp].reshape(3,3)[:,2];g=goals[t,:3];rho=goals[t,6]
            b=n@direction;length=(n@(g-h))/b if b>1e-4 else 3;hit=h+length*direction;error=np.linalg.norm(hit-g)
            col=(.15,1.,.25,1) if b>1e-4 and length>0 and error<=rho else (1.,.2,.1,1)
            theta=np.linspace(0,2*np.pi,49);ring=g+rho*(np.cos(theta)[:,None]*e1+np.sin(theta)[:,None]*e2)
            for a,c in zip(ring[:-1],ring[1:]):line(render.scene,a,c,(.9,.8,.1,1))
            line(render.scene,h,h+min(max(length,.01),5)*direction,col,.008)
            line(render.scene,g-.03*e1,g+.03*e1,(1,1,1,1));line(render.scene,g-.03*e2,g+.03*e2,(1,1,1,1))
            if b>1e-4 and 0<length<5:
                geom=render.scene.geoms[render.scene.ngeom];render.scene.ngeom+=1;mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.025]*3),hit,np.eye(3).ravel(),np.array(col))
            im=Image.fromarray(render.render());ImageDraw.Draw(im).text((12,12),f't={t*.02:.2f}s  target radius={rho*100:.0f}cm  error={error*100:.1f}cm',fill='white');pipe.stdin.write(im.tobytes())
    pipe.stdin.close();pipe.wait()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from supermimic.acp.aiming import setup,DEFAULT_CONFIG
    from supermimic.acp.mdp import unflatten_state
    import json,torch
    config=json.loads(str(z['task_config'].item()))['config'] if 'task_config' in z else json.loads(Path(DEFAULT_CONFIG).read_text())
    task=setup(config,'cpu');S=unflatten_state(torch.tensor(s));bp=task.mdp.kin.forward(*S)[0].numpy();err=task.metrics(S,torch.tensor(goals))[0].numpy()
    fig,axes=plt.subplots(2,1,figsize=(8,6));tt=np.arange(len(s))*.02
    axes[0].plot(tt,err*100);axes[0].axhline(goals[0,6]*100,color='r',ls='--');axes[0].set_ylabel('Target error (cm)')
    axes[1].plot(tt,np.linalg.norm(s[:,:2]-s[0,:2],axis=-1)*100,label='pelvis')
    for i,label in zip(task.feet,['left foot','right foot']):axes[1].plot(tt,np.linalg.norm(bp[:,i,:2]-bp[0,i,:2],axis=-1)*100,label=label)
    axes[1].set_ylabel('Displacement (cm)');axes[1].set_xlabel('Time (s)');axes[1].legend();fig.tight_layout();fig.savefig(Path(args.out).with_suffix('.png'))

if __name__=='__main__':main()
