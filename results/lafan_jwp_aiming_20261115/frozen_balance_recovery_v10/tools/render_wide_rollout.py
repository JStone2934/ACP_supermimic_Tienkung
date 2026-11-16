"""Render recorded physical states and world-fixed target switches; no reference data."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
from pathlib import Path
import subprocess
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from supermimic.acp.robots import get_robot,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trajectory',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--scenario',type=int,default=0);args=p.parse_args();z=np.load(args.trajectory,allow_pickle=False)
    robot=get_robot('tienkung3_29dof');robot.validate_metadata(json.loads(str(z['robot_metadata'].item())))
    i=args.scenario
    root=z['root'][:,i];rotation=z['root_quat'][:,i];joints=z['q'][:,i];targets=z['targets'][:,i];normals=z['target_normals'][:,i]
    if not all(np.isfinite(x).all() for x in (root,rotation,joints,targets,normals)):raise ValueError('Cannot render nonfinite physical states')
    dt=float(z['control_dt']);radius=float(z['hit_radius_m']);spec=robot.get_spec()
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),rgba=(.25,.28,.32,1))
    spec.worldbody.add_light(pos=(0,-3,5),dir=(0,0,-1));model=spec.compile()
    model.vis.global_.offwidth=960;model.vis.global_.offheight=540;data=mujoco.MjData(model)
    addresses,_=robot.joint_addresses(model);tcp=model.body('right_tcp_link').id
    camera=mujoco.MjvCamera();camera.distance=5.5;camera.elevation=-12
    initial_yaw=np.arctan2(2*(rotation[0,0]*rotation[0,3]+rotation[0,1]*rotation[0,2]),1-2*(rotation[0,2]**2+rotation[0,3]**2))
    camera.azimuth=np.degrees(initial_yaw)+110
    args.out.parent.mkdir(parents=True,exist_ok=True)
    pipe=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540',
        '-r',str(1/(2*dt)),'-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p',str(args.out)],stdin=subprocess.PIPE)
    def line(scene,a,b,color,width=.006):
        g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b)
    with mujoco.Renderer(model,540,960) as renderer:
        for t in range(0,len(root)-1,2):
            data.qpos[:3]=root[t];data.qpos[3:7]=rotation[t];data.qpos[addresses]=joints[t]
            mujoco.mj_forward(model,data)
            camera.lookat[:]=(root[t]+targets[t])/2;camera.lookat[2]=1.
            renderer.update_scene(data,camera)
            h=data.xpos[tcp];direction=-data.xmat[tcp].reshape(3,3)[:,2];normal=normals[t]
            side=np.cross(np.array([0.,0.,1.]),normal);side/=np.linalg.norm(side)
            up=np.cross(normal,side);up/=np.linalg.norm(up)
            theta=np.linspace(0,2*np.pi,49);ring=targets[t]+radius*(np.cos(theta)[:,None]*side+np.sin(theta)[:,None]*up)
            for a,b in zip(ring[:-1],ring[1:]):line(renderer.scene,a,b,(.9,.8,.1,1))
            den=normal@direction;distance=normal@(targets[t]-h)/den if den>1e-4 else 3.
            error=z['error'][t,i];color=(.15,1.,.25,1) if np.isfinite(error) and error<=radius else (1.,.2,.1,1)
            line(renderer.scene,h,h+np.clip(distance,.01,5)*direction,color,.008)
            frame=Image.fromarray(renderer.render());ImageDraw.Draw(frame).text((12,12),
                f'Physical rollout  t={t*dt:.2f}s  target radius={radius:.2f}m  error={error:.3f}m',fill='white')
            pipe.stdin.write(frame.tobytes())
    pipe.stdin.close()
    if pipe.wait():raise RuntimeError('ffmpeg failed')
    digest=None
    if 'checkpoint_sha256' in z.files:
        try:
            value=z['checkpoint_sha256'].item()
            if isinstance(value,str) and len(value)==64:digest=value
        except ValueError:
            pass  # Old diagnostic trajectories used an object-valued None.
    args.out.with_suffix('.json').write_text(json.dumps(dict(video_sha256=sha256(args.out),
        trajectory=str(args.trajectory.resolve()),trajectory_sha256=sha256(args.trajectory),
        checkpoint_sha256=digest,scenario=i,seconds=(len(root)-1)*dt,control_dt=dt,
        hit_radius_m=radius,robot_metadata=robot.metadata(),scope='Actual recorded physical states'),indent=2))


if __name__=='__main__':main()
