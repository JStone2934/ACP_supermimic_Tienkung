"""Render the configured default pose with collision feet aligned to the ground."""
import argparse
import os
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image
from supermimic.acp.robots import get_robot


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--robot',default='tienkung3_29dof'); ap.add_argument('--out',required=True); args=ap.parse_args()
    robot=get_robot(args.robot); spec=robot.get_spec()
    spec.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,1),rgba=(.84,.86,.89,1))
    spec.worldbody.add_light(pos=(0,-3,4),dir=(0,1,-1),diffuse=(.8,.8,.8))
    m=spec.compile(); m.vis.global_.offwidth=1000; m.vis.global_.offheight=1000
    d=mujoco.MjData(m); d.qpos[2]=robot.stand_height; qadr,_=robot.joint_addresses(m); d.qpos[qadr]=robot.default_q; mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera(); camera.lookat[:]=(0,0,.9); camera.distance=3.2; camera.elevation=-10; camera.azimuth=135
    with mujoco.Renderer(m,1000,1000) as r:
        r.update_scene(d,camera); Image.fromarray(r.render()).save(args.out)
    print(args.out)

if __name__=='__main__': main()
