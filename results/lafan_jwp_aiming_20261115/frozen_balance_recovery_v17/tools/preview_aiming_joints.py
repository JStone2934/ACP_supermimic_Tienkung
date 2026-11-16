"""Render the proposed standing-aiming joint split; kinematic review only."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
from pathlib import Path
import subprocess
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from supermimic.acp.robots import get_robot

FIXED = ['shoulder_pitch_r_joint','shoulder_roll_r_joint','shoulder_yaw_r_joint','elbow_pitch_r_joint']
MOVING = ['elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint']
COLORS = [(0.12,.75,.95,1),(.25,.85,.35,1),(.75,.35,1,1)]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='results/analysis/standing_aiming')
    ap.add_argument('--still-only',action='store_true')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    robot=get_robot('tienkung3_29dof'); spec=robot.get_spec()
    spec.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,1),rgba=(.18,.21,.26,1))
    spec.worldbody.add_light(pos=(2,-3,4),dir=(-.4,.5,-1),diffuse=(.9,.9,.9))
    m=spec.compile(); m.vis.global_.offwidth=640; m.vis.global_.offheight=720
    d=mujoco.MjData(m); d.qpos[2]=robot.stand_height
    adr,_=robot.joint_addresses(m); d.qpos[adr]=robot.default_q
    vals=[.2,-.1,0.,-np.pi/2]
    for name,value in zip(FIXED,vals): d.qpos[m.joint(name).qposadr[0]]=value
    for name in MOVING: d.qpos[m.joint(name).qposadr[0]]=0.
    base=d.qpos.copy()
    # Color the links downstream of each joint; axes/markers show actual joint locations.
    for name in FIXED+MOVING:
        bid=m.jnt_bodyid[m.joint(name).id]
        color=(1.,.55,.12,1) if name in FIXED else COLORS[MOVING.index(name)]
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid]==bid: m.geom_rgba[gid]=color
    tcp=m.body('right_tcp_link').id
    mujoco.mj_forward(m,d)
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid]==tcp: m.geom_rgba[gid]=COLORS[-1]
    cams=[]
    for az in (145,70):
        c=mujoco.MjvCamera(); c.lookat[:]=(.10,-.16,1.36); c.distance=1.55; c.elevation=-10; c.azimuth=az; cams.append(c)
    font=ImageFont.truetype(FONT,23); small=ImageFont.truetype(FONT,18)
    def geom(scene,typ,a,b,color,width=.007):
        g=scene.geoms[scene.ngeom]; scene.ngeom+=1
        mujoco.mjv_initGeom(g,typ,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
        mujoco.mjv_connector(g,typ,width,a,b)
    def frame(renderer,active=-1,angle=0.):
        d.qpos[:]=base
        if active>=0: d.qpos[m.joint(MOVING[active]).qposadr[0]]=angle
        mujoco.mj_forward(m,d)
        canvas=Image.new('RGB',(1640,800),(20,25,34)); draw=ImageDraw.Draw(canvas)
        for k,c in enumerate(cams):
            renderer.update_scene(d,c)
            for i,name in enumerate(FIXED+MOVING):
                jid=m.joint(name).id; p=d.xanchor[jid]; axis=d.xaxis[jid]
                color=(1.,.55,.12,1) if i<4 else COLORS[i-4]
                geom(renderer.scene,mujoco.mjtGeom.mjGEOM_ARROW,p-axis*.065,p+axis*.065,color,.006)
                # Numeric marker matches the legend, without hiding joint geometry.
                g=renderer.scene.geoms[renderer.scene.ngeom]; renderer.scene.ngeom+=1
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.014]*3),p,np.eye(3).ravel(),np.array(color,dtype=np.float32))
                g.label=str(i+1)
            p=d.xpos[tcp]; direction=-d.xmat[tcp].reshape(3,3)[:,2]
            geom(renderer.scene,mujoco.mjtGeom.mjGEOM_ARROW,p,p+.32*direction,(1,.2,.25,1),.012)
            canvas.paste(Image.fromarray(renderer.render()),(k*620,60))
        draw=ImageDraw.Draw(canvas)
        draw.text((20,12),'站立瞄准 · 关节定义预览（右臂）',font=font,fill='white')
        draw.text((25,68),'前侧视角',font=small,fill='white'); draw.text((645,68),'右侧视角',font=small,fill='white')
        x=1260
        draw.text((x,70),'橙色 1–4：保持角度',font=font,fill=(255,160,65))
        labels=['1 肩部 pitch：+11.46°','2 肩部 roll：−5.73°','3 肩部 yaw：0°','4 肘部 pitch：−90°']
        for i,label in enumerate(labels): draw.text((x,112+34*i),label,font=small,fill='white')
        draw.text((x,275),'彩色 5–7：允许转动',font=font,fill='white')
        for i,(cn,name) in enumerate(zip(['5 前臂旋转','6 手腕 pitch','7 手腕 roll'],MOVING)):
            col=tuple(int(v*255) for v in COLORS[i][:3]); y=320+i*74
            draw.text((x,y),cn,font=font,fill=col); draw.text((x,y+30),name,font=small,fill=col)
        draw.text((x,560),'红箭头：TCP 局部 −Z',font=small,fill=(255,95,100))
        draw.text((x,590),'候选指向轴，待你确认',font=small,fill='white')
        status='候选初始姿态；末端三轴均为 0°' if active<0 else f'仅转动 {active+5} 号：{np.degrees(angle):+.1f}°'
        draw.text((25,756),status,font=font,fill='white')
        draw.text((x,665),'身体其余关节仅为展示冻结',font=small,fill=(190,195,205))
        draw.text((x,698),'运动学回放，不代表站立控制',font=small,fill=(190,195,205))
        return canvas
    report={'fixed':dict(zip(FIXED,vals)),'moving':{},'pointing_axis_local':[0,0,-1],'scope':'Kinematic review; no dynamics or trained standing policy.'}
    for name in FIXED+MOVING:
        j=m.joint(name); assert j.range[0]<=d.qpos[j.qposadr[0]]<=j.range[1]
        if name in MOVING: report['moving'][name]={'range_deg':np.degrees(j.range).tolist(),'axis_local':j.axis.tolist()}
    report['tcp_direction_world_at_neutral']=(-d.xmat[tcp].reshape(3,3)[:,2]).tolist()
    with mujoco.Renderer(m,720,620) as renderer:
        frame(renderer).save(out/'joint_overview.png')
        if not args.still_only:
            pipe=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1640x800','-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'joint_sweep.mp4')],stdin=subprocess.PIPE)
            try:
                for _ in range(30): pipe.stdin.write(frame(renderer).tobytes())
                for active in range(3):
                    for i in range(100):
                        angle=np.deg2rad(45)*np.sin(2*np.pi*i/99)
                        pipe.stdin.write(frame(renderer,active,angle).tobytes())
                pipe.stdin.close()
                if pipe.wait()!=0: raise RuntimeError('ffmpeg failed')
            finally:
                if pipe.poll() is None: pipe.kill(); pipe.wait()
    (out/'joint_definition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out.resolve(),flush=True)

if __name__=='__main__': main()
