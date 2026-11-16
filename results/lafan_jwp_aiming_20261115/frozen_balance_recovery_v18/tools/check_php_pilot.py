"""Independent full-mesh quality checks and free-root reference-PD playback for the pilot."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.motion import load_php_npz
from supermimic.acp.terrain import load_terrain
from supermimic.acp.make_labels import build_model,geom_distance
from tools.retarget_php_pilot import sole,runs

BASELINE_PAIRS={frozenset(('knee_pitch_l_link','ankle_roll_l_link')),frozenset(('knee_pitch_r_link','ankle_roll_r_link'))}


def longest(mask): return int(max((z-a for a,z in runs(mask)),default=0))


def scene(robot,boxes,self_collision=False,actuators=False):
    s=robot.get_spec();s.option.timestep=.005;s.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.iterations=100;s.option.ls_iterations=50
    s.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,1],friction=[1,.005,.0001])
    for k,(cx,cy,top,c,ss,hx,hy,hz) in enumerate(boxes):
        yaw=np.arctan2(ss,c)
        s.worldbody.add_geom(name=f'box{k}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[cx,cy,top-hz],size=[hx,hy,hz],quat=[np.cos(yaw/2),0,0,np.sin(yaw/2)],friction=[1,.005,.0001])
    for g in s.geoms:
        if g.conaffinity and g.parent.name not in ('world',''):
            g.contype=int(self_collision);g.conaffinity=1
    if actuators:
        for n,k,d,e,a in zip(robot.joint_names,robot.kp,robot.kd,robot.effort,robot.armature):
            s.joint(n).armature=a
            act=s.add_actuator(name=n+'_pd',target=n,trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=k,kv=d);act.forcelimited=True;act.forcerange=[-e,e]
    return s.compile()


def quality(robot,path):
    p,quat,q,fps=load_php_npz(path,robot);boxes=load_terrain(str(path))[0];boxes=boxes[boxes[:,5]>0]
    m=scene(robot,boxes,True);d=mujoco.MjData(m);qa,_=robot.joint_addresses(m)
    intent=np.load(str(path).replace('_motion.npz','_intent.npz'));contact=intent['contact']
    feet=[robot.contact_bodies[n][0] for n in ('left_foot','right_foot')]
    centers=np.array([sole(robot,n) for n in ('left_foot','right_foot')]);fc=[];dist=[]
    footgeoms=[[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name==n and m.geom_conaffinity[g]] for n in feet]
    terrain=[g for g in range(m.ngeom) if m.geom_bodyid[g]==0 and m.geom_conaffinity[g]]
    pen_terrain=[];pen_self=[];pen_raw=[];pair_max=defaultdict(float);terrain_pairs=defaultdict(float);witness=np.zeros(6)
    for t in range(len(p)):
        d.qpos[:3]=p[t];d.qpos[3:7]=quat[t];d.qpos[qa]=q[t];mujoco.mj_forward(m,d)
        pt=ps=pr=0.
        for c in d.contact:
            b1,b2=(m.body(m.geom_bodyid[g]).name for g in c.geom)
            penetration=max(0.,-float(c.dist))
            if 0 in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]):
                pt=max(pt,penetration);terrain_pairs[" / ".join(sorted((b1,b2)))]=max(terrain_pairs[" / ".join(sorted((b1,b2)))],penetration)
            else:
                pr=max(pr,penetration);pair_max[' / '.join(sorted((b1,b2)))]=max(pair_max[' / '.join(sorted((b1,b2)))],penetration)
                if frozenset((b1,b2)) not in BASELINE_PAIRS:ps=max(ps,penetration)
        pen_terrain.append(pt);pen_self.append(ps);pen_raw.append(pr)
        fc.append([d.xpos[m.body(n).id]+d.xmat[m.body(n).id].reshape(3,3)@centers[b] for b,n in enumerate(feet)])
        dist.append([min(geom_distance(m,d,g,tg,.3,witness,strict_reference=True) for g in gs for tg in terrain) for gs in footgeoms])
    fc=np.array(fc);dist=np.array(dist);slip=np.linalg.norm(np.diff(fc[...,:2],axis=0),axis=-1)*fps
    support_pair=contact[1:]&contact[:-1]
    support_error=np.abs(dist[contact]);slip_values=slip[support_pair]
    qa_lo=np.array([m.joint(n).range[0] for n in robot.joint_names]);qa_hi=np.array([m.joint(n).range[1] for n in robot.joint_names])
    limit=max(0.,float(np.maximum(qa_lo-q,q-qa_hi).max()))
    speed=np.abs(np.gradient(q,axis=0)*fps);velocity_max_ratio=float((speed/robot.velocity).max())
    pt=np.array(pen_terrain);ps=np.array(pen_self);pr=np.array(pen_raw)
    touch95=float(np.percentile(support_error,95)) if len(support_error) else None
    slip95=float(np.percentile(slip_values,95)) if len(slip_values) else None
    reasons=[]
    if limit>1e-4:reasons.append('joint_limit')
    if pt.max()>.005 or longest(pt>.002)>=3:reasons.append('terrain_penetration')
    if ps.max()>.005 or longest(ps>.002)>=3:reasons.append('nonbaseline_self_collision')
    if touch95 is None or touch95>.01:reasons.append('support_distance')
    if slip95 is None or slip95>.05:reasons.append('support_slip')
    if velocity_max_ratio>1:reasons.append('joint_velocity')
    raw_ok=bool(pr.max()<=.005 and longest(pr>.002)<3)
    return dict(frames=len(p),duration_s=(len(p)-1)/fps,source_motion_hash=sha256(path),joint_limit_max_rad=limit,quaternion_norm_max_error=float(np.abs(np.linalg.norm(quat,axis=1)-1).max()),joint_velocity_max_ratio=velocity_max_ratio,
                terrain_penetration_max_m=float(pt.max()),terrain_penetration_over_2mm_longest_frames=longest(pt>.002),self_penetration_raw_max_m=float(pr.max()),self_penetration_excluding_baseline_max_m=float(ps.max()),self_penetration_over_2mm_longest_frames=longest(ps>.002),self_pair_max_m=dict(pair_max),terrain_pair_max_m=dict(terrain_pairs),support_frames=int(contact.sum()),support_distance_p95_m=touch95,support_slip_p95_m_s=slip95,strict_geometry_pass=not reasons and raw_ok,pilot_screen_pass=not reasons,rejection_reasons=reasons,baseline_self_collision_exception=not raw_ok)


def playback(robot,path,out):
    p,quat,q,fps=load_php_npz(path,robot);boxes=load_terrain(str(path))[0];boxes=boxes[boxes[:,5]>0]
    m=scene(robot,boxes,False,True);d=mujoco.MjData(m);qa,va=robot.joint_addresses(m)
    R=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();v=np.gradient(p,axis=0)*fps;qd=np.gradient(q,axis=0)*fps
    omega=(Rotation.from_matrix(R[1]) * Rotation.from_matrix(R[0]).inv()).as_rotvec()*fps
    d.qpos[:3]=p[0];d.qpos[3:7]=quat[0];d.qpos[qa]=q[0];d.qvel[:3]=v[0];d.qvel[3:6]=R[0].T@omega;d.qvel[va]=qd[0]
    mujoco.mj_forward(m,d)
    rp=[];rq=[];rj=[];errors=[];joint_errors=[];sat=0;vel_over=0;total=0;failed=False
    for t in range(len(p)-1):
        rp.append(d.qpos[:3].copy());rq.append(d.qpos[3:7].copy());rj.append(d.qpos[qa].copy())
        for sub in range(4):
            a=(sub+1)/4;d.ctrl[:]=(1-a)*q[t]+a*q[t+1]
            mujoco.mj_step(m,d)
            sat+=int((np.abs(d.actuator_force)>=.99*np.array(robot.effort)).sum())
            vel_over+=int((np.abs(d.qvel[va])>np.array(robot.velocity)).sum());total+=robot.num_joints
        errors.append(float(np.linalg.norm(d.qpos[:3]-p[t+1])));joint_errors.append(float(np.sqrt(np.mean((d.qpos[qa]-q[t+1])**2))))
        up=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()[2,2]
        if not np.isfinite(d.qpos).all() or d.qpos[2]<p[t+1,2]-.25 or up<.5 or np.linalg.norm(d.qpos[:3]-p[t+1])>.5:
            failed=True;break
    T=len(rp);folder=out/path.parent.name;folder.mkdir(parents=True,exist_ok=True);traj=folder/path.name.replace('_motion.npz','_pd.npz')
    np.savez_compressed(traj,robot_metadata=json.dumps(robot.metadata()),names=np.array([path.parent.name+'/'+path.stem.replace('_motion','')]),root_p=np.array(rp)[:,None],root_quat=np.array(rq)[:,None],q=np.array(rj)[:,None],f=np.arange(T)[:,None],alive=np.ones((T,1),bool),clip_len=[len(p)],boxes=boxes[None],ref_root_p=p,ref_root_R=R,ref_q=q,tgt_w=np.zeros((T,1,9,3)),tgt_valid=np.zeros((T,1,9),bool))
    return dict(protocol='free-root joint-position PD; no root support or reference forces; no self collisions as in ACP',completed=not failed,tracking_pass=bool(not failed and np.sqrt(np.mean(np.square(errors)))<=.1 and max(errors)<=.2 and np.sqrt(np.mean(np.square(joint_errors)))<=.15),simulated_s=T/fps,reference_s=(len(p)-1)/fps,error_sampling='post-step against the next reference frame, including terminal transition',root_error_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),root_error_max_m=max(errors),joint_error_rmse_rad=float(np.sqrt(np.mean(np.square(joint_errors)))),torque_saturation_fraction=sat/max(total,1),joint_velocity_exceed_fraction=vel_over/max(total,1),trajectory=str(traj))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--motion-dir',default='data/tienkung3_29dof/motions/php_pilot_final');ap.add_argument('--out',default='docs/tienkung3_29dof/php_pilot/evidence/quality_final.json');ap.add_argument('--record-dir',default='results/php_pilot_final');ap.add_argument('--only',nargs='+');ap.add_argument('--no-pd',action='store_true');ap.add_argument('--export-screened');ap.add_argument('--allow-baseline-self-overlap',action='store_true');args=ap.parse_args()
    robot=get_robot('tienkung3_29dof');root=Path(args.motion_dir);records={}
    report=dict(robot_metadata=robot.metadata(),mujoco=mujoco.__version__,terrain_variant=0,baseline_overlap_pairs=[sorted(x) for x in sorted(BASELINE_PAIRS,key=lambda x:sorted(x))],self_collision_policy='MuJoCo default same-weld and parent-child exclusions; raw and two baseline ankle-overlap exceptions separately reported',clips=records)
    for p in sorted(root.glob('*/*_motion.npz')):
        key=str(p.relative_to(root)).replace('_motion.npz','')
        if args.only and key not in args.only:continue
        q=quality(robot,p)
        if not args.no_pd:q['pd']=playback(robot,p,Path(args.record_dir))
        if args.export_screened and q['pilot_screen_pass'] and (q['strict_geometry_pass'] or args.allow_baseline_self_overlap):
            folder=Path(args.export_screened)/p.parent.name;folder.mkdir(parents=True,exist_ok=True)
            for suffix in ('_motion.npz','_terrain.npy','_intent.npz'):
                src=Path(str(p).replace('_motion.npz',suffix));dst=folder/src.name
                if dst.exists() and sha256(src)!=sha256(dst):raise ValueError('refusing to overwrite a different screened dataset')
                shutil.copyfile(src,dst)
            q['exported_for_exploratory_training']=True
        else:q['exported_for_exploratory_training']=False
        records[key]=q;print(key,json.dumps(q),flush=True)
        dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
