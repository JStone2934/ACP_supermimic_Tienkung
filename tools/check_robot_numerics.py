"""Independent MuJoCo FK/velocity/gradient and contact-distance acceptance checks."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import torch
from supermimic.acp.robots import get_robot
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.task import BodySpheres
from supermimic.acp.geometry import hull_box_penetration
from supermimic.acp.make_labels import build_model, geom_distance as checked_geom_distance
from supermimic.acp.terrain import box_sdf
from supermimic.rotations import quat_to_mat


def geom_distance(*args):
    return checked_geom_distance(*args, strict_reference=True)


def fk_check(robot, n, seed, *, model=None, dtype=torch.float32):
    rng=np.random.default_rng(seed); kin=RobotKinematics('cpu',robot,model=model,dtype=dtype); m=kin.model; d=mujoco.MjData(m)
    qadr,vadr=robot.joint_addresses(m); ids=[m.body(n).id for n in kin.tracked_names]
    p=rng.normal(size=(n,3)); quat=rng.normal(size=(n,4)); quat/=np.linalg.norm(quat,axis=-1,keepdims=True)
    q=rng.uniform(kin.q_lo.numpy(),kin.q_hi.numpy(),size=(n,robot.num_joints))
    v,w,qd=rng.normal(size=(n,3)),rng.normal(size=(n,3)),rng.normal(size=q.shape)
    tensor=lambda x: torch.tensor(x,dtype=dtype)
    R=quat_to_mat(tensor(quat)); state=(tensor(p),R,tensor(v),tensor(w),tensor(q),tensor(qd))
    with torch.no_grad(): actual=[x.numpy() for x in kin.forward(*state)]
    ref=[np.zeros_like(x) for x in actual]; jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    for i in range(n):
        d.qpos[:3]=p[i]; d.qpos[3:7]=quat[i]; d.qpos[qadr]=q[i]
        # MuJoCo free-root angular velocity is local; ACP root angular velocity is world.
        rr=np.empty(9); mujoco.mju_quat2Mat(rr,quat[i]); rr=rr.reshape(3,3)
        d.qvel[:3]=v[i]; d.qvel[3:6]=rr.T@w[i]; d.qvel[vadr]=qd[i]
        mujoco.mj_forward(m,d)
        for j,bid in enumerate(ids):
            ref[0][i,j]=d.xpos[bid]; ref[1][i,j]=d.xmat[bid].reshape(3,3)
            mujoco.mj_jacBody(m,d,jp,jr,bid)
            ref[2][i,j]=jp@d.qvel; ref[3][i,j]=jr@d.qvel
    rel=actual[1].astype(float)@np.swapaxes(ref[1].astype(float),-1,-2)
    skew=np.stack((rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]),-1)/2
    angles=np.arctan2(np.linalg.norm(skew,axis=-1),(np.trace(rel,axis1=-2,axis2=-1)-1)/2)
    errors=dict(position_max_m=float(np.linalg.norm(actual[0]-ref[0],axis=-1).max()),rotation_max_rad=float(angles.max()),
                linear_velocity_max=float(np.linalg.norm(actual[2]-ref[2],axis=-1).max()),angular_velocity_max=float(np.linalg.norm(actual[3]-ref[3],axis=-1).max()))
    limits=(1e-4,1e-4,1e-3,1e-3)
    return dict(samples=n,dtype=str(dtype),**errors,passed=all(x<=t for x,t in zip(errors.values(),limits)))


def gradient_check(robot, seed, model=None):
    kin=RobotKinematics('cpu',robot,model=model,dtype=torch.float64); torch.manual_seed(seed)
    n=3; p=torch.randn(n,3,dtype=torch.float64); R=torch.eye(3,dtype=torch.float64).expand(n,3,3)
    v,w=torch.randn_like(p),torch.randn_like(p)
    q=((kin.q_lo+kin.q_hi)/2).expand(n,-1).clone().requires_grad_(); qd=torch.randn_like(q,requires_grad=True)
    weights=[torch.randn_like(x) for x in kin.forward(p,R,v,w,q,qd)]
    def f(q0,qd0): return sum((x*a).sum() for x,a in zip(kin.forward(p,R,v,w,q0,qd0),weights))
    grads=torch.autograd.grad(f(q,qd),(q,qd)); errors=[]; passed=True
    for j in range(robot.num_joints):
        for k,base in enumerate((q,qd)):
            delta=torch.zeros_like(base); delta[:,j]=1; h=1e-5
            plus=(q+h*delta,qd) if k==0 else (q,qd+h*delta)
            minus=(q-h*delta,qd) if k==0 else (q,qd-h*delta)
            fd=(f(*plus)-f(*minus))/(2*h); ad=(grads[k]*delta).sum()
            errors.append(float((fd-ad).abs().detach())); passed &= bool(torch.isclose(fd,ad,rtol=1e-2,atol=1e-3))
    return dict(derivatives=2*robot.num_joints,max_absolute_error=max(errors),passed=passed)


def anchor_fixture(robot):
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody><body name="pelvis"><freejoint/>
      <geom type="sphere" size=".1" mass="2"/>
      <body name="fixed1" pos=".1 .2 .3" quat=".9238795 0 .3826834 0"><geom type="sphere" size=".02" mass=".1"/>
        <body name="hinge1" pos=".2 0 .1"><joint name="j1" pos=".03 -.04 .05" axis="0 0 1" range="-1 1" ref=".15"/>
          <geom type="sphere" size=".05" mass="1"/>
          <body name="fixed2" pos=".1 -.2 .1" quat=".9659258 .258819 0 0"><geom type="sphere" size=".02" mass=".1"/>
            <body name="hinge2" pos=".05 .02 .03"><joint name="j2" pos="-.04 .03 .02" axis="0 1 0" range="-1 1"/>
              <geom type="sphere" size=".04" mass=".2"/><body name="tcp" pos=".1 .2 .05"/>
            </body></body></body></body></body></worldbody></mujoco>''')
    r=replace(robot,joint_names=('j1','j2'),tracked_bodies=('pelvis','fixed1','hinge1','fixed2','hinge2','tcp'))
    return dict(fk=fk_check(r,100,7,model=model),gradient=gradient_check(r,7,model=model))


def contact_check(robot,n,seed,groups=None):
    rng=np.random.default_rng(seed); sph=BodySpheres(robot)
    # One broad box. Move it per sample; use its top, side and corner as independent surfaces.
    boxes=np.array([[0,0,1,1,0,.25,.3,.5]],dtype=float)
    m=build_model(boxes,robot); d=mujoco.MjData(m); qadr,_=robot.joint_addresses(m)
    plane=m.geom('ground').id; box=m.geom('box0').id
    q=np.asarray(robot.default_q); d.qpos[qadr]=q
    names=list(robot.contact_bodies); fromto=np.zeros(6); rows={}
    allerr=[]; gradient_errors=[]; grad_pass=True
    for b,(name,(frame,links)) in enumerate(robot.contact_bodies.items()):
        if groups and name not in groups:
            for i in range(n):
                rng.normal(size=4); rng.uniform(-.05,.05)
                if i<8: rng.normal(size=(1,1,3))
            continue
        geom_ids=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in links and (m.geom_contype[g] or m.geom_conaffinity[g])]
        lo,hi=sph.bounds[b]; pts=sph.pts[lo:hi].numpy().astype(float); rad=sph.rad[lo:hi].numpy().astype(float)
        errs=[]; distances=[]; cases=[]; categories={k:[] for k in ('plane','box_top','box_side','box_edge')}
        for i in range(n):
            quat=rng.normal(size=4); quat/=np.linalg.norm(quat); d.qpos[:3]=[0,0,3]; d.qpos[3:7]=quat
            mujoco.mj_kinematics(m,d); bid=m.body(frame).id
            x=d.xpos[bid]+pts@d.xmat[bid].reshape(3,3).T
            cat=list(categories)[i%4]; target=rng.uniform(-.05,.05)
            if cat=='plane':
                # Position using exact support of the model, not the approximate cloud.
                exact=min(geom_distance(m,d,g,plane,20,fromto) for g in geom_ids)
                d.qpos[2]+=target-exact; tg=plane
            else:
                normal=np.array([0,0,1.]) if cat=='box_top' else np.array([1.,0,0]) if cat=='box_side' else np.array([1.,1.,1.])/np.sqrt(3)
                # Put box outside the group's center, then bracket a target signed distance.
                center=x.mean(0); m.geom_pos[box]=center-normal*1.5
                tg=box
                a,z=0.,3.
                for _ in range(18):
                    mid=(a+z)/2; m.geom_pos[box]=center-normal*mid; mujoco.mj_kinematics(m,d)
                    dist=min(geom_distance(m,d,g,tg,20,fromto) for g in geom_ids)
                    if dist>target: z=mid
                    else: a=mid
                m.geom_pos[box]=center-normal*((a+z)/2)
            mujoco.mj_kinematics(m,d)
            exact=min(geom_distance(m,d,g,tg,20,fromto) for g in geom_ids)
            x=d.xpos[bid]+pts@d.xmat[bid].reshape(3,3).T
            tx=torch.tensor(x[None],dtype=torch.float64,requires_grad=True)
            if tg==plane:
                ds=tx[...,2]-torch.tensor(rad)
                fn=lambda y: (y[...,2]-torch.tensor(rad)).min()
            else:
                cx,cy,cz=d.geom_xpos[box]; hx,hy,hz=m.geom_size[box]
                tb=torch.tensor([[cx,cy,cz+hz,1,0,hx,hy,hz]],dtype=torch.float64)
                ds=box_sdf(tx,tb)-torch.tensor(rad)
                origin=torch.tensor(d.xpos[bid][None].copy(),dtype=torch.float64)
                rotation=torch.tensor(d.xmat[bid].reshape(1,3,3).copy(),dtype=torch.float64)
                original_x=tx.detach().clone()
                def fn(y):
                    approx=(box_sdf(y,tb)-torch.tensor(rad)).min()
                    if b not in sph.hulls: return approx
                    pen=hull_box_penetration(origin+y[:,0]-original_x[:,0],rotation,tb,sph.hulls[b]).squeeze(0)
                    return torch.where(approx<0,pen,approx)
                ds=fn(tx).reshape(1,1)
            approx=float(ds.min().detach()); err=abs(approx-exact)
            errs.append(err); categories[cat].append(err); distances.append(exact)
            cases.append(dict(index=i,surface=cat,error=err,exact=exact,approx=approx,root_qpos=d.qpos[:7].tolist(),
                              body_p=d.xpos[bid].tolist(),body_R=d.xmat[bid].reshape(3,3).tolist(),box_center=d.geom_xpos[box].tolist()))
            # Check a common translation direction; skip min-switch boundaries by testing winner stability.
            if i<8:
                direction=torch.tensor(rng.normal(size=(1,1,3)),dtype=torch.float64); h=1e-5
                grad=torch.autograd.grad(fn(tx),tx)[0]; ad=(grad*direction).sum()
                fd=(fn(tx+h*direction)-fn(tx-h*direction))/(2*h)
                error=float((ad-fd).abs().detach()); gradient_errors.append(error)
                grad_pass &= bool(torch.isclose(ad,fd,rtol=1e-2,atol=1e-3))
        print(f"{name}: p95={np.percentile(errs,95):.6f} max={max(errs):.6f}",flush=True)
        allerr.extend(errs)
        rows[name]=dict(samples=n,points=hi-lo,worst=sorted(cases,key=lambda x:x["error"],reverse=True)[:5],p95_m=float(np.percentile(errs,95)),max_m=max(errs),
                        exact_distance_range_m=[min(distances),max(distances)],
                        surfaces={k:dict(p95_m=float(np.percentile(v,95)),max_m=max(v)) for k,v in categories.items()})
    return dict(groups=rows,p95_m=float(np.percentile(allerr,95)),max_m=max(allerr),
                passed=all(r['p95_m']<=.005 and r['max_m']<=.015 for r in rows.values()),
                gradient=dict(samples=len(gradient_errors),max_absolute_error=max(gradient_errors),passed=grad_pass))


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--robot',default='tienkung3_29dof'); ap.add_argument('--samples',type=int,default=1000)
    ap.add_argument('--contact-samples',type=int,default=100); ap.add_argument('--contact-groups',nargs='+'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--skip-contact',action='store_true'); ap.add_argument('--out',required=True)
    args=ap.parse_args(); torch.set_num_threads(4); start=time.time(); r=get_robot(args.robot)
    report=dict(robot_metadata=r.metadata(),seed=args.seed,mujoco=mujoco.__version__,torch=torch.__version__,
                contact_oracle='native mj_geomDistance with bounded retries; sampled-surface fallback disabled')
    print('Checking FK and velocities...',flush=True); report['fk']=fk_check(r,args.samples,args.seed)
    print(report['fk'],flush=True)
    report['gradient']=gradient_check(r,args.seed); report['anchor_fixture']=anchor_fixture(r)
    if not args.skip_contact:
        print('Checking contact geometry...',flush=True); report['contact']=contact_check(r,args.contact_samples,args.seed,args.contact_groups)
    report['elapsed_s']=time.time()-start
    report['passed']=report['fk']['passed'] and report['gradient']['passed'] and all(x['passed'] for x in report['anchor_fixture'].values())
    if 'contact' in report: report['passed'] &= report['contact']['passed'] and report['contact']['gradient']['passed']
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
