"""Optimistic instantaneous reference contact-wrench feasibility screen, NOT a trajectory solver."""
import sys,json
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import linprog
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));OUT=ROOT/'results/climb_diagnosis_20261111'
from tools.check_php_pilot import scene
from supermimic.acp.robots import get_robot
from supermimic.acp.motion import load_php_npz
from supermimic.acp.terrain import load_terrain
robot=get_robot('tienkung3_29dof');data=ROOT/'data/tienkung3_29dof/stage1/php_all_variants_20261110/motions'
names=json.load(open(ROOT/'results/stage_training_analysis_20261111/cohort.json'))['clips'];rng=np.random.default_rng(20261111)
chosen=[]
for group in sorted({n.split('/')[0] for n in names}):
    candidates=[n for n in names if n.startswith(group+'/') and n.endswith('variant_00')]
    chosen+=sorted(rng.choice(candidates,4,replace=False).tolist())
report=dict(protocol='Instantaneous inverse-dynamics LP on reference poses; friction=1, eight inscribed friction rays, contacts within 1 cm admitted optimistically. No contact-acceleration or future stability constraints; success is necessary-screen only, not dynamic tracking proof. static uses zero velocity and acceleration; dynamic uses finite differences of qpos at 50 Hz.',clips={})
def feasible(G,b,va,eff,limits):
    if G.shape[1]==0:return False
    kw={}
    if limits:kw=dict(A_ub=np.concatenate((G[va],-G[va])),b_ub=np.concatenate((b[va]+eff,-b[va]+eff)))
    r=linprog(np.ones(G.shape[1])*1e-6,A_eq=G[:6],b_eq=b[:6],bounds=(0,None),method='highs',**kw)
    return bool(r.success)
for name in chosen:
    path=data/(name+'_motion.npz');p,quat,q,fps=load_php_npz(path,robot);boxes=load_terrain(str(path))[0];boxes=boxes[boxes[:,5]>0]
    m=scene(robot,boxes,False,True);d=mujoco.MjData(m);qa,va=robot.joint_addresses(m)
    # Conservative in declaring infeasibility: admit near contacts that do not yet exert physical force.
    m.geom_margin[:]=.01
    pos=np.tile(m.qpos0,(len(p),1));pos[:,:3]=p;pos[:,3:7]=quat;pos[:,qa]=q
    vel=np.zeros((len(p),m.nv))
    for t in range(len(p)):
        lo=max(t-1,0);hi=min(t+1,len(p)-1);mujoco.mj_differentiatePos(m,vel[t],(hi-lo)/fps,pos[lo],pos[hi])
    acc=np.gradient(vel,axis=0)*fps
    eff=np.array(robot.effort); rows=[];M=np.zeros((m.nv,m.nv))
    for t in range(0,len(p),10):
        d.qpos[:]=pos[t];d.qvel[:]=vel[t];mujoco.mj_forward(m,d)
        try:mujoco.mj_fullM(m,d,M)
        except TypeError:mujoco.mj_fullM(m,M,d.qM)
        dyn=M@acc[t]+d.qfrc_bias-d.qfrc_passive
        cols=[];bodies=[];dists=[]
        for c in d.contact:
            b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            if not ((b1==0)^(b2==0)) or c.dist>.01:continue
            bid=b2 if b1==0 else b1;sgn=1 if b1==0 else -1
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,c.pos,bid)
            n,u,v=c.frame.reshape(3,3);n=n*sgn
            for angle in np.arange(8)*np.pi/4:
                force=n+np.cos(angle)*u+np.sin(angle)*v
                cols.append(jp.T@force)
            bodies.append(m.body(bid).name);dists.append(float(c.dist))
        G=np.stack(cols,1) if cols else np.zeros((m.nv,0))
        d.qvel[:]=0;mujoco.mj_forward(m,d);static=d.qfrc_bias-d.qfrc_passive
        rows.append(dict(frame=t,phase=t/(len(p)-1),bodies=sorted(set(bodies)),contacts=len(bodies),min_distance=min(dists) if dists else None,base_acc=acc[t,:3].tolist(),static_wrench=feasible(G,static,va,eff,False),static_torque=feasible(G,static,va,eff,True),dynamic_wrench=feasible(G,dyn,va,eff,False),dynamic_torque=feasible(G,dyn,va,eff,True),required_base_force=dyn[:3].tolist()))
    report['clips'][name]=dict(frames=len(p),rows=rows,max_root_acc=float(np.linalg.norm(acc[:,:3],axis=1).max()),p95_root_acc=float(np.quantile(np.linalg.norm(acc[:,:3],axis=1),.95)))
    (OUT/'support.json').write_text(json.dumps(report))
    print(name,len(rows),{k:round(np.mean([r[k] for r in rows]),3) for k in ['static_wrench','static_torque','dynamic_wrench','dynamic_torque']},flush=True)
