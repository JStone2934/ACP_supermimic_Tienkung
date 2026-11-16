"""Native-pose collision refinement, preserving the complete root pose."""
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from supermimic.acp.php_conversion import STRUCTURAL_PAIRS
from supermimic.acp.robots import CONTACT_NAMES
from supermimic.acp.retarget_fidelity import free_arm_mask,score_arm_trajectories,ARM_NAMES
from tools.check_php_pilot import scene

def velocity_bounds(joints, frame, velocity, fps):
    """Absolute bounds preserving every affected np.gradient velocity stencil."""
    n,dof=joints.shape
    if not 0<=frame<n or fps<=0 or np.shape(velocity)!=(dof,):
        raise ValueError('invalid velocity bound input')
    lower=np.full(dof,-np.inf);upper=np.full(dof,np.inf)
    if n<2:return lower,upper
    for derivative in range(max(0,frame-1),min(n,frame+2)):
        if derivative==0:a,b,scale=0,1,fps
        elif derivative==n-1:a,b,scale=n-2,n-1,fps
        else:a,b,scale=derivative-1,derivative+1,fps/2
        if frame not in (a,b):continue
        other=b if frame==a else a
        radius=np.asarray(velocity)/scale
        lower=np.maximum(lower,joints[other]-radius)
        upper=np.minimum(upper,joints[other]+radius)
    return lower,upper


def violating_arm_phases(report):
    policy=report['policy'];selected=set()
    for point,name in enumerate(ARM_NAMES):
        for row in report['points'][name]['phases']:
            bad=row['rmse_m']>policy['rmse_max_m'] or row['p95_m']>policy['p95_max_m']
            if row['dynamic']:
                bad=bad or not policy['amplitude_ratio_min']<=row['amplitude_ratio']<=policy['amplitude_ratio_max']
                bad=bad or row['correlation']<policy['correlation_min'] or abs(row['best_lag_frames'])>policy['max_lag_frames']
            if bad:selected.add((point//2,row['start'],row['end']))
    return sorted(selected)

def polish(robot,motion,boxes,contact,offsets,max_nfev=120,arm_target=None):
    if (motion.ndim != 2 or motion.shape[1] != 36 or len(motion) == 0 or
        contact.shape != (len(motion), 9) or offsets.shape != (len(motion), 9, 3) or
        not np.isfinite(motion).all() or not np.isfinite(offsets).all() or max_nfev < 1):
        raise ValueError('invalid native polish input')
    if arm_target is not None and (np.shape(arm_target)!=(len(motion),4,3) or not np.isfinite(arm_target).all()):
        raise ValueError('invalid original arm reference')
    model=scene(robot,boxes,self_collision=True);d=mujoco.MjData(model)
    qa,_=robot.joint_addresses(model)
    contact_ids=np.array([model.body(robot.contact_bodies[n][0]).id for n in CONTACT_NAMES])
    arm_ids=np.array([model.body(robot.tracked_bodies[i]).id for i in [9,10,12,13]])
    torso=model.body(robot.tracked_bodies[7]).id
    bodies=sorted({int(model.geom_bodyid[g]) for g in range(model.ngeom) if model.geom_conaffinity[g]})
    pairs=[(a,b) for i,a in enumerate(bodies) for b in bodies[i+1:]
           if frozenset((model.body(a).name,model.body(b).name)) not in STRUCTURAL_PAIRS]
    pair_index={pair:i for i,pair in enumerate(pairs)}
    lo=np.array([model.joint(n).range[0] for n in robot.joint_names])
    hi=np.array([model.joint(n).range[1] for n in robot.joint_names])
    free=free_arm_mask(contact);out=motion.copy();rows=[]
    def set_pose(frame,x):
        d.qpos[:3]=motion[frame,:3]
        d.qpos[3:7]=motion[frame,3:7]
        d.qpos[qa]=motion[frame,7:]+x
        mujoco.mj_forward(model,d)
    def violations():
        penetration=np.zeros(len(pairs))
        for c in d.contact:
            pair=tuple(sorted(int(model.geom_bodyid[g]) for g in c.geom))
            if pair in pair_index:penetration[pair_index[pair]]=max(penetration[pair_index[pair]],-float(c.dist))
        return penetration
    selected=[]
    for frame in range(len(motion)):
        set_pose(frame,np.zeros(29))
        if violations().max()>.002:selected.append(frame)
    for frame in selected:
        started=time.monotonic();set_pose(frame,np.zeros(29))
        before=float(violations().max());active=contact[frame]
        def poses():
            R=d.xmat[torso].reshape(3,3).copy()
            points=d.xpos[contact_ids]+np.einsum('bij,bj->bi',d.xmat[contact_ids].reshape(-1,3,3),offsets[frame])
            arms=(d.xpos[arm_ids]-d.xpos[torso])@R
            return points,R,arms
        support0,torso0,arms0=poses()
        def residual(x):
            set_pose(frame,x);support,R,arms=poses()
            return np.concatenate((1000*np.maximum(0,violations()-.0015),
                100*(support[active]-support0[active]).ravel(),100*(R-torso0).ravel(),
                20*(arms[free[frame]]-arms0[free[frame]]).ravel(),.1*x))
        def jacobian(x):
            eps=1e-5;cols=[]
            for i in range(len(x)):
                dx=np.zeros_like(x);dx[i]=eps
                cols.append((residual(x+dx)-residual(x-dx))/(2*eps))
            return np.stack(cols,axis=1)
        lower=np.maximum(-.4,lo-motion[frame,7:]+1e-5)
        upper=np.minimum(.4,hi-motion[frame,7:]-1e-5)
        vlo,vhi=velocity_bounds(out[:,7:],frame,np.asarray(robot.velocity)*(1-1e-7),1/robot.control_dt)
        lower=np.maximum(lower,vlo-motion[frame,7:])
        upper=np.minimum(upper,vhi-motion[frame,7:])
        if np.any(lower>=upper):
            rows.append(dict(frame=frame,before_m=before,after_m=before,applied=False,reason='infeasible_velocity_bounds'))
            continue
        initial=np.clip(np.zeros(29),lower+1e-12,upper-1e-12)
        result=least_squares(residual,initial,jac=jacobian,bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        set_pose(frame,result.x);support,R,arms=poses();after=float(violations().max())
        improved=after<before and np.isfinite(result.x).all()
        if improved:
            out[frame,7:]+=result.x
        rows.append(dict(frame=frame,before_m=before,after_m=after,applied=bool(improved),
            optimizer_success=bool(result.success),nfev=result.nfev,seconds=time.monotonic()-started,
            support_displacement_m=np.linalg.norm(support[active]-support0[active],axis=1).tolist(),
            torso_change_deg=float(Rotation.from_matrix(torso0.T@R).magnitude()*180/np.pi),
            free_arm_displacement_m=np.linalg.norm(arms[free[frame]]-arms0[free[frame]],axis=1).tolist(),
            root_displacement_m=[0.,0.,0.],max_joint_change_rad=float(np.abs(result.x).max())))
    arm_rows=[]
    if arm_target is not None:
        def arm_poses(frame):
            d.qpos[:7]=out[frame,:7];d.qpos[qa]=out[frame,7:];mujoco.mj_forward(model,d)
            return (d.xpos[arm_ids]-d.xpos[torso])@d.xmat[torso].reshape(3,3)
        actual=np.stack([arm_poses(i) for i in range(len(out))])
        report=score_arm_trajectories(arm_target,actual,free)
        for side,a,z in violating_arm_phases(report):
            joints=np.arange(15,22) if side==0 else np.arange(22,29)
            points=slice(2*side,2*side+2)
            for frame in range(a,z):
                assert free[frame,2*side:2*side+2].all()
                original=out[frame,7:].copy()
                before_points=arm_poses(frame)[points].copy();before_collision=float(violations().max())
                target=arm_target[frame,points]
                before_error=float(np.square(before_points-target).sum())
                def arm_residual(x):
                    d.qpos[:7]=out[frame,:7];d.qpos[qa]=original;d.qpos[qa[joints]]=original[joints]+x
                    mujoco.mj_forward(model,d)
                    points_now=((d.xpos[arm_ids]-d.xpos[torso])@d.xmat[torso].reshape(3,3))[points]
                    return np.r_[100*(points_now-target).ravel(),1000*np.maximum(0,violations()-.0015),.1*x]
                vlo,vhi=velocity_bounds(out[:,7:],frame,np.asarray(robot.velocity)*(1-1e-7),1/robot.control_dt)
                lower=np.maximum.reduce([np.full(7,-.25),lo[joints]-original[joints]+1e-5,vlo[joints]-original[joints]])
                upper=np.minimum.reduce([np.full(7,.25),hi[joints]-original[joints]-1e-5,vhi[joints]-original[joints]])
                if np.any(lower>=upper):
                    arm_rows.append(dict(side=side,frame=frame,applied=False,reason='infeasible_velocity_bounds'))
                    continue
                initial=np.clip(np.zeros(7),lower+1e-12,upper-1e-12)
                result=least_squares(arm_residual,initial,jac='3-point',diff_step=1e-5,bounds=(lower,upper),max_nfev=60,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                residual_value=arm_residual(result.x);after_error=float(np.square(residual_value[:6]/100).sum());after_collision=float(violations().max())
                applied=after_error<before_error and after_collision<=max(.002,before_collision+1e-8) and np.isfinite(result.x).all()
                if applied:
                    out[frame,7+joints]+=result.x
                    selected.append(frame)
                arm_rows.append(dict(side=side,frame=frame,applied=bool(applied),point_error_before_m2=before_error,point_error_after_m2=after_error,collision_before_m=before_collision,collision_after_m=after_collision,nfev=result.nfev,optimizer_success=bool(result.success)))
    selected=sorted(set(selected))
    np.testing.assert_array_equal(out[:,:7],motion[:,:7])
    unchanged=np.ones(len(motion),bool);unchanged[selected]=False
    np.testing.assert_array_equal(out[unchanged],motion[unchanged])
    return out,dict(policy='native-joint-polish-fixed-root-velocity-free-arm-v1',selected_frames=selected,rows=rows,arm_rows=arm_rows)
