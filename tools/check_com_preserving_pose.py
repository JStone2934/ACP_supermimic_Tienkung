"""Offline leg IK preserving foot attitude and restoring COM relative to support."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('old','source','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();torch.set_num_threads(2)
    old=torch.load(a.old,map_location='cpu',weights_only=False)
    source=torch.load(a.source,map_location='cpu',weights_only=False)
    task=WideAimingTask('cpu',source['config']);previous=WideAimingTask('cpu',old['config'])
    assert torch.equal(task.initial_quat,previous.initial_quat) and task.initial_height==previous.initial_height
    model=task.robot.get_spec().compile();data=mujoco.MjData(model)
    addr,dof=task.robot.joint_addresses(model)
    names=[task.robot.contact_bodies[n][0] for n in ('left_foot','right_foot')]
    foot_ids=[model.body(n).id for n in names]
    data.qpos[:3]=[0,0,task.initial_height];data.qpos[3:7]=task.initial_quat
    def forward(q):
        data.qpos[addr]=q;mujoco.mj_forward(model,data)
        return data.xpos[foot_ids].copy(),data.xmat[foot_ids].reshape(2,3,3).copy(),data.subtree_com[0].copy()
    def loads(q):
        forward(q);jac=[]
        for body in foot_ids:
            jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacBody(model,data,jp,jr,body);jac.extend((jp,jr))
        jac=np.concatenate(jac);w=np.tile([1.,1.,1.,.04,.04,.04],2);A=jac[:,:6].T
        force=(w[:,None]*np.linalg.pinv(A*w[None,:]))@data.qfrc_bias[:6]
        tau=data.qfrc_bias-jac.T@force
        assert np.linalg.norm(tau[:6])<1e-5
        f=force.reshape(2,6)
        assert (f[:,2]>0).all() and (np.linalg.norm(f[:,:2],axis=1)<.5*f[:,2]).all()
        assert (np.abs(f[:,3:5])/f[:,2,None]<.08).all()
        return tau[dof].copy(),force
    qold=previous.initial_q.numpy().astype(float);qbase=task.initial_q.numpy().astype(float)
    feet_old,rot_old,com_old=forward(qold);feet_base,rot_base,com_base=forward(qbase)
    np.testing.assert_allclose(feet_old,feet_base,atol=1e-8)
    old_relative=com_old[:2]-feet_old[:,:2].mean(0)
    base_relative=com_base[:2]-feet_base[:,:2].mean(0)
    tau_old,wrench_old=loads(qold);rows=[]
    for fraction in (.5,1.):
        desired=base_relative+fraction*(old_relative-base_relative)
        def residual(x):
            q=qbase.copy();q[:12]=x[:12]
            feet,rot,com=forward(q)
            target=feet_base.copy();target[:,:2]+=x[12:14]
            angles=Rotation.from_matrix(np.swapaxes(rot_base,1,2)@rot).as_rotvec()
            return np.concatenate(((feet-target).ravel(),.2*angles.ravel(),com[:2]-feet[:,:2].mean(0)-desired))
        lo=np.r_[np.maximum(task.mdp.soft_lo[:12].numpy(),qbase[:12]-.05),[-.03,-.03]]
        hi=np.r_[np.minimum(task.mdp.soft_hi[:12].numpy(),qbase[:12]+.05),[.03,.03]]
        solution=least_squares(residual,np.r_[qbase[:12],[0.,0.]],bounds=(lo,hi),
                               ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=300)
        assert solution.success and np.max(np.abs(residual(solution.x)))<1e-6
        q=qbase.copy();q[:12]=solution.x[:12];feet,rot,com=forward(q)
        tau,wrench=loads(q);dq=q-qbase;pd_delta=dq+(tau-tau_old)/np.array(task.robot.kp)
        state=list(task.initial_state(1));state[4]=torch.tensor(q,dtype=torch.float32)[None]
        with torch.no_grad(): bp,br,_,_=task.kin.forward(*state,all_bodies=True)
        ids=[task.kin.body_names.index(n) for n in names]
        pos_error=float(np.max(np.abs(bp[0,ids].numpy()-feet)))
        rot_error=float(np.max(np.abs(br[0,ids].numpy()-rot)))
        assert pos_error<2e-6 and rot_error<2e-6
        rows.append(dict(fraction=fraction,initial_q=q.tolist(),leg_q_delta_rad=dq[:12].tolist(),
            max_leg_q_delta_rad=float(np.max(np.abs(dq[:12]))),foot_shift_xy_m=solution.x[12:14].tolist(),
            com_relative_support_xy_m=(com[:2]-feet[:,:2].mean(0)).tolist(),
            maximum_solver_residual=float(np.max(np.abs(residual(solution.x)))),
            independent_fk_position_error_m=pos_error,independent_fk_rotation_matrix_error=rot_error,
            delta_leg_pd_target_rad=pd_delta[:12].tolist(),contact_wrench=wrench.tolist()))
    report=dict(scope=__doc__+' No physical acceptance.',old_sha256=sha256(a.old),source_sha256=sha256(a.source),
        joint_names=list(task.robot.joint_names),old_com_relative_support_xy_m=old_relative.tolist(),
        uncorrected_com_relative_support_xy_m=base_relative.tolist(),old_contact_wrench=wrench_old.tolist(),
        rows=rows,physical_acceptance=False)
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
