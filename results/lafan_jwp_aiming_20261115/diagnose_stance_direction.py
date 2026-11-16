"""Offline native-joint inward-foot displacement Jacobian; no policy mutation."""
from pathlib import Path
import json
import mujoco
import numpy as np
from supermimic.acp.robots import get_robot, sha256


def main():
    base=Path(__file__).resolve().parent
    source=base/'evaluation_observed_support_preactor/iteration_000200/seed_17.npz'
    robot=get_robot('tienkung3_29dof');model=robot.get_spec().compile();data=mujoco.MjData(model)
    addresses,dof=robot.joint_addresses(model)
    with np.load(source) as z:
        data.qpos[:3]=z['root'][150,0];data.qpos[3:7]=z['root_quat'][150,0]
        data.qpos[addresses]=z['q'][150,0]
    mujoco.mj_forward(model,data)
    initial=data.qpos.copy();root=model.body(robot.root_body).id
    lateral=data.xmat[root].reshape(3,3)[:,1].copy();lateral[2]=0;lateral/=np.linalg.norm(lateral)
    correction=np.zeros(robot.num_joints);rows=[]
    # Semantic contact bodies, not incidental tracked-body ordering.
    for name,sign in [('left_foot',-1),('right_foot',1)]:
        body_name=robot.contact_bodies[name][0]
        if not isinstance(body_name,str):raise TypeError((name,body_name))
        bid=model.body(body_name).id;b=bid;joints=[]
        while b!=root:
            for jid in range(model.body_jntadr[b],model.body_jntadr[b]+model.body_jntnum[b]):
                joint_name=model.joint(jid).name
                if joint_name in robot.joint_names:joints.append(robot.joint_names.index(joint_name))
            b=int(model.body_parentid[b])
            if b==0:raise ValueError('Foot chain does not reach root')
        joints=sorted(joints);assert len(joints)==6,(name,joints)
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacBody(model,data,jp,jr,bid)
        J=np.concatenate([jp[:,dof[joints]],jr[:,dof[joints]]])
        desired=sign*.01*lateral
        dq=J.T@np.linalg.solve(J@J.T+1e-6*np.eye(6),np.r_[desired,[0,0,0]])
        correction[joints]=dq
        rows.append(dict(foot=name,body=body_name,joints=[robot.joint_names[i] for i in joints],
                         requested_delta_m=desired.tolist(),initial_position=data.xpos[bid].tolist(),
                         initial_rotation=data.xmat[bid].reshape(3,3).tolist(),condition=float(np.linalg.cond(J))))
    data.qpos[addresses]+=correction;mujoco.mj_forward(model,data)
    for row in rows:
        bid=model.body(row['body']).id
        delta=data.xpos[bid]-row.pop('initial_position')
        rot=data.xmat[bid].reshape(3,3)@np.asarray(row.pop('initial_rotation')).T
        row.update(actual_delta_m=delta.tolist(),linearization_error_m=float(np.linalg.norm(delta-row['requested_delta_m'])),
                   orientation_change_rad=float(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1))))
    report=dict(scope='Offline 1 cm inward-foot kinematics only; no physical acceptance or policy mutation',
                source_trajectory=str(source),source_sha256=sha256(source),frame=150,scenario=0,
                robot_metadata=robot.metadata(),joint_order=robot.joint_names,
                source_native_q=initial[addresses].tolist(),
                native29joint_target_correction_rad=correction.tolist(),feet=rows)
    (base/'stance_inward_direction_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
