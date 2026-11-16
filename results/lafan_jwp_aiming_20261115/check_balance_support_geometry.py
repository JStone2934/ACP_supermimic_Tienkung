"""Read-only mass/support geometry diagnostic on saved physical trajectories.

Static COM projection is descriptive only: it does not prove dynamic stability,
and a moving robot may legitimately leave its instantaneous support polygon.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import ConvexHull,QhullError
from supermimic.acp.robots import get_robot


def main():
    b=Path('results/lafan_jwp_aiming_20261115');robot=get_robot('tienkung3_29dof')
    spec=robot.get_spec();spec.worldbody.add_geom(name='diagnostic_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05))
    model=spec.compile();data=mujoco.MjData(model);addresses,_=robot.joint_addresses(model)
    floor=model.geom('diagnostic_floor').id
    feet={model.body(robot.contact_bodies[name][0]).id for name in ['left_foot','right_foot']}
    root_body=int(model.jnt_bodyid[0]);rows=[]
    for label,path in [('pre_actor',b/'evaluation_balance_recovery_quiet/iteration_001500/seed_17.npz'),
                       ('failed_h8',b/'evaluation_balance_recovery_reward_gradient/iteration_001750/seed_17.npz')]:
        z=np.load(path);samples=[]
        for seconds in [0,.16,.32,.64,1,2,3,10,20,30]:
            t=round(seconds/.02);data.qpos[:3]=z['root'][t,0];data.qpos[3:7]=z['root_quat'][t,0]
            data.qpos[addresses]=z['q'][t,0];data.qvel[:]=0;mujoco.mj_forward(model,data)
            com=data.subtree_com[root_body].copy();points=[]
            for contact in data.contact[:data.ncon]:
                g1,g2=map(int,contact.geom)
                other=g2 if g1==floor else g1 if g2==floor else None
                if other is not None and model.geom_bodyid[other] in feet and contact.dist<.005:
                    points.append(contact.pos[:2].copy())
            margin=None
            if len(points)>=3:
                try:
                    hull=ConvexHull(np.array(points))
                    margin=float(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2]).max())
                except QhullError:pass
            samples.append(dict(time_s=seconds,root_tilt_deg=float(z['tilt'][t,0]),com_world_m=com.tolist(),
                foot_floor_contact_points=len(points),static_com_support_margin_m=margin))
        rows.append(dict(label=label,trajectory=str(path),scenario=0,samples=samples))
    result=dict(scope=__doc__,robot_mass_kg=float(model.body_subtreemass[root_body]),rows=rows,policy_acceptance=False)
    (b/'balance_support_geometry.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
