"""Offline COM/contact geometry on recorded physical trajectories; no policy mutation."""
from pathlib import Path
import json
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from supermimic.acp.robots import get_robot, sha256


def main():
    base=Path('/home/js/robot/locomotion/standing-aiming/results/lafan_jwp_aiming_20261115')
    paths={'source3250':base/'evaluation_balance_recovery_conservative_3250/iteration_003250/seed_17.npz',
           'observed250':base/'evaluation_observed_support/iteration_000250/seed_17.npz'}
    robot=get_robot('tienkung3_29dof');spec=robot.get_spec()
    spec.worldbody.add_geom(name='diagnostic_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05))
    model=spec.compile();data=mujoco.MjData(model);addresses,_=robot.joint_addresses(model)
    root=model.body(robot.root_body).id;floor=model.geom('diagnostic_floor').id
    def pose(z,t,i):
        data.qpos[:3]=z['root'][t,i];data.qpos[3:7]=z['root_quat'][t,i];data.qpos[addresses]=z['q'][t,i]
        mujoco.mj_forward(model,data)
        return data.subtree_com[root].copy()
    def margin(points,p):
        points=np.unique(np.round(np.asarray(points)[:,:2],6),axis=0) if len(points) else np.zeros((0,2))
        if len(points)<3:return None
        try:hull=ConvexHull(points)
        except Exception:return None
        # scipy outward normals are unit length; positive means inside all edges.
        return float(np.min(-(hull.equations[:,:2]@p+hull.equations[:,2])))
    result={}
    for label,path in paths.items():
        with np.load(path) as archive:
            z={k:archive[k].copy() for k in ('root','root_quat','q','feet','tilt')}
        rows=[]
        for sec in (3,5,10,12,13,14,14.4):
            t=round(sec/.02);values=[]
            for i in range(z['root'].shape[1]):
                cm0=pose(z,t-1,i);cm1=pose(z,t+1,i);com=pose(z,t,i)
                velocity=(cm1-cm0)/.04
                xcom=com[:2]+velocity[:2]*np.sqrt(max(.01,com[2])/9.81)
                contacts=[c.pos.copy() for c in data.contact if floor in c.geom and c.dist<=.01]
                centre=z['feet'][t,i].mean(0)
                values.append(dict(com_minus_foot_midpoint_xy_m=(com[:2]-centre[:2]).tolist(),
                                   com_margin_m=margin(contacts,com[:2]),xcom_margin_m=margin(contacts,xcom),
                                   contact_count=len(contacts),com_speed_m_s=float(np.linalg.norm(velocity[:2])),
                                   tilt_deg=float(z['tilt'][t,i])))
            valid=[v for v in values if v['com_margin_m'] is not None]
            rows.append(dict(time_s=sec,valid_support_hulls=len(valid),
                             median_com_margin_m=float(np.median([v['com_margin_m'] for v in valid])) if valid else None,
                             median_xcom_margin_m=float(np.median([v['xcom_margin_m'] for v in valid])) if valid else None,
                             outside_com_hulls=sum(v['com_margin_m']<0 for v in valid),
                             outside_xcom_hulls=sum(v['xcom_margin_m']<0 for v in valid),
                             median_com_speed_m_s=float(np.median([v['com_speed_m_s'] for v in values])),
                             scenario0=values[0]))
        result[label]=dict(trajectory=str(path),trajectory_sha256=sha256(path),rows=rows)
    report=dict(scope='Descriptive offline COM/contact-position geometry. Contact forces unavailable; hull is not a certified support polygon. XCoM uses the linear inverted-pendulum approximation, not a proof of dynamic stability.',
                physical_acceptance=False,weights_changed=False,comparisons=result)
    (base/'observed_support_com_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({label:[{k:v for k,v in row.items() if k!='scenario0'} for row in x['rows']] for label,x in result.items()},indent=2))


if __name__=='__main__':main()
