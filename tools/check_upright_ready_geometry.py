"""Independent native-model checks for training-only upright pose candidates."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.spatial import ConvexHull
from supermimic.acp.robots import get_robot
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.task import BodySpheres
from supermimic.acp.mdp import MDP
from supermimic.rotations import quat_to_mat,mat_to_quat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();torch.set_num_threads(2)
    robot=get_robot('tienkung3_29dof');kin=RobotKinematics('cpu',robot);spheres=BodySpheres(robot,'spheres224')
    mdp=MDP(kin,None,torch.tensor(robot.default_q));model=robot.get_spec().compile();data=mujoco.MjData(model);addr,_=robot.joint_addresses(model)
    foot_names=[robot.contact_bodies[k][0] for k in ('left_foot','right_foot')];foot_ids=[model.body(n).id for n in foot_names]
    mask=spheres.owner.numpy()<2;points=spheres.pts.numpy()[mask];radii=spheres.rad.numpy()[mask];owners=spheres.owner.numpy()[mask]
    rows=[]
    for source in json.loads(a.candidates.read_text()):
        q=np.array(source['joint_pos']);root=np.array(source['root_pose_wxyz'])
        R=quat_to_mat(torch.tensor(root[3:],dtype=torch.float64)).numpy();yaw=np.arctan2(R[1,0],R[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        R=np.array([[c,s,0],[-s,c,0],[0,0,1]])@R
        quat=mat_to_quat(torch.tensor(R,dtype=torch.float64)).numpy()
        data.qpos[:3]=[0,0,root[2]];data.qpos[3:7]=quat;data.qpos[addr]=q;mujoco.mj_forward(model,data)
        pos=data.xpos[foot_ids];rot=data.xmat[foot_ids].reshape(2,3,3)
        centers=pos[owners]+np.einsum('nij,nj->ni',rot[owners],points)
        clearance=centers[:,2]-radii;mins=np.array([clearance[owners==i].min() for i in (0,1)])
        data.qpos[2]-=float(mins.min());mujoco.mj_forward(model,data)
        # Conservative proxy using points near each sole; not a contact-force or stability certificate.
        near=clearance<=mins[owners]+.015;cloud=centers[near,:2]
        hull=ConvexHull(cloud);com=data.subtree_com[0].copy()
        margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])))
        chest_R=data.xmat[model.body(robot.torso_body).id].reshape(3,3)
        angle=lambda r:float(np.rad2deg(np.arctan2(np.linalg.norm(r[:2,2]),r[2,2])))
        zero=torch.zeros(1,3);state=(torch.tensor(data.qpos[:3],dtype=torch.float32)[None],torch.tensor(R,dtype=torch.float32)[None],zero,zero,torch.tensor(q,dtype=torch.float32)[None],torch.zeros(1,29))
        with torch.no_grad():bp,br,_,_=kin.forward(*state,all_bodies=True)
        names=foot_names+[robot.torso_body];ids=[kin.body_names.index(n) for n in names];mids=[model.body(n).id for n in names]
        error=max(float(np.max(np.abs(bp[0,ids].numpy()-data.xpos[mids]))),float(np.max(np.abs(br[0,ids].numpy()-data.xmat[mids].reshape(-1,3,3)))))
        assert error<2e-6
        limits=bool(np.all(q>=mdp.soft_lo.numpy()) and np.all(q<=mdp.soft_hi.numpy()))
        passed=bool(limits and margin>=.015 and np.ptp(mins)<=.01 and angle(chest_R)<=7 and angle(R)<=10)
        rows.append(dict(source=source,geometry_passed=passed,initial_q=q.tolist(),initial_height=float(data.qpos[2]),initial_quat=quat.tolist(),
                         chest_tilt_deg=angle(chest_R),pelvis_tilt_deg=angle(R),foot_height_gap_m=float(np.ptp(mins)),
                         approximate_support_margin_m=margin,joint_soft_limits_passed=limits,native_fk_max_error=error))
    rows.sort(key=lambda r:(not r['geometry_passed'],r['source']['score']))
    a.out.write_text(json.dumps(dict(scope=__doc__+' Not physical acceptance.',passed_candidates=sum(r['geometry_passed'] for r in rows),
        candidate_count=len(rows),grounding='Global yaw removed; rootXY zero; rootZ shifted to lowest foot-sphere surface; no joint changes',
        limitations='Sole convex hull is approximate; force feasibility and physical rollout still required',candidates=rows),indent=2))
    print(json.dumps([dict(key=r['source']['key'],frame=r['source']['frame'],passed=r['geometry_passed'],chest=r['chest_tilt_deg'],margin=r['approximate_support_margin_m'],gap=r['foot_height_gap_m']) for r in rows[:8]],indent=2))


if __name__=='__main__':main()
