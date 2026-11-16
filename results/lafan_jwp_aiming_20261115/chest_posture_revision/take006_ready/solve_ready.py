import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,hashlib
from pathlib import Path
import numpy as np,torch,mujoco
from scipy.optimize import minimize,linprog
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from supermimic.acp.robots import get_robot
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP
P=Path(__file__).resolve().parent
r=get_robot('tienkung3_29dof');m=r.get_spec().compile();d=mujoco.MjData(m);addr,dof=r.joint_addresses(m);mdp=MDP(RobotKinematics('cpu',r),None,torch.tensor(r.default_q));lo=mdp.soft_lo.numpy();hi=mdp.soft_hi.numpy()
c=json.load(open(P.parent/'jwp_random_candidates/pose_candidates_ranked.json'))[0];assert c['key']=='jwp/take006_chr00' and c['frame']==452
from tools.render_upright_pose_gallery import normalize
z=np.load(c['motion']);initial=normalize(z['joint_pos'][452:453],z['joint_pos'][452])[0];q0=initial[7:].copy();root=initial[:7].copy();feet=[m.body(f'ankle_roll_{s}_link').id for s in ['l','r']]
def fk(x):
 d.qpos[:7]=root;d.qpos[2]=x[12];d.qpos[addr]=q0;d.qpos[addr[:12]]=x[:12];d.qvel[:]=0;mujoco.mj_forward(m,d)
x0=np.r_[q0[:12],root[2]];fk(x0)
# Exact native collision mesh sole vertices transformed into foot body frames.
sole=[]
for b in feet:
 vv=[]
 for g in range(m.ngeom):
  if m.geom_bodyid[g]!=b or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
  mesh=m.geom_dataid[g];a=m.mesh_vertadr[mesh];n=m.mesh_vertnum[mesh];v=m.mesh_vert[a:a+n].astype(float);R=Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix();vv.append(v@R.T+m.geom_pos[g])
 v=np.concatenate(vv);h=ConvexHull(v);areas=np.linalg.norm(np.cross(v[h.simplices[:,1]]-v[h.simplices[:,0]],v[h.simplices[:,2]]-v[h.simplices[:,0]]),axis=1)/2;eligible=np.where(h.equations[:,2]<-.8)[0];face=eligible[np.argmax(areas[eligible])];plane=h.equations[face];v=v[abs(v@plane[:3]+plane[3])<2e-5];sole.append(v)
targets=[];yaw=[]
for b,v in zip(feet,sole):
 R=d.xmat[b].reshape(3,3).copy();yaw.append(np.arctan2(R[1,0],R[0,0]));targets.append(d.xpos[b,:2].copy())
def cons(x):
 fk(x);out=[]
 for k,(b,v) in enumerate(zip(feet,sole)):
  R=d.xmat[b].reshape(3,3);world=v@R.T+d.xpos[b];out.extend((d.xpos[b,:2]-targets[k]).tolist());out.append(world[:,2].mean());normal=np.linalg.svd(v-v.mean(0))[2][-1];normal*=np.sign(normal[2]);nw=R@normal;out.extend(nw[:2]);out.append(np.arctan2(R[1,0],R[0,0])-yaw[k])
 return np.array(out)
def obj(x):return np.sum((x[:12]-x0[:12])**2)+20*(x[12]-x0[12])**2
bounds=list(zip(lo[:12]+1e-5,hi[:12]-1e-5))+[(root[2]-.15,root[2]+.1)]
start=np.clip(x0,[a for a,b in bounds],[b for a,b in bounds]);res=minimize(obj,start,method='SLSQP',bounds=bounds,constraints=[dict(type='eq',fun=cons)],options=dict(maxiter=1500,ftol=1e-12));fk(res.x);q=d.qpos[addr].copy();fixed=d.qpos[:7].copy();pose=np.r_[fixed,q]
# Static contact wrench LP: native sole convex-hull vertices, friction pyramid mu=.7, effort and soft-limit-compatible gravity-compensated PD targets.
points=[];bodies=[]
for b,v in zip(feet,sole):
 for i in ConvexHull(v[:,:2]).vertices:points.append(v[i]@d.xmat[b].reshape(3,3).T+d.xpos[b]);bodies.append(b)
points=np.array(points);N=len(points);J=[]
for point,b in zip(points,bodies):
 jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,point,b);J.append(jp)
A=np.zeros((m.nv,N*3+29))
for i,j in enumerate(J):A[:,i*3:i*3+3]=j.T
for i,j in enumerate(dof):A[j,N*3+i]=1
ub=[]
for i in range(N):
 for axis in [0,1]:
  for sign in [-1,1]:
   row=np.zeros(N*3+29);row[3*i+axis]=sign;row[3*i+2]=-.7;ub.append(row)
eff=np.array(r.effort);kp=np.array(r.kp);taulo=np.maximum(-eff,(lo-q)*kp);tauhi=np.minimum(eff,(hi-q)*kp)
lp=linprog(np.r_[np.tile([0,0,1.],N),np.zeros(29)],A_ub=ub,b_ub=np.zeros(len(ub)),A_eq=A,b_eq=d.qfrc_bias,bounds=[(-10000,10000),(-10000,10000),(0,10000)]*N+list(zip(taulo,tauhi)),method='highs')
def metrics(row):
 fk(np.r_[row[7:19],row[2]]);cloud=[];heights=[]
 for b,v in zip(feet,sole):
  w=v@d.xmat[b].reshape(3,3).T+d.xpos[b];cloud.extend(w[:,:2]);heights.append([float(w[:,2].min()),float(w[:,2].max())])
 hull=ConvexHull(cloud);com=d.subtree_com[0].copy();margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])));tilt=lambda R:float(np.degrees(np.arctan2(np.linalg.norm(R[:2,2]),R[2,2])))
 return dict(foot_sole_z_ranges_m=heights,com_xyz_m=com.tolist(),support_margin_m=margin,chest_deg=tilt(d.xmat[m.body(r.torso_body).id].reshape(3,3)),pelvis_deg=tilt(d.xmat[m.body('pelvis').id].reshape(3,3)),all_soft_limits=bool(np.all(row[7:]>=lo)&np.all(row[7:]<=hi)))
original_metrics=metrics(initial);repaired_metrics=metrics(pose);fk(res.x)
report=dict(selection=dict(user_selected='JWP new gallery 01',source=c,source_motion_sha256=hashlib.sha256(Path(c['motion']).read_bytes()).hexdigest(),frame=452),method='SLSQP minimum leg-joint squared change plus20*rootZ squared change; exact native sole plane, fixed original footXY/yaw; arms waist and root orientation fixed',solver_success=bool(res.success),solver_message=res.message,equality_max_error=float(abs(cons(res.x)).max()),initial_pose=initial.tolist(),repaired_pose=pose.tolist(),joint_changes_deg={n:float(np.degrees(q[i]-q0[i])) for i,n in enumerate(r.joint_names)},root_height_change_m=float(fixed[2]-root[2]),original=original_metrics,repaired=repaired_metrics,static_lp_success=bool(lp.success),static_lp_message=lp.message,friction_coefficient=.7,physical_acceptance=False,training_started=False)
if lp.success:
 tau=lp.x[N*3:];report['static']=dict(torques_nm=tau.tolist(),max_effort_fraction=float(np.max(abs(tau)/eff)),pd_targets_rad=(q+tau/kp).tolist(),pd_targets_all_soft_limits=bool(np.all(q+tau/kp>=lo-1e-7)&np.all(q+tau/kp<=hi+1e-7)),force_balance_max_residual=float(np.max(abs(A@lp.x-d.qfrc_bias))),contact_points=points.tolist(),forces=lp.x[:N*3].reshape(N,3).tolist())
np.savez(P/'ready_pose.npz',initial_pose=initial,repaired_pose=pose,joint_names=r.joint_names);(P/'repair.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['selection','initial_pose','repaired_pose','joint_changes_deg','static']},indent=2))
