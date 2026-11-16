"""Fit affine distal feedforward to an independently checked offline IK grid."""
import argparse,json
from pathlib import Path
import numpy as np,torch,mujoco
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256
p=argparse.ArgumentParser(description=__doc__)
for k in ('source','geometry','trajectory','out'):p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--gains',type=float,nargs='+',default=[3.,5.])
a=p.parse_args();assert all(0<g<=9 for g in a.gains);torch.set_num_threads(2);ck=torch.load(a.source,map_location='cpu',weights_only=False);digest=sha256(a.source);g=json.loads(a.geometry.read_text());assert g['trajectory_sha256']==sha256(a.trajectory)
if g['source_sha256']==digest:
 c=g['cases'][0];assert c['proximal_max_change_bound_rad']==0
else:
 assert ck['calibration']['geometry_sha256']==sha256(a.geometry)
 assert ck['source_checkpoint']['sha256']==g['source_sha256']
 proximal=np.array(ck['calibration']['shared_arm_pose_rad'][:4])
 c=next(case for case in g['cases'] if np.allclose(case['shared_proximal_q_rad'],proximal,atol=1e-6,rtol=0))
assert c['within5cm']==75
T=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,T);ids=T.arm_ids.tolist()[4:];model=T.robot.get_spec().compile();data=mujoco.MjData(model);addr,dof=T.robot.joint_addresses(model);tcp=model.body('right_tcp_link').id;torso=model.body(T.robot.torso_body).id
with np.load(a.trajectory) as z:
 assert str(z['checkpoint_sha256'])==g['source_sha256']
 data.qpos[:3]=z['root'][750,37];data.qpos[3:7]=z['root_quat'][750,37]
features=[];outputs=[]
for q in c['q_rad']:
 data.qpos[addr]=q;mujoco.mj_forward(model,data)
 direction=data.xmat[torso].reshape(3,3).T@(-data.xmat[tcp].reshape(3,3)[:,2]);features.append([*direction,1.])
 target=np.array(q)+data.qfrc_bias[dof]/np.array(T.robot.kp);outputs.append(((torch.tensor(target)-T.mdp.default_q)/T.alpha)[ids].numpy())
X=torch.tensor(np.array(features),dtype=torch.float64);Y=torch.tensor(np.array(outputs),dtype=torch.float64);coef=torch.linalg.lstsq(X,Y).solution.float();rms=float(((X@coef.double()-Y)*T.alpha[ids]).square().mean().sqrt())

from itertools import product
results=[]
generator=torch.Generator().manual_seed(315);perm=torch.randperm(75,generator=generator);train,test=perm[:60],perm[60:]
for degree in (1,2,3):
 powers=[v for v in product(range(degree+1),repeat=3) if sum(v)<=degree and v[2]<2]
 F=torch.stack([torch.prod(X[:,:3]**torch.tensor(power),dim=1) for power in powers],dim=1)
 coeff=torch.linalg.lstsq(F[train],Y[train]).solution
 residual=(F@coeff-Y)*T.alpha[ids]
 results.append(dict(degree=degree,feature_count=len(powers),condition_number=float(torch.linalg.cond(F[train])),train_rms_rad=float(residual[train].square().mean().sqrt()),heldout_rms_rad=float(residual[test].square().mean().sqrt()),heldout_max_abs_rad=float(residual[test].abs().max())))
a.out.write_text(json.dumps(dict(scope='Offline regression diagnostic only; 60train15heldout IK solutions, not physical acceptance',source_sha256=digest,results=results),indent=2));print(json.dumps(results))
