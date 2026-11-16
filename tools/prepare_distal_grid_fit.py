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
# Retain the independently audited local feedback; replace only its feedforward term.
oldgain=ck['calibration']['feedback_gain'];Jinv=torch.tensor(ck['calibration']['pseudoinverse']);W=Jinv/T.alpha[ids,None]
a.out.mkdir(parents=True,exist_ok=True)
for n,gain in enumerate(a.gains,1):
 out=a.out/f'ckpt_{n:06d}.pt';assert not out.exists();pi={k:v.clone() for k,v in ck['pi'].items()}
 pi['distal_command_skip.weight'].zero_();pi['distal_command_skip.weight'][:,:3]=coef[:3].T+gain*W;pi['distal_command_skip.weight'][:,3:6]=-gain*W;pi['distal_command_skip.bias']=coef[3]-pi['dec.8.bias'][ids]
 for key in pi:
  if not key.startswith('distal_command_skip.'):assert torch.equal(pi[key],ck['pi'][key])
 calibration=dict(method=__doc__,geometry_sha256=sha256(a.geometry),feedback_gain=gain,affine_fit_rms_rad=rms,body_and_reset_exact=True,physical_acceptance=False,online_ik=False)
 artifact=dict(ck,stage='calibration',pi=pi,source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=calibration)
 validate_checkpoint(artifact,T);save_checkpoint(out,artifact);json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration));print(str(out),rms,flush=True)
