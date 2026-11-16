"""Independent MuJoCo FK audit of offline distal head on fresh development targets."""
import argparse,json
from pathlib import Path
import torch,numpy as np,mujoco
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.wide_curriculum import grid,sample
from supermimic.acp.robots import sha256
from supermimic.rotations import mat_to_quat
p=argparse.ArgumentParser(description=__doc__)
for k in ('source','head','out'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2);ck=torch.load(a.source,map_location='cpu',weights_only=False);head=torch.load(a.head,map_location='cpu',weights_only=False);assert head['source_sha256']==sha256(a.source)
T=WideAimingTask('cpu',ck['config']);ids=head['joint_ids'];assert ids==T.arm_ids.tolist()[4:]
net=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,3));net.load_state_dict(head['network']);net.eval()
params=torch.cat((grid(4,'cpu'),sample(4,1024,'cpu',torch.Generator().manual_seed(2671))[0]));S=T.initial_state(len(params));ctx=T.context(S,params,4.);q=S[4].clone();rows=[]
with torch.no_grad():
 for count in range(1,6):
  h,_,R=T.hand((*S[:4],q,S[5]));unit=torch.nn.functional.normalize((R.transpose(-1,-2)@(ctx['aim_target_w']-h)[...,None]).squeeze(-1),dim=-1);q=S[4].clone();q[:,ids]=head['mid']+head['half']*torch.tanh(net(unit));error=T.metrics((*S[:4],q,S[5]),ctx)[0]
  rows.append(dict(feedforward_refinements=count,grid_within5cm=int((error[:75]<=.05).sum()),random_within5cm=float((error[75:]<=.05).float().mean()),p95_error_m=float(torch.quantile(error,.95)),max_error_m=float(error.max())))
model=T.robot.get_spec().compile();data=mujoco.MjData(model);addr,_=T.robot.joint_addresses(model);tcp=model.body('right_tcp_link').id;quat=mat_to_quat(S[1]);difference=0.;independent=[]
for i in range(len(params)):
 data.qpos[:3]=S[0][i];data.qpos[3:7]=quat[i];data.qpos[addr]=q[i];mujoco.mj_forward(model,data);h=data.xpos[tcp];d=-data.xmat[tcp].reshape(3,3)[:,2];target=ctx['aim_target_w'][i].numpy();normal=ctx['aim_normal_w'][i].numpy();den=normal@d;distance=normal@(target-h)/den if den>1e-4 else -1.;e=float(np.linalg.norm(h+distance*d-target)) if distance>0 else float('inf')
 assert np.isfinite(e)==bool(error[i].isfinite())
 if np.isfinite(e):difference=max(difference,abs(e-float(error[i])))
 independent.append(e)
assert difference<1e-4
r=dict(scope=__doc__,source_sha256=sha256(a.source),head_sha256=sha256(a.head),development_seed=2671,rows=rows,torch_mujoco_max_error_difference_m=difference,physical_acceptance=False,limitations='Fixed body pose, no dynamics or torque; refinements diagnose consistency rather than add runtime IK')
a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
