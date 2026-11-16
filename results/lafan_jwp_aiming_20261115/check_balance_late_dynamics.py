"""Read-only thirty-second collection for late-episode dynamics diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint,flat_boxes
from supermimic.acp.aiming_fullbody_validation import validate_world_model
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.wide_curriculum import sample
from supermimic.acp.mdp import unflatten_state,normalize_xi
from supermimic.acp.robots import sha256

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(919);torch.backends.cuda.matmul.allow_tf32=False
 digest=sha256(a.checkpoint);ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False);task=WideAimingTask(a.device,ck['config']);validate_checkpoint(ck,task)
 pi=FullbodyPolicy(task.robot).to(a.device);pi.load_state_dict(ck['pi']);pi.eval();wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(a.device);wm.load_state_dict(ck['wm']);wm.eval();freeze_normalizers(pi,wm)
 cfg=dict(ck['config'],horizon=32,validation_horizons=[1,8,16,32],wm_validation_envs=8,wm_validation_steps=1500,body_noise_rad=0.,arm_noise_rad=0.)
 report=validate_world_model(pi,wm,task,lambda n:sample(0,n,a.device),cfg,a.out)
 data={k:torch.as_tensor(v.copy(),device=a.device) for k,v in np.load(a.out.with_suffix('.npz')).items()};segments={};horizon=32
 starts=torch.arange(0,len(data['a'])-horizon+1,horizon,device=a.device);S=unflatten_state(data['s'][starts].flatten(0,1));alive=torch.ones(len(S[0]),dtype=torch.bool,device=a.device)
 for j in range(horizon):
  S=wm.step(S,data['a'][starts+j].flatten(0,1),normalize_xi(data['xi'][starts+j].flatten(0,1)),task.mdp.scan(S,flat_boxes(S)))
  if j<horizon-1:alive &= ~data['done'][starts+j].flatten()
 truth=unflatten_state(data['s1'][starts+horizon-1].flatten(0,1));pred=task.kin.forward(*S);actual=task.kin.forward(*truth);age=data['ep_step'][starts].flatten()*task.dt
 for lo,hi in [(0,3),(3,10),(10,20),(20,30)]:
  keep=alive & (age>=lo) & (age<hi)
  if not keep.any():segments[f'{lo}-{hi}']={'windows':0};continue
  pe=(pred[0][:,task.feet]-actual[0][:,task.feet])[keep];ve=(pred[2][:,task.feet]-actual[2][:,task.feet])[keep]
  segments[f'{lo}-{hi}']=dict(windows=int(keep.sum()),feet_position_error_p95_m=float(pe.norm(dim=-1).amax(-1).quantile(.95)),
    foot_velocity_error_p95_m_s=float(ve.norm(dim=-1).quantile(.95)),signed_foot_velocity_error_xyz_m_s=ve.mean((0,1)).tolist(),
    actual_foot_speed_median_m_s=float(actual[2][keep][:,task.feet].norm(dim=-1).median()),predicted_foot_speed_median_m_s=float(pred[2][keep][:,task.feet].norm(dim=-1).median()))
 assert sha256(a.checkpoint)==digest
 report.update(checkpoint_sha256=digest,scope=__doc__,weights_changed=False,policy_acceptance=False,seed=919,collection_steps=1500,
  warning='Physical termination resets collection environments; per-window episode age filters exclude reset crossings. This is model prediction diagnostics, not a no-reset30s policy acceptance.',late_h32=segments)
 a.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checkpoint_sha256':digest,'passed':report['passed'],'late_h32':segments}),flush=True)
if __name__=='__main__':main()
