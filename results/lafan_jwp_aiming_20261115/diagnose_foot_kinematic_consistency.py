"""Compare FK foot velocities with observed control-interval displacements."""
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.mdp import unflatten_state,normalize_xi
from supermimic.acp.aiming_fullbody import flat_boxes
from supermimic.acp.robots import sha256

@torch.no_grad()
def main():
 torch.set_num_threads(2)
 b=Path('results/lafan_jwp_aiming_20261115');run=Path('logs/tienkung3_29dof/lafan_jwp_20261115/balance_recovery_conservative/train')
 ckpath=run/'ckpt_002500.pt';recorded=run/'wm_validation/002500.npz';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
 task=WideAimingTask('cpu',ck['config']);wm=WorldModel(task.mdp.scan_dim,robot=task.robot);wm.load_state_dict(ck['wm']);freeze_normalizers(wm)
 d={k:torch.from_numpy(v.copy()) for k,v in np.load(recorded).items()};S=unflatten_state(d['s'].flatten(0,1));T=unflatten_state(d['s1'].flatten(0,1))
 P=wm.step(S,d['a'].flatten(0,1),normalize_xi(d['xi'].flatten(0,1)),task.mdp.scan(S,flat_boxes(S)))
 f0=task.kin.forward(*S);f1=task.kin.forward(*T);fp=task.kin.forward(*P);ids=task.feet
 rows={}
 for label,mask in [('all',torch.ones(len(S[0]),dtype=torch.bool)),('after1s',d['ep_step'].flatten()>=50)]:
  groups={}
  for name,f in [('physical',f1),('model',fp)]:
   displacement_velocity=(f[0][:,ids]-f0[0][:,ids])/task.dt
   mean_velocity=(f[2][:,ids]+f0[2][:,ids])/2
   error=(displacement_velocity-mean_velocity)[mask]
   groups[name]=dict(samples=int(mask.sum()),median_displacement_speed_m_s=float(displacement_velocity[mask].norm(dim=-1).median()),
    median_fk_midpoint_speed_m_s=float(mean_velocity[mask].norm(dim=-1).median()),
    consistency_error_norm_p95_m_s=float(error.norm(dim=-1).quantile(.95)),
    signed_consistency_error_xyz_m_s=error.mean((0,1)).tolist())
  rows[label]=groups
 result=dict(scope=__doc__,checkpoint_sha256=sha256(ckpath),recorded_sha256=sha256(recorded),rows=rows,
  caveat='Midpoint velocity is an approximation across 20ms with contact changes. This is not an exact integration identity, causal attribution, or acceptance.',weights_changed=False)
 (b/'balance_conservative2500_foot_kinematic_consistency.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
