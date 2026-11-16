"""Read-only checkpoint dynamics diagnostics on recorded independent episodes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.mdp import unflatten_state,normalize_xi
from supermimic.acp.aiming_fullbody import flat_boxes
from supermimic.acp.robots import sha256
from supermimic.rotations import yaw_mat

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','recorded','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False);task=WideAimingTask(a.device,ck['config'])
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(a.device);wm.load_state_dict(ck['wm']);freeze_normalizers(wm);wm.eval()
    raw=np.load(a.recorded);data={k:torch.as_tensor(raw[k].copy(),device=a.device) for k in raw.files};reports={}
    for horizon in [1,8,16,32]:
        starts=torch.arange(0,len(data['a'])-horizon+1,horizon,device=a.device)
        S=unflatten_state(data['s'][starts].flatten(0,1));alive=torch.ones(len(S[0]),dtype=torch.bool,device=a.device)
        for j in range(horizon):
            S=wm.step(S,data['a'][starts+j].flatten(0,1),normalize_xi(data['xi'][starts+j].flatten(0,1)),task.mdp.scan(S,flat_boxes(S)))
            if j<horizon-1:alive &= ~data['done'][starts+j].flatten()
        truth=unflatten_state(data['s1'][starts+horizon-1].flatten(0,1))
        predicted_fk=task.kin.forward(*S);true_fk=task.kin.forward(*truth)
        vel=predicted_fk[2][:,task.feet][alive];true=true_fk[2][:,task.feet][alive]
        error=vel-true
        heading_error=(yaw_mat(truth[1][alive]).transpose(-1,-2)[:,None] @ error[...,None]).squeeze(-1)
        reports[str(horizon)]={'windows':int(alive.sum()),'foot_velocity_error_norm_p95_m_s':float(error.norm(dim=-1).quantile(.95)),
            'foot_velocity_error_component_rmse_m_s':float(error.square().mean().sqrt()),'signed_foot_velocity_error_xyz_m_s':error.mean((0,1)).tolist(),
            'predicted_foot_speed_median_m_s':float(vel.norm(dim=-1).median()),'actual_foot_speed_median_m_s':float(true.norm(dim=-1).median()),
            'predicted_quiet_reward_mean':float((1/(1+vel.square().sum(-1)/.05**2)).mean()),
            'actual_quiet_reward_mean':float((1/(1+true.square().sum(-1)/.05**2)).mean())}
        reports[str(horizon)]['signed_velocity_error_by_foot_heading_xyz_m_s']=heading_error.mean(0).tolist()
        reports[str(horizon)]['feet_position_error_p95_m']=float((predicted_fk[0][:,task.feet]-true_fk[0][:,task.feet])[alive].norm(dim=-1).amax(-1).quantile(.95))
        reports[str(horizon)]['root_position_error_p95_m']=float((S[0]-truth[0])[alive].norm(dim=-1).quantile(.95))
    result=dict(scope=__doc__,checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),recorded=str(a.recorded),recorded_sha256=sha256(a.recorded),
        collection_duration_s=len(data['a'])*task.dt,
        observation_limit='Prediction on supplied recorded transitions only. Resets are excluded from windows; collection identity and policy identity must be checked separately.',
        policy_acceptance=False,weights_changed=False,horizons=reports)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
