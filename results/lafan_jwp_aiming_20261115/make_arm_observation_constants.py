"""Raw observation constants from the exact accepted balance-stage policy."""
from pathlib import Path
import json
import numpy as np
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.mdp import unflatten_state
from supermimic.acp.robots import sha256


def main():
    torch.set_num_threads(2);b=Path(__file__).resolve().parent
    source=b/'stance_bias_candidates/inward_0.25.pt';ck=torch.load(source,map_location='cpu',weights_only=False)
    task=WideAimingTask('cpu',ck['config']);metadata=task.robot.metadata()
    data_path=b/'balance_source_arm_observations.npz';r=json.loads(data_path.with_suffix('.json').read_text())
    assert r['checkpoint_sha256']==sha256(source) and 'late_h32' in r
    with np.load(data_path) as z:
        steps=torch.tensor(z['ep_step'][1:]);mask=(steps>=250)&(steps<1250)&~torch.tensor(z['done'][:-1])
        states=torch.tensor(z['s'][1:])[mask];previous=torch.tensor(z['a'][:-1])[mask];xi=torch.tensor(z['xi'][1:])[mask]
    observations=task.mdp.obs(unflatten_state(states),previous,xi)
    assert len(observations)>7000 and observations.isfinite().all()
    S=task.initial_state(1);initial_xi=torch.zeros(1,task.robot.xi_dim);initial_xi[:,0]=1
    initial_action=(S[4]-task.mdp.default_q)/task.alpha
    initial=task.mdp.obs(S,initial_action,initial_xi)[0]
    arm_source=b/'observed_arm_center_candidates/ckpt_000001.pt';a=torch.load(arm_source,map_location='cpu',weights_only=False)
    q=S[4].clone();q[:,task.arm_ids]=torch.tensor(a['calibration']['arm_pose_rad'])
    ap=initial_action.clone();ap[:,task.arm_ids]=(torch.tensor(a['calibration']['arm_pd_targets_rad'])-task.mdp.default_q[task.arm_ids])/task.alpha[task.arm_ids]
    observed=task.mdp.obs((*S[:4],q,S[5]),ap,initial_xi)[0]
    values={'initial':initial,'steady':observations.median(0).values,'observed_arm':observed}
    out=b/'arm_observation_constants';out.mkdir(exist_ok=True)
    for name,value in values.items():
        path=out/f'{name}.json'
        if path.exists():raise FileExistsError(path)
        record=dict(scope='Fixed raw proprioception for calibration; initial state or observed5–25s median of trained balance policy, or previously observed constant arm pose',
            mode=name,robot_metadata=metadata,observation_dim=99,raw_observation_values=value.tolist(),
            balance_checkpoint=str(source),balance_checkpoint_sha256=sha256(source),
            data_path=str(data_path),data_sha256=sha256(data_path),measured_samples=len(observations),measured_episode_age_range_s=[5,25],
            actual_arm_constant_checkpoint_sha256=sha256(arm_source) if name=='observed_arm' else None,
            runtime_data_dependency=False,physical_policy_acceptance=False)
        path.write_text(json.dumps(record,indent=2));print(path,flush=True)


if __name__=='__main__':main()
