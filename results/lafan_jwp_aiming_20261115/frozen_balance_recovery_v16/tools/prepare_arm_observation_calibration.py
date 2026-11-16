"""Fold fixed arm proprioception into decoder weights, retaining measured body feedback."""
import argparse,json
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256


def folded_observation_state(policy,columns,values):
    columns=list(columns);obs_dim=policy.obs_norm.mean.numel()
    values=torch.as_tensor(values,dtype=policy.dec[0].weight.dtype,device=policy.dec[0].weight.device)
    if not columns or len(set(columns))!=len(columns) or min(columns)<0 or max(columns)>=obs_dim:
        raise ValueError('Expected unique in-range observation columns')
    if values.shape!=(len(columns),) or not values.isfinite().all():raise ValueError('Expected matching finite raw observation values')
    latent=policy.enc[-1].out_features
    assert policy.dec[0].in_features==latent+policy.hist*obs_dim
    flat_columns=[latent+t*obs_dim+c for t in range(policy.hist) for c in columns]
    normalized=(values-policy.obs_norm.mean[columns])/policy.obs_norm.std()[columns]
    state={k:v.detach().clone() for k,v in policy.state_dict().items()}
    state['dec.0.bias']+=state['dec.0.weight'][:,flat_columns]@normalized.repeat(policy.hist)
    state['dec.0.weight'][:,flat_columns]=0
    return state,flat_columns


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--constants',type=Path,required=True);p.add_argument('--joints',choices=('right_arm','distal'),default='right_arm')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    digest=sha256(a.source);ck=torch.load(a.source,map_location='cpu',weights_only=False)
    if ck.get('smoke') or ck.get('stage') not in ('calibration','training'):raise ValueError('Formal source required')
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    constants=json.loads(a.constants.read_text());task.robot.validate_metadata(constants['robot_metadata'])
    values=torch.tensor(constants['raw_observation_values'],dtype=torch.float32)
    assert values.shape==(99,) and values.isfinite().all()
    joints=task.arm_ids.tolist() if a.joints=='right_arm' else [task.robot.joint_names.index(n) for n in ('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')]
    # MDP.obs: q(29),qd(29),angular_velocity(3),gravity(3),previous_action(29),attitude(6).
    columns=joints+[29+i for i in joints]+[64+i for i in joints]
    policy=FullbodyPolicy(task.robot);policy.load_state_dict(ck['pi']);policy.eval()
    state,flat_columns=folded_observation_state(policy,columns,values[columns])
    candidate=FullbodyPolicy(task.robot);candidate.load_state_dict(state);candidate.eval()
    torch.manual_seed(882)
    history=policy.obs_norm.mean+torch.randn(8,10,99)*policy.obs_norm.std();command=torch.randn(8,107);command[:,:6]*=.02
    masked=history.clone();masked[:,:,columns]=values[columns]
    with torch.no_grad():
        expected=policy(masked,command);actual=candidate(history,command)
        error=float(((expected-actual)*task.alpha).abs().max())
        assert error<1e-5,('Folded proprioception differs from explicit substitution',error)
        altered=history.clone();altered[:,:,columns]+=torch.randn_like(altered[:,:,columns])*10
        torch.testing.assert_close(candidate(altered,command),actual,atol=0,rtol=0)
    for key,value in ck['pi'].items():
        if key not in ('dec.0.weight','dec.0.bias'):assert torch.equal(value,state[key]),key
    other=[i for i in range(policy.dec[0].in_features) if i not in flat_columns]
    assert torch.equal(ck['pi']['dec.0.weight'][:,other],state['dec.0.weight'][:,other])
    cfg=dict(ck['config'],initialization_mode='arm_proprioception_calibration',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
    calibration=dict(method='Fixed measured arm observation folded into first decoder affine layer',joint_names=[task.robot.joint_names[i] for i in joints],
        raw_observation_columns=columns,raw_observation_values=values[columns].tolist(),decoder_columns=flat_columns,
        constants=str(a.constants.resolve()),constants_sha256=sha256(a.constants),constant_source=constants['scope'],
        explicit_substitution_max_error_rad=error,selected_observation_invariance_exact=True,normalizers_unchanged=True,
        measured_body_observations_retained=True,extra_runtime_controller=False,physical_acceptance=False)
    artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),robot_metadata=ck['robot_metadata'],
        pi=state,wm=ck['wm'],source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),provenance=ck.get('provenance'),calibration=calibration)
    assert sha256(a.source)==digest;save_checkpoint(a.out,artifact)
    json_save(a.out.with_suffix('.json'),dict(checkpoint_sha256=sha256(a.out),source_sha256=digest,calibration=calibration))
    print(json.dumps(dict(output=str(a.out),joints=a.joints,fold_max_error_rad=error,physical_acceptance=False)))


if __name__=='__main__':main()
