"""Fold an audited native-joint stance correction into a learned policy output bias."""
import argparse
import json
import math
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--direction',type=Path,required=True)
    p.add_argument('--scale',type=float,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    if not math.isfinite(a.scale) or not 0<abs(a.scale)<=2:raise ValueError('Scale must be finite, nonzero and <=2')
    digest=sha256(a.source);ck=torch.load(a.source,map_location='cpu',weights_only=False)
    if ck.get('smoke') or ck.get('stage') not in ('training','calibration') or ck['config'].get('curriculum_level')!=0:
        raise ValueError('Requires a formal balance-stage source')
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    direction=json.loads(a.direction.read_text());task.robot.validate_metadata(direction['robot_metadata'])
    if sha256(direction['source_trajectory'])!=direction['source_sha256']:raise ValueError('Geometry source changed')
    if direction['joint_order']!=list(task.robot.joint_names):raise ValueError('Joint order mismatch')
    if any(f['linearization_error_m']>.0003 or f['orientation_change_rad']>.001 for f in direction['feet']):
        raise ValueError('Direction diagnostic exceeds kinematic tolerance')
    dq=torch.tensor(direction['native29joint_target_correction_rad'],dtype=torch.float32)*a.scale
    leg_ids=[i for i,n in enumerate(task.robot.joint_names) if any(n.startswith(s) for s in ('hip_','knee_','ankle_'))]
    assert len(leg_ids)==12
    other_ids=[i for i in range(29) if i not in leg_ids]
    if torch.count_nonzero(dq[other_ids]) or not dq.isfinite().all() or dq.abs().max()>.1:
        raise ValueError('Unexpected non-leg or excessive correction')
    pi={k:v.detach().clone() for k,v in ck['pi'].items()}
    pi['dec.8.bias']+=dq/task.alpha
    source=FullbodyPolicy(task.robot);source.load_state_dict(ck['pi']);candidate=FullbodyPolicy(task.robot);candidate.load_state_dict(pi)
    torch.manual_seed(739);history=torch.randn(8,10,99);command=torch.randn(8,107)
    with torch.no_grad():
        difference=(candidate(history,command)-source(history,command))*task.alpha
        torch.testing.assert_close(difference,dq.expand_as(difference),atol=2e-6,rtol=2e-5)
    for k,v in ck['pi'].items():
        if k!='dec.8.bias':assert torch.equal(v,pi[k]),k
    assert torch.equal(ck['pi']['dec.8.bias'][other_ids],pi['dec.8.bias'][other_ids])
    cfg=dict(ck['config'],initialization_mode='stance_bias_calibration',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
    calibration=dict(method='fixed native-joint PD correction folded into learned decoder bias',
        scale=a.scale,nominal_inward_distance_m=.01*a.scale,direction=str(a.direction.resolve()),direction_sha256=sha256(a.direction),
        native29_correction_rad=dq.tolist(),changed_tensor='dec.8.bias: 12 leg joints only',
        all_other_policy_tensors_exact=True,normalizers_exact=True,extra_inference_controller=False,
        raw_action_function_check_max_error_rad=float((difference-dq).abs().max()),
        caveat='Soft joint clipping still applies. Actual physical displacement and stability require full simulation.',
        physical_acceptance=False)
    artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),
        robot_metadata=ck['robot_metadata'],pi=pi,wm=ck['wm'],source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),
        provenance=ck.get('provenance'),calibration=calibration)
    assert sha256(a.source)==digest
    save_checkpoint(a.out,artifact)
    json_save(a.out.with_suffix('.json'),dict(checkpoint_sha256=sha256(a.out),source_sha256=digest,calibration=calibration))
    print(json.dumps(dict(output=str(a.out),scale=a.scale,max_correction_rad=float(dq.abs().max()),
                          function_error_rad=calibration['raw_action_function_check_max_error_rad'])))


if __name__=='__main__':main()
