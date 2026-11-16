"""Calibrate inherited support feedback while preserving the independently tested aiming head."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from tools.prepare_support_gain_calibration import scaled_feedback_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','balance-source','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gains',type=float,nargs='+',default=[.8,1.2])
    a=p.parse_args();torch.set_num_threads(2)
    assert all(np.isfinite(g) and .5<=g<=1.5 for g in a.gains)
    ck=torch.load(a.source,map_location='cpu',weights_only=False);digest=sha256(a.source)
    balance=torch.load(a.balance_source,map_location='cpu',weights_only=False)
    assert not ck.get('smoke') and ck['stage'] in ('training','calibration')
    assert not balance.get('smoke') and balance['config']['curriculum_level']==0
    assert ck['config']['curriculum_level'] in range(1,7)
    assert ck['config']['support_command_mode']=='error' and ck['config']['support_capture_steps']==150
    assert not torch.count_nonzero(ck['pi']['goal_norm.mean'][:6])
    assert torch.equal(ck['pi']['enc.0.weight'][:,:6],balance['pi']['enc.0.weight'][:,:6])
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    original=FullbodyPolicy(task.robot);original.load_state_dict(ck['pi'])
    torch.manual_seed(233);history=torch.randn(16,10,99);command=torch.randn(16,107);command[:,:6]=0
    with torch.no_grad():baseline=original(history,command)
    a.out.mkdir(parents=True,exist_ok=True)
    for number,gain in enumerate(a.gains,1):
        out=a.out/f'ckpt_{number:06d}.pt'
        if out.exists():raise FileExistsError(out)
        pi=scaled_feedback_state(ck['pi'],gain)
        for key in pi:
            current,old=(pi[key][:,6:],ck['pi'][key][:,6:]) if key=='enc.0.weight' else (pi[key],ck['pi'][key])
            assert torch.equal(current,old),key
        policy=FullbodyPolicy(task.robot);policy.load_state_dict(pi)
        with torch.no_grad():torch.testing.assert_close(policy(history,command),baseline,atol=0,rtol=0)
        cfg=dict(ck['config'],initialization_mode='aiming_support_gain_calibration',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100)
        calibration=dict(method=__doc__,relative_support_gain=gain,balance_source=str(a.balance_source.resolve()),balance_source_sha256=sha256(a.balance_source),
            inherited_support_weights_verified_exact=True,changed_tensor='enc.0.weight[:,:6]',all_other_weights_exact=True,
            normalizers_exact=True,zero_support_error_actions_exact=True,initial_pose_unchanged=True,
            support_capture_and_release_unchanged=True,physical_acceptance=False,acceptance_criteria_changed=False)
        artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),
            robot_metadata=ck['robot_metadata'],pi=pi,wm=ck['wm'],provenance=ck.get('provenance'),
            source_checkpoint=dict(path=str(a.source.resolve()),sha256=digest),calibration=calibration)
        validate_checkpoint(artifact,WideAimingTask('cpu',cfg));save_checkpoint(out,artifact)
        json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration))
        print(json.dumps(dict(output=str(out),relative_support_gain=gain)),flush=True)
    assert sha256(a.source)==digest


if __name__=='__main__':main()
