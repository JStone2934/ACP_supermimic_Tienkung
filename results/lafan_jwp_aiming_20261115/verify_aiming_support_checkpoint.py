"""Verify support-only updates, frozen aiming outputs and two-rank integrity."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    source=Path(ck['source_checkpoint']['path']);digest=sha256(source)
    assert digest==ck['source_checkpoint']['sha256']
    initial=torch.load(source,map_location='cpu',weights_only=False)
    assert bool(ck['smoke'])==a.smoke and not initial.get('smoke')
    assert ck['config']['policy_training_scope']=='aiming_support_error_inputs'
    assert ck['config']['curriculum_level'] in range(1,7)
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    for key,value in initial['pi'].items():
        old,new=(value[:,6:],ck['pi'][key][:,6:]) if key=='enc.0.weight' else (value,ck['pi'][key])
        assert torch.equal(old,new),key
    delta=float((ck['pi']['enc.0.weight'][:,:6]-initial['pi']['enc.0.weight'][:,:6]).norm())
    d=ck['distributed'];assert d['world_size']==2 and not d['skipped'] and d['state_sha256']
    assert {rank['device'] for rank in d['ranks']}=={'cuda:0','cuda:1'}
    assert bool(delta>0)==bool(d['updates'].get('pi',0))
    finite={name:all(bool(v.isfinite().all()) for v in ck[name].values() if torch.is_tensor(v)) for name in ('pi','wm','critic')}
    assert all(finite.values())
    if a.smoke:assert ck['iteration']==2 and all(d['updates'][name]==2 for name in ('wm','pi','critic'))
    original=FullbodyPolicy(task.robot);original.load_state_dict(initial['pi'])
    candidate=FullbodyPolicy(task.robot);candidate.load_state_dict(ck['pi'])
    torch.manual_seed(271);history=torch.randn(8,10,99);command=torch.randn(8,107)
    ids=[task.robot.joint_names.index(name) for name in ('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')]
    assert torch.count_nonzero(ck['pi']['dec.8.weight'][ids])==0
    with torch.no_grad():
        torch.testing.assert_close(candidate(history,command)[:,ids],original(history,command)[:,ids],atol=0,rtol=0)
        command[:,:6]=0
        torch.testing.assert_close(candidate(history,command),original(history,command),atol=0,rtol=0)
    report=dict(passed=True,scope=__doc__+' Not physical policy acceptance.',checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=sha256(a.checkpoint),source_sha256=digest,smoke=a.smoke,
        formal_initialization_allowed=not a.smoke,iteration=ck['iteration'],updates=d['updates'],
        frozen_aiming_and_zero_support_error_actions_exact=True,normalizers_exact=True,
        only_six_support_input_columns_changed=True,support_weight_change_l2=delta,
        all_weights_finite=finite,world_size=2,state_sha256=d['state_sha256'],
        model_ready=ck['model_ready'],physical_acceptance=False)
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
