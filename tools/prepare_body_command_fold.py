"""Isolate a fixed-posture body controller from the independent aiming heads."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();torch.set_num_threads(2)
    ck=torch.load(a.source,map_location='cpu',weights_only=False)
    assert not ck.get('smoke') and ck['config'].get('distal_kinematic_head')
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    assert not ck['pi']['dec.8.weight'][22:29].count_nonzero()
    state=task.initial_state(1)
    context=task.context(state,torch.tensor([[3.,0.,0.,.05]]))
    reference=task.command(state,context)[0,93:104]
    pi={k:v.clone() for k,v in ck['pi'].items()}
    pi['enc.0.bias']+=pi['enc.0.weight'][:,93:104]@reference
    pi['enc.0.weight'][:,93:104]=0
    old,new=FullbodyPolicy(task.robot),FullbodyPolicy(task.robot)
    old.load_state_dict(ck['pi']);new.load_state_dict(pi)
    torch.manual_seed(271);h,c=torch.randn(16,10,99),torch.randn(16,107)
    masked=c.clone();masked[:,93:104]=reference
    with torch.no_grad():
        torch.testing.assert_close(new(h,c)[:,:22],old(h,masked)[:,:22],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(new(h,c)[:,22:29],old(h,c)[:,22:29],atol=0,rtol=0)
    for k,v in ck['pi'].items():
        if k not in ('enc.0.weight','enc.0.bias'):assert torch.equal(v,pi[k]),k
    diagnostic=dict(diagnostic_only=True,physical_acceptance=False,
        method=__doc__,reference_raw_command_93_104=reference.tolist(),
        body_equivalent_to_constant_aim_command=True,right_arm_actions_exact=True,
        normalizers_and_reset_exact=True,velocity_command_columns_preserved=True)
    candidate=dict(ck,stage='diagnostic',pi=pi,
        source_checkpoint=dict(path=str(a.source.resolve()),sha256=sha256(a.source)),
        body_command_fold=diagnostic)
    validate_checkpoint(candidate,task);save_checkpoint(a.out,candidate)
    json_save(a.out.with_suffix('.json'),dict(checkpoint_sha256=sha256(a.out),**diagnostic))
    print(a.out)


if __name__=='__main__':main()
