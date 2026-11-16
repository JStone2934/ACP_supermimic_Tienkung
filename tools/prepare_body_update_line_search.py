"""Bounded physical direction probes around an ACP body-only update; not promotion."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','trained','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--factors',type=float,nargs='+',required=True)
    a=p.parse_args();assert all(-1<=v<=1 and v!=0 for v in a.factors)
    torch.set_num_threads(2)
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    trained=torch.load(a.trained,map_location='cpu',weights_only=False)
    assert not base.get('smoke') and not trained.get('smoke')
    assert trained['config']['policy_training_scope']=='body_outputs'
    original_path=Path(trained['source_checkpoint']['path'])
    assert sha256(original_path)==trained['source_checkpoint']['sha256']
    original=torch.load(original_path,map_location='cpu',weights_only=False)
    for k,v in base['pi'].items():
        assert torch.equal(v,original['pi'][k]),k
        if k in ('dec.8.weight','dec.8.bias'):
            assert torch.equal(v[22:29],trained['pi'][k][22:29])
        else:assert torch.equal(v,trained['pi'][k]),k
    task=WideAimingTask('cpu',base['config']);validate_checkpoint(base,task)
    old=FullbodyPolicy(task.robot);old.load_state_dict(base['pi'])
    torch.manual_seed(271);h,c=torch.randn(16,10,99),torch.randn(16,107)
    a.out.mkdir(parents=True,exist_ok=True)
    for i,factor in enumerate(a.factors,1):
        out=a.out/f'ckpt_{i:06d}.pt';assert not out.exists()
        pi={k:v.clone() for k,v in base['pi'].items()}
        for k in ('dec.8.weight','dec.8.bias'):
            pi[k][:22]+=factor*(trained['pi'][k][:22]-base['pi'][k][:22])
        new=FullbodyPolicy(task.robot);new.load_state_dict(pi)
        with torch.no_grad():
            before,after=old(h,c),new(h,c)
            assert after.isfinite().all()
            torch.testing.assert_close(after[:,22:29],before[:,22:29],atol=0,rtol=0)
        diagnostic=dict(diagnostic_only=True,physical_acceptance=False,factor=factor,
            method=__doc__,trained_checkpoint=dict(path=str(a.trained.resolve()),sha256=sha256(a.trained)),
            only_body_output_rows_changed=True,right_arm_actions_exact=True,
            normalizers_and_reset_exact=True,automatic_promotion=False)
        ck=dict(base,stage='diagnostic',pi=pi,
                source_checkpoint=dict(path=str(a.base.resolve()),sha256=sha256(a.base)),
                body_update_direction_probe=diagnostic)
        validate_checkpoint(ck,task);save_checkpoint(out,ck)
        json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),**diagnostic))
        print(out,flush=True)


if __name__=='__main__':main()
