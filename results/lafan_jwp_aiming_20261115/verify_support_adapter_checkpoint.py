"""Audit a formal feedback checkpoint against its frozen-backbone initialization."""
import argparse,hashlib,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    base=Path('/home/js/robot/locomotion/standing-aiming/results/lafan_jwp_aiming_20261115')
    source=base/'support_error_adapter_init.pt'
    initial=torch.load(source,map_location='cpu',weights_only=False)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert not ck['smoke'] and ck['config']['policy_training_scope']=='support_error_inputs'
    assert ck['config']['support_command_mode']=='error'
    unchanged={}
    for k,v in initial['pi'].items():
        unchanged[k]=torch.equal(v[:,6:],ck['pi'][k][:,6:]) if k=='enc.0.weight' else torch.equal(v,ck['pi'][k])
    assert all(unchanged.values()),[k for k,v in unchanged.items() if not v]
    d=ck['distributed'];assert d['world_size']==2 and not d['skipped'] and d['state_sha256']
    weights=ck['pi']['enc.0.weight'][:,:6]
    changed=not torch.equal(weights,initial['pi']['enc.0.weight'][:,:6])
    assert changed==bool(d['updates'].get('pi',0))
    finite={n:all(bool(v.isfinite().all()) for v in ck[n].values() if torch.is_tensor(v)) for n in ('pi','wm','critic')}
    assert all(finite.values())
    report=dict(passed=True,scope='Formal artifact identity, finite values, distributed synchronization and frozen policy preservation. Not physical policy acceptance.',
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        initialization_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),iteration=ck['iteration'],updates=d['updates'],
        all_previous_policy_tensors_preserved=True,new_columns_changed=changed,
        new_columns_l2_norm=float(weights.norm()),new_columns_max_abs=float(weights.abs().max()),
        all_weights_finite=finite,world_size=d['world_size'],state_sha256=d['state_sha256'],
        model_ready=ck['model_ready'],policy_acceptance=False)
    a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
