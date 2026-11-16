"""Check two-rank integration and exact preservation of the frozen balance network."""
import hashlib
import json
from pathlib import Path
import torch


def main():
    torch.set_num_threads(2)
    root=Path('/home/js/robot/locomotion/standing-aiming')
    base=root/'results/lafan_jwp_aiming_20261115'
    run=root/'logs/tienkung3_29dof/lafan_jwp_20261115/support_error_adapter_smoke/train'
    path=run/'ckpt_000002.pt';source=base/'support_error_adapter_init.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    initial=torch.load(source,map_location='cpu',weights_only=False)
    status=json.loads((run/'status.json').read_text());distributed=ck['distributed']
    assert ck['smoke'] and ck['iteration']==2 and distributed['world_size']==2
    assert not distributed['skipped'] and distributed['state_sha256']
    assert all(distributed['updates'][name]==2 for name in ('wm','pi','critic'))
    assert {r['device'] for r in distributed['ranks']}=={'cuda:0','cuda:1'}
    assert ck['config']['support_command_mode']=='error'
    assert ck['config']['policy_training_scope']=='support_error_inputs'
    frozen={}
    for key,value in initial['pi'].items():
        if key=='enc.0.weight':frozen[key+'[:,6:]']=torch.equal(value[:,6:],ck['pi'][key][:,6:])
        else:frozen[key]=torch.equal(value,ck['pi'][key])
    assert all(frozen.values())
    changed=not torch.equal(initial['pi']['enc.0.weight'][:,:6],ck['pi']['enc.0.weight'][:,:6])
    assert changed and status['observed_support_locked_fraction']>0
    finite={name:all(bool(v.isfinite().all()) for v in ck[name].values() if torch.is_tensor(v))
            for name in ('pi','wm','critic')}
    assert all(finite.values())
    assert ck['source_checkpoint']['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    result=dict(passed=True,scope='Two dual-GPU integration updates. All previous policy parameters and normalization buffers exactly preserved; only the six new error-input columns changed.',
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),distributed=distributed,
                all_weights_finite=finite,frozen_policy_elements_unchanged=all(frozen.values()),
                new_input_columns_changed=changed,observed_support_locked_fraction=status['observed_support_locked_fraction'],
                world_model_gate_passed=ck['model_ready'],smoke_gate_bypass=True,
                policy_acceptance=False,formal_initialization_allowed=False)
    (base/'support_error_adapter_smoke_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
