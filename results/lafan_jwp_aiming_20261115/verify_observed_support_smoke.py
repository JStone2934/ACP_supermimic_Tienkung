"""Verify bounded integration execution, never physical policy acceptance."""
import hashlib
import json
from pathlib import Path
import torch


def main():
    torch.set_num_threads(2)
    root = Path('/home/js/robot/locomotion/standing-aiming')
    base = root / 'results/lafan_jwp_aiming_20261115'
    run = root / 'logs/tienkung3_29dof/lafan_jwp_20261115/observed_support_smoke/train'
    path = run / 'ckpt_000002.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    status = json.loads((run / 'status.json').read_text())
    distributed = ck['distributed']
    finite = {name: all(bool(v.isfinite().all()) for v in ck[name].values()
                       if torch.is_tensor(v)) for name in ('pi', 'wm', 'critic')}
    assert ck['smoke'] and ck['iteration'] == 2
    assert distributed['world_size'] == 2 and not distributed['skipped']
    assert all(distributed['updates'][name] == 2 for name in ('wm', 'pi', 'critic'))
    assert {r['device'] for r in distributed['ranks']} == {'cuda:0', 'cuda:1'}
    assert distributed['state_sha256'] and all(finite.values())
    assert ck['config']['support_capture_steps'] == 150
    assert ck['config']['wm_validation_steps'] >= 166
    assert status['observed_support_locked_fraction'] > 0
    assert ck['source_checkpoint']['sha256'] == hashlib.sha256((base / 'observed_support_init.pt').read_bytes()).hexdigest()
    result = dict(passed=True, scope='Exactly two dual-GPU integration updates; finite synchronized execution only.',
                  checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  distributed=distributed, all_weights_finite=finite,
                  observed_support_locked_fraction=status['observed_support_locked_fraction'],
                  world_model_gate_passed=ck['model_ready'], smoke_gate_bypass=True,
                  policy_acceptance=False, formal_initialization_allowed=False)
    (base / 'observed_support_smoke_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
