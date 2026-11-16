"""Fold the legacy relative-heading encoder input to zero for a narrow-target diagnostic."""
import argparse
import json
from pathlib import Path

import torch

from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    digest = sha256(a.source)
    ck = torch.load(a.source, map_location='cpu', weights_only=False)
    assert not ck.get('smoke') and ck['stage'] in ('training', 'calibration')
    assert ck['config']['curriculum_level'] == 2
    assert ck['config']['distal_command_skip']
    task = WideAimingTask('cpu', ck['config'])
    validate_checkpoint(ck, task)
    original = FullbodyPolicy(task.robot); original.load_state_dict(ck['pi'])
    pi = {key: value.clone() for key, value in ck['pi'].items()}
    constant = -pi['goal_norm.mean'][29] / original.goal_norm.std()[29]
    pi['enc.0.bias'] += pi['enc.0.weight'][:, 29] * constant
    pi['enc.0.weight'][:, 29] = 0
    for key in pi:
        if key not in ('enc.0.weight', 'enc.0.bias'):
            assert torch.equal(pi[key], ck['pi'][key]), key
    keep = [i for i in range(107) if i != 29]
    assert torch.equal(pi['enc.0.weight'][:, keep], ck['pi']['enc.0.weight'][:, keep])
    policy = FullbodyPolicy(task.robot); policy.load_state_dict(pi)
    torch.manual_seed(137)
    history = torch.randn(32, 10, 99)
    command = torch.randn(32, 107)
    fixed = command.clone(); fixed[:, 29] = 0
    with torch.no_grad():
        expected = original(history, fixed)
        actual = policy(history, command)
        error = float(((actual - expected) * task.alpha).abs().max())
        assert error < 1e-5, error
        varied = command.clone(); varied[:, 29] += 10
        torch.testing.assert_close(policy(history, varied), actual, atol=0, rtol=0)
    cfg = dict(ck['config'], initialization_mode='narrow_legacy_heading_input_fold', distill_iters=0,
               wm_only_iters=100, critic_warmup_iters=100)
    calibration = dict(method=__doc__, command_column=29, constant_raw_value=0.,
        original_with_explicit_fixed_input_max_pd_difference_rad=error,
        decoder_and_all_normalizers_exact=True, support_error_columns_exact=True,
        direct_target_feedback_head_exact=True, physical_acceptance=False,
        limitation='Disables the legacy encoder heading response; intended only for narrow targets reachable by the distal arm. Wide-target turning remains unresolved.')
    artifact = dict(task=ck['task'], stage='calibration', smoke=False, config=cfg,
        task_config=WideAimingTask('cpu', cfg).metadata(), robot_metadata=ck['robot_metadata'], pi=pi,
        wm=ck['wm'], provenance=ck.get('provenance'), calibration=calibration,
        source_checkpoint=dict(path=str(a.source.resolve()), sha256=digest))
    validate_checkpoint(artifact, WideAimingTask('cpu', cfg))
    save_checkpoint(a.out, artifact)
    json_save(a.out.with_suffix('.json'), dict(checkpoint_sha256=sha256(a.out), calibration=calibration))
    assert sha256(a.source) == digest
    print(json.dumps(calibration))


if __name__ == '__main__':
    main()
