"""Ablate narrow-stage support updates without changing the aiming controller."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--support-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    torch.set_num_threads(2)
    current = torch.load(a.source, map_location='cpu', weights_only=False)
    old = torch.load(a.support_source, map_location='cpu', weights_only=False)
    assert not current.get('smoke') and not old.get('smoke')
    task = WideAimingTask('cpu', current['config'])
    validate_checkpoint(current, task)
    pi = {k: v.clone() for k, v in current['pi'].items()}
    pi['enc.0.weight'][:, :6] = old['pi']['enc.0.weight'][:, :6]
    for k, v in current['pi'].items():
        if k == 'enc.0.weight':
            assert torch.equal(v[:, 6:], pi[k][:, 6:])
        else:
            assert torch.equal(v, pi[k]), k
    before = FullbodyPolicy(task.robot)
    before.load_state_dict(current['pi'])
    after = FullbodyPolicy(task.robot)
    after.load_state_dict(pi)
    torch.manual_seed(271)
    h, c = torch.randn(16, 10, 99), torch.randn(16, 107)
    with torch.no_grad():
        # All right-arm rows are constant plus the unchanged command heads.
        torch.testing.assert_close(before(h, c)[:, 22:29], after(h, c)[:, 22:29], atol=0, rtol=0)
        c[:, :6] = 0
        torch.testing.assert_close(before(h, c), after(h, c), atol=0, rtol=0)
    provenance = dict(path=str(a.support_source.resolve()), sha256=sha256(a.support_source))
    diagnostic = dict(diagnostic_only=True, physical_acceptance=False,
                      only_six_support_columns_restored=True,
                      arm_actions_and_zero_support_actions_exact=True,
                      support_source=provenance,
                      support_change_l2=float((pi['enc.0.weight'][:, :6] - current['pi']['enc.0.weight'][:, :6]).norm()))
    candidate = dict(current, stage='diagnostic', pi=pi,
                     source_checkpoint=dict(path=str(a.source.resolve()), sha256=sha256(a.source)),
                     support_restore_diagnostic=diagnostic)
    validate_checkpoint(candidate, task)
    save_checkpoint(a.out, candidate)
    json_save(a.out.with_suffix('.json'), dict(checkpoint_sha256=sha256(a.out), **diagnostic))
    print(a.out)


if __name__ == '__main__':
    main()
