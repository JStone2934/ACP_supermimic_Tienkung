"""Add a local target feedforward term to an audited affine distal feedback head."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import json_save, save_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256
from supermimic.rotations import quat_to_mat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'geometry-source', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--gains', type=float, nargs='+', default=[.5, 1.])
    a = p.parse_args()
    torch.set_num_threads(2)
    assert all(np.isfinite(x) and 0 < x <= 1.5 for x in a.gains)
    ck = torch.load(a.source, map_location='cpu', weights_only=False)
    geo = torch.load(a.geometry_source, map_location='cpu', weights_only=False)
    assert not ck.get('smoke') and ck['stage'] in ('training', 'calibration')
    assert geo['stage'] == 'calibration' and not geo.get('smoke')
    geometry = geo['calibration']['geometry']
    assert geometry['torch_mujoco_jacobian_max_abs_difference'] < 1e-5
    # This calibration assumes the exact audited feedback policy, before actor updates.
    assert ck['pi'].keys() == geo['pi'].keys()
    for key in ck['pi']:
        assert torch.equal(ck['pi'][key], geo['pi'][key]), key
    task = WideAimingTask('cpu', ck['config'])
    validate_checkpoint(ck, task)
    path = Path(geometry['trajectory'])
    assert sha256(path) == geometry['trajectory_sha256']
    with np.load(path) as z:
        frame, index = geometry['frame'], geometry['scenario']
        root = torch.tensor(z['root'][frame, index])[None]
        quat = torch.tensor(z['root_quat'][frame, index])[None]
        q = torch.tensor(z['q'][frame, index])[None]
    state = (root, quat_to_mat(quat), torch.zeros_like(root), torch.zeros_like(root), q, torch.zeros_like(q))
    _, direction, torso = task.hand(state)
    reference = (torso.transpose(-1, -2) @ direction[..., None]).squeeze()[...]
    ids = [task.robot.joint_names.index(n) for n in geometry['joint_names']]
    inverse = torch.tensor(geometry['pseudoinverse'])
    source_hash = sha256(a.source)
    a.out.mkdir(parents=True, exist_ok=True)
    for number, gain in enumerate(a.gains, 1):
        out = a.out / f'ckpt_{number:06d}.pt'
        if out.exists():
            raise FileExistsError(out)
        pi = {key: value.clone() for key, value in ck['pi'].items()}
        weight = gain * inverse / task.alpha[ids, None]
        pi['distal_command_skip.weight'][:, :3] += weight
        pi['distal_command_skip.bias'] -= weight @ reference
        for key in pi:
            if key not in ('distal_command_skip.weight', 'distal_command_skip.bias'):
                assert torch.equal(pi[key], ck['pi'][key]), key
        original = FullbodyPolicy(task.robot); original.load_state_dict(ck['pi'])
        policy = FullbodyPolicy(task.robot); policy.load_state_dict(pi)
        torch.manual_seed(191)
        history = torch.randn(16, 10, 99)
        command = torch.randn(16, 107)
        command[0, 93:96] = reference
        with torch.no_grad():
            delta = policy(history, command) - original(history, command)
            expected = (torch.nn.functional.normalize(command[:, 93:96], dim=-1) - reference) @ weight.T
        body = [i for i in range(29) if i not in ids]
        torch.testing.assert_close(delta[:, body], torch.zeros_like(delta[:, body]), atol=0, rtol=0)
        torch.testing.assert_close(delta[:, ids], expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(delta[0], torch.zeros_like(delta[0]), atol=2e-5, rtol=0)
        cfg = dict(ck['config'], initialization_mode='local_distal_feedforward_calibration', distill_iters=0, wm_only_iters=100, critic_warmup_iters=100)
        calibration = dict(method=__doc__, feedforward_gain=gain, feedback_gain=geo['calibration']['gain'],
            geometry_source=str(a.geometry_source.resolve()), geometry_source_sha256=sha256(a.geometry_source),
            reference_direction_torso=reference.tolist(), all_encoder_decoder_weights_exact=True,
            normalizers_exact=True, online_ik=False, runtime_motion_files=False, physical_acceptance=False,
            formula='Existing feedback + gain * J_pseudoinverse * (unit_target_direction - reference_direction) / action_scale')
        artifact = dict(task=ck['task'], stage='calibration', smoke=False, config=cfg,
            task_config=WideAimingTask('cpu', cfg).metadata(), robot_metadata=ck['robot_metadata'], pi=pi, wm=ck['wm'],
            provenance=ck.get('provenance'), source_checkpoint=dict(path=str(a.source.resolve()), sha256=source_hash),
            calibration=calibration)
        validate_checkpoint(artifact, WideAimingTask('cpu', cfg))
        save_checkpoint(out, artifact)
        json_save(out.with_suffix('.json'), dict(checkpoint_sha256=sha256(out), calibration=calibration))
        print(json.dumps(dict(output=str(out), gain=gain)), flush=True)
    assert sha256(a.source) == source_hash


if __name__ == '__main__':
    main()
