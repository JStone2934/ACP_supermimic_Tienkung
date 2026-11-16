"""Lossless contact-tracker initialization for the 107-column aiming policy."""
import torch
from .aiming_fullbody import FullbodyPolicy


def transfer_tracker(checkpoint, robot, device='cpu'):
    if checkpoint.get('robot_metadata') != robot.metadata():
        raise ValueError('Tracker robot identity mismatch')
    if checkpoint.get('task_config') is not None:
        raise ValueError('Expected a full-joint tracking policy, not a fixed-arm task')
    args = checkpoint.get('args', {})
    if args.get('hist') != 10 or args.get('task', 'tracking') != 'tracking':
        raise ValueError('Expected ten-frame tracking history')
    cfg = checkpoint.get('ctc_cfg', {})
    if cfg.get('blind', False) or cfg.get('contact_geometry') != 'spheres224':
        raise ValueError('Expected 93-column spheres224 contact interface')
    state = {k: v.detach().clone().to(device) for k, v in checkpoint['pi'].items()}
    weight = state['enc.0.weight']
    if weight.shape[1] != 93:
        raise ValueError('Unsupported command width')
    widened = weight.new_zeros(weight.shape[0], 107)
    widened[:, :93] = weight
    state['enc.0.weight'] = widened
    policy = FullbodyPolicy(robot, freeze_norm=True).to(device)
    policy.load_state_dict(state, strict=True)
    return policy, dict(source_command_dim=93, destination_command_dim=107,
                        preserved_output_names=list(robot.joint_names),
                        new_columns='zero', normalizers='source frozen',
                        critic='new', optimizers='new',
                        scope='Exact policy migration only; task transfer still requires training')
