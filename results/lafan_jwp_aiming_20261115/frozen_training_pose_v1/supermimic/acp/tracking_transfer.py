"""Lossless contact-tracker initialization for the 107-column aiming policy."""
import torch
from .aiming_fullbody import FullbodyPolicy
from .task import to_base, goal_command
from .terrain import height_scan
from .aiming_fullbody import flat_boxes


def fixed_support_command(task,state,context):
    """Teacher's fixed initial support points, computed from geometry, not labels."""
    initial=task.initial_state(len(state[0]))
    initial=(context['root_anchor_w'],*initial[1:])
    bp,br,_,_=task.kin.forward(*initial)
    centers=bp[:,task.foot_links]+(br[:,task.foot_links]@task.foot_points[...,None]).squeeze(-1)
    targets=state[0].new_zeros(len(state[0]),9,3)
    for foot in (0,1):
        mask=task.foot_owners==foot;points=centers[:,mask]
        index=(points[...,2]-task.foot_radii[mask]).argmin(-1)
        point=points.gather(1,index[:,None,None].expand(-1,1,3)).squeeze(1)
        targets[:,foot,:2]=point[:,:2]
    local=to_base(targets,state[0],state[1])
    local[:,2:]=0
    return torch.cat((local.flatten(1),
        goal_command(state,context['root_anchor_w'],context['yaw_anchor'][:,0],.5),
        height_scan(state[0],state[1],flat_boxes(state),task.policy_scan)),-1)


def transfer_tracker(checkpoint, robot, device='cpu', fixed_q=None):
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
    pose_conditioned = bool(cfg.get('reference_pose', False))
    if weight.shape[1] != 93 + (robot.num_joints if pose_conditioned else 0):
        raise ValueError('Unsupported command width')
    if pose_conditioned:
        fixed_q = torch.as_tensor(fixed_q, device=device, dtype=weight.dtype) if fixed_q is not None else None
        if fixed_q is None or fixed_q.shape != (robot.num_joints,) or not bool(fixed_q.isfinite().all()):
            raise ValueError('Pose-conditioned transfer requires a finite fixed joint pose')
        # Constant reference pose is absorbed exactly into the first layer bias.
        # The aiming policy needs no reference input or motion file at runtime.
        normalized_q = (fixed_q-state['goal_norm.mean'][93:]) / (state['goal_norm.var'][93:] + 1e-4).sqrt()
        state['enc.0.bias'] = state['enc.0.bias'] + weight[:, 93:] @ normalized_q
        state['goal_norm.mean'] = state['goal_norm.mean'][:93].clone()
        state['goal_norm.var'] = state['goal_norm.var'][:93].clone()
    widened = weight.new_zeros(weight.shape[0], 107)
    widened[:, :93] = weight[:, :93]
    state['enc.0.weight'] = widened
    policy = FullbodyPolicy(robot, freeze_norm=True).to(device)
    policy.load_state_dict(state, strict=True)
    return policy, dict(source_command_dim=weight.shape[1], destination_command_dim=107,
                        fixed_pose_folded_into_bias=pose_conditioned,
                        physics_substeps=args.get('physics_substeps') or robot.decimation,
                        preserved_output_names=list(robot.joint_names),
                        new_columns='zero', normalizers='source frozen',
                        critic='new', optimizers='new',
                        scope='Exact policy migration only; task transfer still requires training')
