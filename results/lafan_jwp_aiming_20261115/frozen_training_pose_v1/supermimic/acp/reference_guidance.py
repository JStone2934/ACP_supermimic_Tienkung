"""Dense training-only joint guidance in the simulator's PD target coordinates."""


def expand_pose_command(state, command_dim):
    """Preserve the contact policy exactly while adding initially ignored pose columns."""
    result = {k: v.detach().clone() for k, v in state.items()}
    weight = result['enc.0.weight']
    if command_dim <= weight.shape[1]:
        raise ValueError('Pose command must increase input width')
    expanded = weight.new_zeros(weight.shape[0], command_dim)
    expanded[:, :weight.shape[1]] = weight
    result['enc.0.weight'] = expanded
    for key, fill in [('goal_norm.mean', 0.), ('goal_norm.var', 1.)]:
        old = result[key]
        result[key] = old.new_full((command_dim,), fill)
        result[key][:len(old)] = old
    return result


def joint_target_loss(action, xi, default_q, action_scales, reference_q):
    joints=action.shape[-1]
    requested_q=default_q+xi[...,1:1+joints]+action_scales*action
    return (requested_q-reference_q).square().mean()
