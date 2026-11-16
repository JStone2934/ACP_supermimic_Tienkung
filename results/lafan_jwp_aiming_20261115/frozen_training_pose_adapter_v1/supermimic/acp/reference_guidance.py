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


def expand_critic_command(state, extra_dim):
    """Adding ignored command columns preserves a trained value function exactly."""
    if extra_dim < 0:raise ValueError('Cannot remove critic input columns')
    result={k:v.detach().clone() for k,v in state.items()}
    weight=result['net.0.weight'];width=weight.shape[1]+extra_dim
    expanded=weight.new_zeros(weight.shape[0],width);expanded[:,:weight.shape[1]]=weight
    result['net.0.weight']=expanded
    for key,fill in [('norm.mean',0.),('norm.var',1.)]:
        old=result[key];result[key]=old.new_full((width,),fill);result[key][:len(old)]=old
    return result


def cap_reference_value(value, frames, clip_end, clip_id, max_reward, gamma):
    """A finite reference episode cannot exceed its remaining discounted reward."""
    remaining=(clip_end[clip_id[frames]]-frames).clamp_min(0).to(value.dtype)
    upper=max_reward*(1-gamma**remaining)/(1-gamma)
    return value.minimum(upper)


def joint_target_loss(action, xi, default_q, action_scales, reference_q):
    joints=action.shape[-1]
    requested_q=default_q+xi[...,1:1+joints]+action_scales*action
    return (requested_q-reference_q).square().mean()
