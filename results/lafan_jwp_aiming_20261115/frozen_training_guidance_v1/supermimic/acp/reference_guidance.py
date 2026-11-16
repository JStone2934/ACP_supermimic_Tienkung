"""Dense training-only joint guidance in the simulator's PD target coordinates."""


def joint_target_loss(action, xi, default_q, action_scales, reference_q):
    joints=action.shape[-1]
    requested_q=default_q+xi[...,1:1+joints]+action_scales*action
    return (requested_q-reference_q).square().mean()
