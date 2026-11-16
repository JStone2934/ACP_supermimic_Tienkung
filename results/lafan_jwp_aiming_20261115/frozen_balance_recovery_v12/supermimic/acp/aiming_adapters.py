"""Restrict balance/aim adaptation while retaining the learned body network."""
import torch


def configure_policy_training(policy,config):
    scope=config.get('policy_training_scope','full')
    if scope=='full':return
    level=config.get('curriculum_level',0)
    allowed=(scope=='support_error_inputs' and level==0 or
             scope in ('support_and_right_arm_outputs','right_arm_outputs') and level in range(1,7))
    if not allowed or config.get('support_command_mode')!='error' or not config.get('support_capture_steps'):
        raise ValueError('Restricted training requires captured error inputs and the corresponding balance/aiming stage')
    if torch.count_nonzero(policy.goal_norm.mean[:6]):
        raise ValueError('Support error normalizer must map zero error to zero input')
    policy.requires_grad_(False)
    if scope!='right_arm_outputs':
        weight=policy.enc[0].weight
        weight.requires_grad_(True)
        mask=torch.zeros_like(weight);mask[:,:6]=1
        weight.register_hook(lambda gradient:gradient*mask)
    if scope in ('support_and_right_arm_outputs','right_arm_outputs'):
        from .standing import ARM_NAMES
        ids=[policy.robot.joint_names.index(name) for name in ARM_NAMES]
        # The decoder's final layer has separate output rows for native joints.
        # With fixed inputs and zero support error, all other joint outputs stay exact.
        for parameter in (policy.dec[-1].weight,policy.dec[-1].bias):
            parameter.requires_grad_(True)
            arm_mask=torch.zeros_like(parameter);arm_mask[ids]=1
            parameter.register_hook(lambda gradient,mask=arm_mask:gradient*mask)
