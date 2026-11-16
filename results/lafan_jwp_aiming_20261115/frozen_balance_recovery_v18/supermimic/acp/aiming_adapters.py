"""Restrict balance/aim adaptation while retaining the learned body network."""
import torch
import math


def set_policy_learning_rates(optimizer,config,*,distill=False):
    if distill:
        rates=[config['distill_lr']]*len(optimizer.param_groups)
    elif config.get('support_actor_lr') is not None:
        rates=[config['support_actor_lr'],config['actor_lr']]
    else:
        rates=[config['actor_lr']]
    if len(rates)!=len(optimizer.param_groups):
        raise ValueError('Policy optimizer groups do not match configured learning rates')
    for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate


def policy_optimizer(policy,config):
    """Keep support adaptation separate from the smaller right-arm update rate."""
    support_lr=config.get('support_actor_lr')
    if support_lr is None:
        return torch.optim.Adam(policy.parameters(),lr=config['actor_lr'])
    if config.get('policy_training_scope')!='support_and_right_arm_outputs' or not math.isfinite(support_lr) or support_lr<=0:
        raise ValueError('Separate support learning rate requires combined support/right-arm training')
    support=policy.enc[0].weight
    arm=[p for p in policy.parameters() if p.requires_grad and p is not support]
    if not support.requires_grad or len(arm)!=2:
        raise ValueError('Configure the combined parameter masks before constructing the optimizer')
    return torch.optim.Adam([dict(params=[support],lr=support_lr),dict(params=arm,lr=config['actor_lr'])])


def configure_policy_training(policy,config):
    scope=config.get('policy_training_scope','full')
    if scope=='full':return
    level=config.get('curriculum_level',0)
    allowed=(scope=='support_error_inputs' and level==0 or
             scope=='aiming_support_error_inputs' and level in range(1,7) or
             scope in ('support_and_right_arm_outputs','right_arm_outputs','right_distal_outputs','right_distal_with_command_skip') and level in range(1,7))
    if not allowed or config.get('support_command_mode')!='error' or not config.get('support_capture_steps'):
        raise ValueError('Restricted training requires captured error inputs and the corresponding balance/aiming stage')
    if torch.count_nonzero(policy.goal_norm.mean[:6]):
        raise ValueError('Support error normalizer must map zero error to zero input')
    if scope=='aiming_support_error_inputs':
        names=('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')
        ids=[policy.robot.joint_names.index(name) for name in names]
        if (not config.get('distal_command_skip') or not hasattr(policy,'distal_command_skip')
                or torch.count_nonzero(policy.dec[-1].weight[ids])):
            raise ValueError('Aiming support adaptation requires a fixed distal decoder equilibrium and independent direction head')
    policy.requires_grad_(False)
    if scope not in ('right_arm_outputs','right_distal_outputs','right_distal_with_command_skip'):
        weight=policy.enc[0].weight
        weight.requires_grad_(True)
        mask=torch.zeros_like(weight);mask[:,:6]=1
        weight.register_hook(lambda gradient:gradient*mask)
    if scope in ('support_and_right_arm_outputs','right_arm_outputs','right_distal_outputs','right_distal_with_command_skip'):
        from .standing import ARM_NAMES
        names=('elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint') if scope in ('right_distal_outputs','right_distal_with_command_skip') else ARM_NAMES
        ids=[policy.robot.joint_names.index(name) for name in names]
        # The decoder's final layer has separate output rows for native joints.
        # With fixed inputs and zero support error, all other joint outputs stay exact.
        for parameter in (policy.dec[-1].weight,policy.dec[-1].bias):
            parameter.requires_grad_(True)
            arm_mask=torch.zeros_like(parameter);arm_mask[ids]=1
            parameter.register_hook(lambda gradient,mask=arm_mask:gradient*mask)
    if scope=='right_distal_with_command_skip':
        if not config.get('distal_command_skip') or not hasattr(policy,'distal_command_skip'):
            raise ValueError('Direct distal command training requires the matching policy contract and weights')
        policy.distal_command_skip.requires_grad_(True)
