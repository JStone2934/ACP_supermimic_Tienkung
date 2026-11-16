"""Restrict balance adaptation to new measured support-error input columns."""
import torch


def configure_policy_training(policy,config):
    scope=config.get('policy_training_scope','full')
    if scope=='full':return
    if (scope!='support_error_inputs' or config.get('support_command_mode')!='error'
            or config.get('curriculum_level',0)!=0 or not config.get('support_capture_steps')):
        raise ValueError('Support-input-only training requires captured error inputs at balance level0')
    if torch.count_nonzero(policy.goal_norm.mean[:6]):
        raise ValueError('Support error normalizer must map zero error to zero input')
    policy.requires_grad_(False)
    weight=policy.enc[0].weight
    weight.requires_grad_(True)
    mask=torch.zeros_like(weight);mask[:,:6]=1
    weight.register_hook(lambda gradient:gradient*mask)
