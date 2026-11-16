import unittest
import torch
from supermimic.acp.aiming_adapters import configure_policy_training, policy_optimizer
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.standing import ARM_NAMES


class BodyOutputAdapterTests(unittest.TestCase):
    def test_learning_changes_body_outputs_and_preserves_entire_aiming_path(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        pi=FullbodyPolicy()
        pi.enable_distal_command_skip()
        pi.enable_distal_kinematic_head()
        with torch.no_grad():
            pi.distal_command_skip.weight.normal_(std=.1)
            pi.distal_kinematic_head.output_scale.fill_(.1)
        cfg=dict(curriculum_level=4,support_command_mode='error',support_capture_steps=25,
                 policy_training_scope='body_outputs',actor_lr=1e-4)
        arm=[pi.robot.joint_names.index(n) for n in ARM_NAMES]
        body=[i for i in range(pi.robot.num_joints) if i not in arm]
        h,c=torch.randn(8,10,99),torch.randn(8,107)
        with torch.no_grad(): before=pi(h,c)
        saved={k:v.clone() for k,v in pi.state_dict().items()}
        configure_policy_training(pi,cfg)
        self.assertEqual([n for n,p in pi.named_parameters() if p.requires_grad],['dec.8.weight','dec.8.bias'])
        op=policy_optimizer(pi,cfg)
        for _ in range(3):
            op.zero_grad()
            (pi(h,c)[:,body]-1.).square().mean().backward()
            self.assertEqual(int(pi.dec[-1].weight.grad[arm].count_nonzero()),0)
            self.assertEqual(int(pi.dec[-1].bias.grad[arm].count_nonzero()),0)
            op.step()
        for k,v in pi.state_dict().items():
            if k in ('dec.8.weight','dec.8.bias'):
                torch.testing.assert_close(v[arm],saved[k][arm],atol=0,rtol=0)
            else:
                torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        with torch.no_grad(): after=pi(h,c)
        torch.testing.assert_close(after[:,arm],before[:,arm],atol=0,rtol=0)
        self.assertGreater(float((after[:,body]-before[:,body]).norm()),0)


if __name__=='__main__':unittest.main()
