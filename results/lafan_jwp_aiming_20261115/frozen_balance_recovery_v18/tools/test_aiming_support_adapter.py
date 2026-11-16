import unittest

import torch

from supermimic.acp.aiming_adapters import configure_policy_training,policy_optimizer
from supermimic.acp.aiming_fullbody import FullbodyPolicy


class AimingSupportAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.pi=FullbodyPolicy();self.pi.enable_distal_command_skip()
        self.ids=[26,27,28]
        with torch.no_grad():
            self.pi.dec[-1].weight[self.ids]=0
            self.pi.distal_command_skip.weight.normal_(std=.1)
        self.cfg=dict(curriculum_level=2,support_command_mode='error',support_capture_steps=25,
            distal_command_skip=True,policy_training_scope='aiming_support_error_inputs',actor_lr=.001)

    def test_support_learning_keeps_aiming_and_zero_error_actions_exact(self):
        h=torch.randn(8,10,99);c=torch.randn(8,107);zero=c.clone();zero[:,:6]=0
        with torch.no_grad():before=self.pi(h,c);zero_before=self.pi(h,zero)
        configure_policy_training(self.pi,self.cfg)
        self.assertEqual([name for name,p in self.pi.named_parameters() if p.requires_grad],['enc.0.weight'])
        saved={k:v.clone() for k,v in self.pi.state_dict().items()};op=policy_optimizer(self.pi,self.cfg)
        for _ in range(3):
            op.zero_grad();(self.pi(h,c)[:,:12]-1.).square().mean().backward()
            self.assertEqual(torch.count_nonzero(self.pi.enc[0].weight.grad[:,6:]),0)
            op.step()
        for key,value in self.pi.state_dict().items():
            actual,expected=(value[:,6:],saved[key][:,6:]) if key=='enc.0.weight' else (value,saved[key])
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertGreater(float((self.pi.enc[0].weight[:,:6]-saved['enc.0.weight'][:,:6]).detach().norm()),0)
        with torch.no_grad():
            torch.testing.assert_close(self.pi(h,c)[:,self.ids],before[:,self.ids],atol=0,rtol=0)
            torch.testing.assert_close(self.pi(h,zero),zero_before,atol=0,rtol=0)

    def test_rejects_a_distal_decoder_that_would_change_with_support_learning(self):
        with torch.no_grad():self.pi.dec[-1].weight[26,0]=.1
        with self.assertRaises(ValueError):configure_policy_training(self.pi,self.cfg)

    def test_rejects_missing_head_and_wrong_stage(self):
        with self.assertRaises(ValueError):configure_policy_training(self.pi,dict(self.cfg,distal_command_skip=False))
        with self.assertRaises(ValueError):configure_policy_training(self.pi,dict(self.cfg,curriculum_level=0))


if __name__=='__main__':unittest.main()
