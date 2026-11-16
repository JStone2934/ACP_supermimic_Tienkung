import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_adapters import configure_policy_training,policy_optimizer


class DistalCommandSkipTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(40)
        self.p=FullbodyPolicy();self.h=torch.randn(8,10,99);self.c=torch.randn(8,107)
        self.ids=[26,27,28];self.body=list(range(26))

    def test_zero_initialization_and_serialized_reload_preserve_actions(self):
        with torch.no_grad():before=self.p(self.h,self.c)
        state={k:v.clone() for k,v in self.p.state_dict().items()}
        self.p.enable_distal_command_skip()
        with torch.no_grad():torch.testing.assert_close(self.p(self.h,self.c),before,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in self.p.distal_command_skip.parameters()),45)
        self.assertTrue(all(torch.equal(v,self.p.state_dict()[k]) for k,v in state.items()))
        with torch.no_grad():self.p.distal_command_skip.weight.normal_(std=.1)
        loaded=FullbodyPolicy();loaded.load_state_dict(self.p.state_dict())
        with torch.no_grad():torch.testing.assert_close(loaded(self.h,self.c),self.p(self.h,self.c),atol=0,rtol=0)

    def test_target_feedback_is_scale_invariant_finite_and_distal_only(self):
        with torch.no_grad():base=self.p(self.h,self.c)
        self.p.enable_distal_command_skip()
        with torch.no_grad():
            self.p.distal_command_skip.weight[:,:3]=torch.eye(3)
            after=self.p(self.h,self.c)
            torch.testing.assert_close(after[:,self.body],base[:,self.body],atol=0,rtol=0)
            torch.testing.assert_close(after[:,self.ids]-base[:,self.ids],torch.nn.functional.normalize(self.c[:,93:96],dim=-1))
            changed=self.c.clone();changed[:,93:96]*=7
            # The skip itself is scale-invariant; the original encoder also sees target distance.
            feature=lambda c:torch.cat((torch.nn.functional.normalize(c[:,93:96],dim=-1,eps=1e-6),c[:,96:]),-1)
            torch.testing.assert_close(self.p.distal_command_skip(feature(changed)),self.p.distal_command_skip(feature(self.c)))
            self.assertTrue(bool(self.p(self.h,torch.zeros_like(self.c)).isfinite().all()))

    def test_optimizer_keeps_all_body_support_and_normalizer_tensors_exact(self):
        self.p.enable_distal_command_skip()
        cfg=dict(curriculum_level=1,support_capture_steps=150,support_command_mode='error',
                 policy_training_scope='right_distal_with_command_skip',distal_command_skip=True,actor_lr=.001)
        configure_policy_training(self.p,cfg);saved={k:v.clone() for k,v in self.p.state_dict().items()}
        with torch.no_grad():before=self.p(self.h,self.c)
        optimizer=policy_optimizer(self.p,cfg)
        for _ in range(3):
            optimizer.zero_grad();(self.p(self.h,self.c)[:,self.ids]-1).square().mean().backward();optimizer.step()
        for key,value in self.p.state_dict().items():
            if key.startswith('distal_command_skip.'):continue
            if key in ('dec.8.weight','dec.8.bias'):torch.testing.assert_close(value[self.body],saved[key][self.body],atol=0,rtol=0)
            else:torch.testing.assert_close(value,saved[key],atol=0,rtol=0)
        with torch.no_grad():torch.testing.assert_close(self.p(self.h,self.c)[:,self.body],before[:,self.body],atol=0,rtol=0)
        self.assertGreater(float(self.p.distal_command_skip.weight.detach().abs().sum()),0.)

    def test_reference_free_loader_and_traced_forward_retain_readout(self):
        from supermimic.acp.aiming_policy import AimingPolicy
        self.p.enable_distal_command_skip()
        with torch.no_grad():self.p.distal_command_skip.weight.normal_(std=.01)
        config=dict(initial_q=list(self.p.robot.default_q),initial_height=self.p.robot.stand_height,
                    distal_command_skip=True,support_capture_steps=150,support_command_mode='error')
        task=WideAimingTask('cpu',config)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test_only.pt'
            torch.save(dict(config=config,task_config=task.metadata(),robot_metadata=task.robot.metadata(),pi=self.p.state_dict()),path)
            with patch('supermimic.acp.motion.MotionLib.__init__',side_effect=AssertionError('No motion reads')):
                runtime=AimingPolicy(path);state=task.initial_state(2)
                runtime.reset(state,torch.tensor([[3.,0.,0.,.05]]).expand(2,-1))
                history=task.mdp.obs(state,runtime.previous_action,runtime.xi)[:,None].repeat(1,10,1)
                with torch.no_grad():raw=self.p(history,task.command(state,runtime.context))
                expected=task.compose_action(raw,runtime.xi)[1]
                torch.testing.assert_close(runtime.step(state),expected,atol=0,rtol=0)
        traced=torch.jit.trace(self.p,(self.h,self.c))
        with torch.no_grad():torch.testing.assert_close(traced(self.h,self.c),self.p(self.h,self.c),atol=0,rtol=0)

    def test_layout_and_metadata_mismatches_are_rejected(self):
        self.p.enable_distal_command_skip();state=self.p.state_dict()
        bad=dict(state);bad['distal_command_skip.weight']=torch.zeros(3,15)
        with self.assertRaises(ValueError):FullbodyPolicy().load_state_dict(bad)
        bad=dict(state);del bad['distal_command_skip.bias']
        with self.assertRaises(ValueError):FullbodyPolicy().load_state_dict(bad)
        config=dict(initial_q=list(self.p.robot.default_q),initial_height=self.p.robot.stand_height)
        task=WideAimingTask('cpu',dict(config,distal_command_skip=True))
        ck=dict(pi=state,task_config=task.metadata(),robot_metadata=task.robot.metadata())
        validate_checkpoint(ck,task)
        with self.assertRaises(ValueError):validate_checkpoint(dict(ck,pi={}),task)
        with self.assertRaises(ValueError):validate_checkpoint(ck,WideAimingTask('cpu',config))


if __name__=='__main__':unittest.main()
