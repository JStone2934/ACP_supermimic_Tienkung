import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask

class KinematicHeadTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(713);self.p=FullbodyPolicy();self.h=torch.randn(4,10,99);self.c=torch.randn(4,107)
 def test_zero_preservation_and_distal_only_reload_trace(self):
  with torch.no_grad():old=self.p(self.h,self.c)
  self.p.enable_distal_kinematic_head()
  with torch.no_grad():
   torch.testing.assert_close(self.p(self.h,self.c),old,atol=0,rtol=0)
   self.p.distal_kinematic_head.output_scale.fill_(.5);self.p.distal_kinematic_head.output_bias.fill_(.1)
   new=self.p(self.h,self.c);torch.testing.assert_close(new[:,:26],old[:,:26],atol=0,rtol=0)
  loaded=FullbodyPolicy();loaded.load_state_dict(self.p.state_dict())
  with torch.no_grad():torch.testing.assert_close(loaded(self.h,self.c),new,atol=0,rtol=0)
  traced=torch.jit.trace(loaded,(self.h,self.c))
  with torch.no_grad():torch.testing.assert_close(traced(self.h,self.c),new,atol=0,rtol=0)
 def test_contract_and_layout_reject_mismatch_without_rng_side_effect(self):
  self.p.enable_distal_kinematic_head();cfg=dict(initial_q=list(self.p.robot.default_q),initial_height=self.p.robot.stand_height,distal_kinematic_head=True);task=WideAimingTask('cpu',cfg);ck=dict(pi=self.p.state_dict(),task_config=task.metadata(),robot_metadata=task.robot.metadata());rng=torch.get_rng_state().clone();validate_checkpoint(ck,task);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  with self.assertRaises(ValueError):validate_checkpoint(dict(ck,pi={}),task)
  bad=dict(ck['pi']);bad['distal_kinematic_head.net.0.weight']=torch.zeros(32,4)
  with self.assertRaises(ValueError):FullbodyPolicy().load_state_dict(bad)
 def test_reference_free_runtime_with_both_heads(self):
  from supermimic.acp.aiming_policy import AimingPolicy
  self.p.enable_distal_command_skip();self.p.enable_distal_kinematic_head()
  with torch.no_grad():
   self.p.distal_kinematic_head.output_scale.fill_(.2);self.p.distal_command_skip.weight.normal_(std=.01)
  cfg=dict(initial_q=list(self.p.robot.default_q),initial_height=self.p.robot.stand_height,distal_command_skip=True,distal_kinematic_head=True,support_capture_steps=25,support_command_mode='error')
  task=WideAimingTask('cpu',cfg)
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'runtime_test.pt';torch.save(dict(config=cfg,task_config=task.metadata(),robot_metadata=task.robot.metadata(),pi=self.p.state_dict()),path)
   with patch('supermimic.acp.motion.MotionLib.__init__',side_effect=AssertionError('Runtime must not read motion files')):
    runtime=AimingPolicy(path);state=task.initial_state(2);runtime.reset(state,torch.tensor([[3.,0.,0.,.05]]).expand(2,-1))
    hist=task.mdp.obs(state,runtime.previous_action,runtime.xi)[:,None].repeat(1,10,1)
    with torch.no_grad():expected=task.compose_action(self.p(hist,task.command(state,runtime.context)),runtime.xi)[1]
    torch.testing.assert_close(runtime.step(state),expected,atol=0,rtol=0)
 def test_bounded_finite_and_target_scale_invariant(self):
  self.p.enable_distal_kinematic_head();head=self.p.distal_kinematic_head
  with torch.no_grad():
   head.output_scale.fill_(2);head.output_bias.fill_(.3);x=torch.nn.functional.normalize(self.c[:,93:96],dim=-1);y=head(x)
   self.assertTrue(bool(((y-.3).abs()<=2).all()));torch.testing.assert_close(head(torch.nn.functional.normalize(7*self.c[:,93:96],dim=-1)),y)
   self.assertTrue(bool(self.p(self.h,torch.zeros_like(self.c)).isfinite().all()))
if __name__=='__main__':unittest.main()
