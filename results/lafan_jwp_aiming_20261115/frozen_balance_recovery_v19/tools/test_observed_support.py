"""Observed support targets retain adjustment freedom and require no future references."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_moving import MovingAimingTask
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.aiming_policy import AimingPolicy
from supermimic.acp.robots import get_robot
from tools.prepare_observed_support_init import initialize_support_inputs


class ObservedSupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);r=get_robot('tienkung3_29dof')
        self.cfg=dict(initial_q=list(r.default_q),initial_height=r.stand_height,support_capture_steps=150,
                      soft_termination_mode='compact',term_tilt_deg=30.,reward_weights=dict(feet=4.))
        self.task=WideAimingTask('cpu',self.cfg);self.S=self.task.initial_state(2)
        self.params=torch.tensor([[3.,0.,0.,.2]]).expand(2,-1)
        self.ctx=self.task.context(self.S,self.params,0.)

    def test_capture_uses_current_feet_once_and_rewards_only_locked_support(self):
        task=self.task;a=torch.zeros(2,29)
        self.assertEqual(int(torch.count_nonzero(task.command(self.S,self.ctx)[:,:27])),0)
        self.assertEqual(float(task.reward_terms(self.S,a,a,self.ctx)[1]['feet'].sum()),0.)
        shifted=(self.S[0]+torch.tensor([.2,-.1,0.]),*self.S[1:])
        ctx=task.observe_context(shifted,task.advance_context(self.ctx,149))
        self.assertTrue((ctx['support_age_steps']==149).all())
        ctx=task.observe_context(shifted,task.advance_context(ctx,1))
        self.assertTrue((ctx['support_age_steps']==-1).all())
        torch.testing.assert_close(ctx['feet_anchor_w'],task.kin.forward(*shifted)[0][:,task.feet],atol=0,rtol=0)
        drift=(shifted[0]+torch.tensor([.06,0.,0.]),*shifted[1:])
        later=task.observe_context(drift,task.advance_context(ctx,100))
        torch.testing.assert_close(later['feet_anchor_w'],ctx['feet_anchor_w'],atol=0,rtol=0)
        self.assertLess(float(task.reward_terms(drift,a,a,later)[0]['feet'].max()),.5)
        self.assertEqual(int(torch.count_nonzero(task.command(drift,later)[:,6:27])),0)
        torch.testing.assert_close(later['aim_target_w'],self.ctx['aim_target_w'],atol=0,rtol=0)

    def test_capture_cuts_imagination_but_does_not_terminate_physics(self):
        before=self.task.advance_context(self.ctx,149);boundary=self.task.advance_context(before,1)
        self.assertTrue((self.task.soft_termination(self.S,torch.tensor([150,150]),boundary)==1).all())
        self.assertFalse(self.task.termination(self.S,torch.tensor([150,150]),boundary)[0].any())
        locked=self.task.observe_context(self.S,boundary)
        self.assertTrue((self.task.soft_termination(self.S,torch.tensor([151,151]),locked)==0).all())

    def test_target_change_and_movement_release_observed_targets(self):
        task=MovingAimingTask('cpu',self.cfg);ctx=task.observe_context(self.S,task.advance_context(self.ctx,150))
        task.release_support(ctx,torch.tensor([0]));self.assertEqual(ctx['support_age_steps'].tolist(),[0,-1])
        ctx['velocity_command'][:]=torch.tensor([.2,0.,0.]);ctx=task.observe_context(self.S,ctx)
        self.assertEqual(ctx['support_age_steps'].tolist(),[0,0])
        self.assertEqual(int(torch.count_nonzero(task.command(self.S,ctx)[:,:27])),0)
        root=ctx['root_anchor_w'].clone();advanced=task.advance_context(ctx,2)
        torch.testing.assert_close(advanced['root_anchor_w'],root+torch.tensor([.008,0.,0.]))
        self.assertEqual(advanced['support_age_steps'].tolist(),[2,2])

    def test_initial_column_fold_preserves_policy_for_new_support_commands(self):
        torch.manual_seed(12);p=FullbodyPolicy(self.task.robot)
        p.goal_norm.mean.copy_(torch.randn_like(p.goal_norm.mean));p.goal_norm.var.copy_(torch.rand_like(p.goal_norm.var)+.2)
        history=torch.randn(2,10,99);old=torch.randn(2,107);old[:,:27]=0
        with torch.no_grad():expected=p(history,old);initialize_support_inputs(p);new=old.clone();new[:,:6]=torch.randn(2,6);actual=p(history,new)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_runtime_and_training_capture_same_current_state(self):
        task=self.task;policy=FullbodyPolicy(task.robot).eval()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.pt';torch.save(dict(config=self.cfg,task_config=task.metadata(),robot_metadata=task.robot.metadata(),pi=policy.state_dict()),path)
            with patch('supermimic.acp.motion.MotionLib.__init__',side_effect=AssertionError('No future reference')):
                runtime=AimingPolicy(path);runtime.reset(self.S,self.params);runtime.context['support_age_steps'][:]=149
                ctx={k:v.clone() for k,v in runtime.context.items()};previous=runtime.previous_action.clone();history=None
                for step in range(4):
                    state=(self.S[0]+torch.tensor([step*.01,0.,0.]),*self.S[1:])
                    ctx=task.observe_context(state,ctx);obs=task.mdp.obs(state,previous,runtime.xi)
                    history=obs[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                    with torch.no_grad():raw=policy(history,task.command(state,ctx));previous,expected=task.compose_action(raw,runtime.xi)
                    torch.testing.assert_close(runtime.step(state),expected,atol=0,rtol=0)
                    torch.testing.assert_close(runtime.context['feet_anchor_w'],ctx['feet_anchor_w'],atol=0,rtol=0)
                    ctx=task.advance_context(ctx,1)
                runtime.set_target(runtime.context['aim_target_w'],runtime.context['aim_normal_w'])
                self.assertTrue((runtime.context['support_age_steps']==0).all())


if __name__=='__main__':unittest.main()
