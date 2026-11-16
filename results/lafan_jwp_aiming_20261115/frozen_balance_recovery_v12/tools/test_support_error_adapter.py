import unittest
import torch
from supermimic.acp.aiming_adapters import configure_policy_training
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import get_robot
from tools.prepare_support_error_adapter import initialize_error_normalizer


class SupportAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7);r=get_robot('tienkung3_29dof')
        self.cfg=dict(initial_q=list(r.default_q),initial_height=r.stand_height,support_capture_steps=150,
                      support_command_mode='error',policy_training_scope='support_error_inputs',curriculum_level=0,
                      reward_weights=dict(feet=4.))
        self.task=WideAimingTask('cpu',self.cfg);self.S=self.task.initial_state(2)
        self.ctx=self.task.context(self.S,torch.tensor([[3.,0.,0.,.2]]).expand(2,-1),0.)

    def test_error_coordinates_and_capture_release(self):
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]).expand(2,-1,-1)
        state=(self.S[0],rotation,*self.S[2:])
        task=self.task;ctx=task.observe_context(state,task.advance_context(self.ctx,150))
        torch.testing.assert_close(task.command(state,ctx)[:,:6],torch.zeros(2,6),atol=0,rtol=0)
        displacement=torch.tensor([.04,-.03,.02]);drift=(state[0]+displacement,*state[1:])
        expected=torch.tensor([.03,.04,-.02]).expand(2,3)
        torch.testing.assert_close(task.command(drift,ctx)[:,:6].view(2,2,3),expected[:,None].expand(-1,2,-1))
        task.release_support(ctx)
        torch.testing.assert_close(task.command(drift,ctx)[:,:6],torch.zeros(2,6),atol=0,rtol=0)

    def test_optimizer_changes_only_new_columns_and_preserves_zero_feedback_action(self):
        p=FullbodyPolicy(self.task.robot)
        with torch.no_grad():p.enc[0].weight[:,:6].zero_()
        initialize_error_normalizer(p);configure_policy_training(p,self.cfg)
        saved={k:v.clone() for k,v in p.state_dict().items()}
        history=torch.randn(4,10,99);zero=torch.randn(4,107);zero[:,:6]=0
        before=p(history,zero).detach();cmd=zero.clone();cmd[:,:6]=torch.randn(4,6)*.05
        optimizer=torch.optim.Adam(p.parameters(),lr=.001)
        for _ in range(3):
            optimizer.zero_grad();p(history,cmd).square().mean().backward();optimizer.step()
        self.assertGreater(int(torch.count_nonzero(p.enc[0].weight[:,:6])),0)
        for k,v in p.state_dict().items():
            if k=='enc.0.weight':torch.testing.assert_close(v[:,6:],saved[k][:,6:],atol=0,rtol=0)
            else:torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        torch.testing.assert_close(p(history,zero),before,atol=0,rtol=0)

    def test_rejects_incompatible_scope_and_nonzero_source_columns(self):
        p=FullbodyPolicy(self.task.robot)
        with self.assertRaises(ValueError):initialize_error_normalizer(p)
        with self.assertRaises(ValueError):configure_policy_training(p,dict(self.cfg,curriculum_level=1))
        with self.assertRaises(ValueError):configure_policy_training(p,dict(self.cfg,support_command_mode='position'))

    def test_aiming_updates_only_right_arm_output_rows_and_support_inputs(self):
        p=FullbodyPolicy(self.task.robot)
        with torch.no_grad():p.enc[0].weight[:,:6].zero_()
        initialize_error_normalizer(p)
        configure_policy_training(p,dict(self.cfg,curriculum_level=1,policy_training_scope='support_and_right_arm_outputs'))
        saved={k:v.clone() for k,v in p.state_dict().items()}
        history=torch.randn(4,10,99);zero=torch.randn(4,107);zero[:,:6]=0
        before=p(history,zero).detach();cmd=zero.clone();cmd[:,:6]=torch.randn(4,6)*.05
        optimizer=torch.optim.Adam(p.parameters(),lr=.001)
        optimizer.zero_grad();p(history,cmd).square().mean().backward();optimizer.step()
        arm=self.task.arm_ids.tolist();body=[i for i in range(29) if i not in arm]
        for k,v in p.state_dict().items():
            if k=='enc.0.weight':torch.testing.assert_close(v[:,6:],saved[k][:,6:],atol=0,rtol=0)
            elif k in ('dec.8.weight','dec.8.bias'):torch.testing.assert_close(v[body],saved[k][body],atol=0,rtol=0)
            else:torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        after=p(history,zero)
        torch.testing.assert_close(after[:,body],before[:,body],atol=0,rtol=0)
        self.assertGreater(float((after[:,arm]-before[:,arm]).abs().max()),0.)
        self.assertGreater(int(torch.count_nonzero(p.enc[0].weight[:,:6])),0)

    def test_arm_only_retains_support_feedback_and_body_outputs_for_nonzero_errors(self):
        p=FullbodyPolicy(self.task.robot)
        with torch.no_grad():p.goal_norm.mean[:6].zero_()
        configure_policy_training(p,dict(self.cfg,curriculum_level=1,policy_training_scope='right_arm_outputs'))
        saved={k:v.clone() for k,v in p.state_dict().items()}
        history=torch.randn(8,10,99);command=torch.randn(8,107);command[:,:6]*=.05
        before=p(history,command).detach()
        optimizer=torch.optim.Adam(p.parameters(),lr=.001)
        for _ in range(3):
            optimizer.zero_grad();p(history,command).square().mean().backward();optimizer.step()
        arm=self.task.arm_ids.tolist();body=[i for i in range(29) if i not in arm]
        for k,v in p.state_dict().items():
            if k in ('dec.8.weight','dec.8.bias'):torch.testing.assert_close(v[body],saved[k][body],atol=0,rtol=0)
            else:torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        after=p(history,command).detach()
        torch.testing.assert_close(after[:,body],before[:,body],atol=0,rtol=0)
        self.assertGreater(float((after[:,arm]-before[:,arm]).abs().max()),0.)
        with self.assertRaises(ValueError):
            configure_policy_training(p,dict(self.cfg,curriculum_level=0,policy_training_scope='right_arm_outputs'))


if __name__=='__main__':unittest.main()
