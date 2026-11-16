import math
import unittest
import torch
from supermimic.rotations import so3_exp
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.robots import get_robot


class CompactTerminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);robot=get_robot('tienkung3_29dof')
        cfg=dict(initial_q=list(robot.default_q),initial_height=robot.stand_height,term_tilt_deg=30.,term_height=.25)
        cls.legacy=WideAimingTask('cpu',cfg)
        cls.task=WideAimingTask('cpu',dict(cfg,soft_termination_mode='compact'))

    def context(self,task,n):
        state=task.initial_state(n)
        return state,task.context(state,torch.tensor([[3.,0.,0.,.2]]).expand(n,-1),0.)

    def test_safe_standing_retains_full_imagined_survival(self):
        task=self.task;state,ctx=self.context(task,3)
        angle=torch.tensor([0.,10.,15.])*math.pi/180
        state=(state[0],so3_exp(torch.stack((angle,angle*0,angle*0),-1)),*state[2:])
        done=task.soft_termination(state,torch.zeros(3,dtype=torch.long),ctx)
        torch.testing.assert_close(done,torch.zeros(3),atol=0,rtol=0)
        torch.testing.assert_close((1-done)**128,torch.ones(3),atol=0,rtol=0)
        old=self.legacy.soft_termination(state,torch.zeros(3,dtype=torch.long),ctx)
        self.assertLess(float((1-old[-1])**32),.08)

    def test_compact_transition_gradient_and_actual_limits(self):
        task=self.task;state,ctx=self.context(task,5)
        angle=torch.tensor([27.,29.,30.,31.,33.],requires_grad=True)
        rotation=so3_exp(torch.stack((angle,angle*0,angle*0),-1)*math.pi/180)
        state=(state[0],rotation,*state[2:]);step=torch.zeros(5,dtype=torch.long)
        done=task.soft_termination(state,step,ctx)
        self.assertEqual(float(done[0].detach()),0.);self.assertEqual(float(done[-1].detach()),1.)
        self.assertTrue(bool((done[1:]>done[:-1]).all()))
        gradient=torch.autograd.grad(done.sum(),angle)[0]
        self.assertTrue(bool(gradient.isfinite().all()));self.assertTrue(bool((gradient[1:4]>0).all()))
        torch.testing.assert_close(task.termination(state,step,ctx)[0],self.legacy.termination(state,step,ctx)[0])
        torch.testing.assert_close(task.soft_termination(state,step+1500,ctx),torch.ones(5),atol=0,rtol=0)

    def test_height_and_metadata_identify_new_model_without_changing_old_tasks(self):
        task=self.task;state,ctx=self.context(task,3);state[0][:,2]+=torch.tensor([.1,.25,.28])
        done=task.soft_termination(state,torch.zeros(3,dtype=torch.long),ctx)
        torch.testing.assert_close(done,torch.tensor([0.,.5,1.]),atol=2e-6,rtol=0)
        self.assertNotIn('imagined_termination',self.legacy.metadata())
        checkpoint=dict(task_config=task.metadata(),robot_metadata=task.robot.metadata())
        validate_checkpoint(checkpoint,task)
        with self.assertRaises(ValueError):validate_checkpoint(checkpoint,self.legacy)


if __name__=='__main__':unittest.main()
