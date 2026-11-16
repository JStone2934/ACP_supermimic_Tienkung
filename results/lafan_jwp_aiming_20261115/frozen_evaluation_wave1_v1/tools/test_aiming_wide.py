import unittest
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import get_robot
from supermimic.acp.aiming_targets import fixed_target_grid


class WideTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        robot=get_robot('tienkung3_29dof')
        cls.task=WideAimingTask('cpu',dict(initial_q=list(robot.default_q),initial_height=robot.stand_height))

    def test_all_angles_are_finite_and_have_no_reference_contacts(self):
        p=fixed_target_grid();state=self.task.initial_state(len(p));ctx=self.task.context(state,p)
        before=ctx['aim_target_w'].clone()
        command=self.task.command(state,ctx)
        self.assertEqual(command.shape,(75,107))
        self.assertTrue(command.isfinite().all())
        self.assertTrue((command[:,:27]==0).all())
        moved=(state[0]+torch.tensor([.2,.1,0]),*state[1:])
        self.task.command(moved,ctx)
        torch.testing.assert_close(ctx['aim_target_w'],before,atol=0,rtol=0)

    def test_old_foot_anchors_do_not_penalize_adjustment(self):
        task=self.task;state=task.initial_state(2)
        ctx=task.context(state,torch.tensor([[3.,0.,0.,.05]]).expand(2,-1))
        action=torch.zeros(2,29)
        reward=task.reward(state,action,action,ctx)
        ctx['feet_anchor_w']+=100.
        torch.testing.assert_close(task.reward(state,action,action,ctx),reward,atol=0,rtol=0)
        q=state[4].clone().requires_grad_();state=(*state[:4],q,state[5])
        task.reward(state,action,action,ctx).sum().backward()
        self.assertTrue(q.grad.isfinite().all())


if __name__=='__main__':unittest.main()
