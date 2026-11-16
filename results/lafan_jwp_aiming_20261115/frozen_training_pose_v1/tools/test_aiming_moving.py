import unittest
import torch
from supermimic.acp.aiming_moving import MovingAimingTask
from supermimic.acp.robots import get_robot
from supermimic.rotations import so3_exp


class MovingTaskTests(unittest.TestCase):
    def test_velocity_command_rotates_but_world_target_does_not(self):
        torch.set_num_threads(2);robot=get_robot('tienkung3_29dof')
        task=MovingAimingTask('cpu',dict(initial_q=list(robot.default_q),initial_height=robot.stand_height,physics_substeps=16))
        state=task.initial_state(1);ctx=task.context(state,torch.tensor([[4.,0.,0.,.05]]))
        ctx['velocity_command'][:,0]=.3;target=ctx['aim_target_w'].clone()
        rotation=so3_exp(torch.tensor([[0.,0.,torch.pi/2]]))
        turned=(state[0],rotation,*state[2:]);command=task.command(turned,ctx)
        torch.testing.assert_close(command[:,-3:],torch.tensor([[0.,-.3,0.]]),atol=1e-6,rtol=0)
        torch.testing.assert_close(ctx['aim_target_w'],target,atol=0,rtol=0)
        moving=(*state[:2],ctx['velocity_command'].clone(),*state[3:]);a=torch.zeros(1,29)
        stationary_score=task.reward_terms(state,a,a,ctx)[0]['velocity']
        moving_score=task.reward_terms(moving,a,a,ctx)[0]['velocity']
        self.assertTrue((moving_score>stationary_score).all())
        self.assertEqual(task.metadata()['physics_substeps'],16)


if __name__=='__main__':unittest.main()
