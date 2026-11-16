import unittest
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import get_robot


class QuietFootRewardTests(unittest.TestCase):
    def test_airborne_motion_is_visible_and_moving_command_disables_quiet_bonus(self):
        torch.set_num_threads(2);r=get_robot('tienkung3_29dof')
        task=WideAimingTask('cpu',dict(initial_q=list(r.default_q),initial_height=r.stand_height,
                                     reward_weights=dict(foot_motion=1.)))
        state=task.initial_state(1);ctx=task.context(state,torch.tensor([[3.,0.,0.,.2]]),0.)
        action=torch.zeros(1,29)
        terms,_=task.reward_terms(state,action,action,ctx)
        self.assertEqual(float(terms['foot_motion']),1.)
        p=state[0]+torch.tensor([0.,0.,1.]);v=torch.tensor([[.2,0.,0.]],requires_grad=True)
        airborne=(p,state[1],v,*state[3:])
        terms,_=task.reward_terms(airborne,action,action,ctx)
        self.assertGreater(float(terms['foot_velocity']),.999)
        self.assertLess(float(terms['foot_motion']),.1)
        self.assertLess(float(torch.autograd.grad(terms['foot_motion'].sum(),v)[0][0,0]),0.)
        moving=dict(ctx,velocity_command=torch.tensor([[.2,0.,0.]]))
        terms,_=task.reward_terms(airborne,action,action,moving)
        self.assertEqual(float(terms['foot_motion']),0.)


if __name__=='__main__':unittest.main()
