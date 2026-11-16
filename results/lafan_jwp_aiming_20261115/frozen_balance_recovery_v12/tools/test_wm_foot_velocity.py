"""Cartesian velocity supervision must respect units, joint topology and masks."""
import unittest
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import get_robot
from supermimic.acp.networks import WorldModel
from supermimic.acp.mdp import flatten_state
from supermimic.acp.train_aiming_fullbody import wm_loss


class WorldModelFootVelocityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);r=get_robot('tienkung3_29dof')
        self.task=WideAimingTask('cpu',dict(initial_q=list(r.default_q),initial_height=r.stand_height))
        self.wm=WorldModel(self.task.mdp.scan_dim,hidden=(16,),robot=r)
        with torch.no_grad():self.wm.net[-1].bias[6]=.01/self.wm.out_norm.std()[6]
        state=flatten_state(self.task.initial_state(2))
        self.windows=dict(s=state[:,None],s1=state[:,None].clone(),a=torch.zeros(2,1,29),xi=torch.zeros(2,1,33),mask=torch.ones(2,1))
        self.windows['xi'][:,:,0]=1.

    def loss_grad(self,weight):
        self.task.config['wm_foot_velocity_weight']=weight
        value=wm_loss(self.wm,self.task,self.windows)
        gradient=torch.autograd.grad(value,self.wm.net[-1].bias)[0]
        return value.detach(),gradient

    def test_units_and_velocity_gradient_follow_feet_not_arm_joints(self):
        baseline,g0=self.loss_grad(0.)
        value,g1=self.loss_grad(.05)
        torch.testing.assert_close(value-baseline,torch.tensor(.05*(.01/.05)**2/3),atol=1e-7,rtol=1e-5)
        expected=.05*2*.01/(3*.05**2)*self.wm.out_norm.std()[6]
        torch.testing.assert_close((g1-g0)[6],expected,atol=1e-6,rtol=1e-5)
        arms=self.task.arm_ids+12+29
        torch.testing.assert_close((g1-g0)[arms],torch.zeros_like(g1[arms]),atol=1e-7,rtol=0)
        self.assertGreater(float((g1-g0)[12+29:12+29+12].abs().sum()),1e-5)
        self.assertTrue(torch.isfinite(g1).all())
        del self.task.config['wm_foot_velocity_weight']
        torch.testing.assert_close(wm_loss(self.wm,self.task,self.windows),baseline,atol=0,rtol=0)

    def test_masked_episode_tail_cannot_contribute_velocity_supervision(self):
        self.windows['mask'][1]=0
        before,grad=self.loss_grad(.05)
        self.windows['s1'][1,0,7:10]=100.
        after,changed=self.loss_grad(.05)
        torch.testing.assert_close(after,before,atol=0,rtol=0)
        torch.testing.assert_close(changed,grad,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
