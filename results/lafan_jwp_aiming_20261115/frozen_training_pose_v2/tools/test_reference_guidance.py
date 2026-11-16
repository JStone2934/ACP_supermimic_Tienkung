import unittest
import torch
from types import SimpleNamespace
from supermimic.acp.mdp import MDP, RewardCfg
from supermimic.acp.reference_guidance import joint_target_loss


class ReferenceGuidanceTests(unittest.TestCase):
    def test_root_tracking_failure_and_soft_gradient_agree(self):
        reference=torch.zeros(2,3);rotation=torch.eye(3).repeat(2,1,1)
        mdp=SimpleNamespace(cfg=RewardCfg(term_root_distance=.5),motion=SimpleNamespace(root_p=reference,root_R=rotation))
        p=torch.tensor([[.49,0.,0.],[.51,0.,0.]],requires_grad=True);state=(p,rotation)
        frames=torch.arange(2)
        self.assertEqual(MDP.termination(mdp,state,frames).tolist(),[False,True])
        soft=MDP.soft_termination(mdp,state,frames);self.assertLess(float(soft[0]),float(soft[1]))
        soft.sum().backward();self.assertTrue((p.grad[:,0]>0).all())
        mdp.cfg.term_root_distance=None
        self.assertFalse(MDP.termination(mdp,state,frames).any())

    def test_pd_offset_and_action_scale_match_physical_targets(self):
        action=torch.tensor([[1.,-2.]],requires_grad=True)
        xi=torch.tensor([[.8,.1,-.2,0.,0.,0.]])
        default=torch.tensor([.3,.4]);scale=torch.tensor([.2,.5])
        exact=torch.tensor([[.6,-.8]])
        torch.testing.assert_close(joint_target_loss(action,xi,default,scale,exact),torch.tensor(0.),atol=1e-12,rtol=0)
        loss=joint_target_loss(action,xi,default,scale,exact+torch.tensor([[.1,-.2]]));loss.backward()
        torch.testing.assert_close(action.grad,torch.tensor([[-.02,.1]]),atol=1e-6,rtol=0)


if __name__=='__main__':unittest.main()
