import unittest
from types import SimpleNamespace
import torch
from supermimic.acp.reference_support import ReferenceSupportCfg, ReferenceSupportReward


class ReferenceSupportTests(unittest.TestCase):
    def fixture(self):
        # Deliberately different label and FK orders catch implicit foot indexing.
        robot = SimpleNamespace(control_dt=.02, contact_bodies={
            'right_foot':('right_link',), 'torso':('torso_link',), 'left_foot':('left_link',)})
        kin = SimpleNamespace(tracked_names=['torso_link','left_link','right_link'],
                              forward=lambda root,bp,bv:(bp,None,bv,None))
        motion = SimpleNamespace(root_p=torch.zeros(2,3,dtype=torch.double),
                                 body_p=torch.zeros(2,3,3,dtype=torch.double),
                                 body_v=torch.zeros(2,3,3,dtype=torch.double))
        contacts = SimpleNamespace(labels=torch.tensor([[[False,True,True]],[[False,True,False]]]))
        mdp = SimpleNamespace(robot=robot,kin=kin,motion=motion)
        return ReferenceSupportReward(mdp,contacts,ReferenceSupportCfg(1.,2.,1.))

    def test_labels_gate_support_and_leave_root_gradient(self):
        reward = self.fixture()
        root = torch.tensor([[.1,0,0]],dtype=torch.double,requires_grad=True)
        bp = torch.tensor([[[10.,0,0],[.05,0,0],[.9,0,0]]],dtype=torch.double,requires_grad=True)
        bv = torch.tensor([[[10.,0,0],[.1,0,0],[.9,0,0]]],dtype=torch.double,requires_grad=True)
        frames,variants = torch.tensor([0]),torch.tensor([0])
        value = reward((root,bp,bv),frames,variants)
        torch.testing.assert_close(value,torch.tensor([.04],dtype=torch.double))
        value.sum().backward()
        self.assertLess(float(root.grad[0,0]),0)
        self.assertLess(float(bp.grad[0,1,0]),0)
        self.assertLess(float(bv.grad[0,1,0]),0)
        self.assertEqual(float(bp.grad[:,[0,2]].abs().sum()),0)
        self.assertEqual(float(bv.grad[:,[0,2]].abs().sum()),0)
        # No labeled feet: no fabricated support reward, but root remains active.
        torch.testing.assert_close(reward((root,bp,bv),torch.tensor([1]),variants),
                                   torch.tensor([.01],dtype=torch.double))

    def test_upper_bound_and_gradients(self):
        reward = self.fixture()
        frames,variants = torch.tensor([0]),torch.tensor([0])
        zero=(torch.zeros(1,3,dtype=torch.double),torch.zeros(1,3,3,dtype=torch.double),torch.zeros(1,3,3,dtype=torch.double))
        torch.testing.assert_close(reward(zero,frames,variants),torch.tensor([reward.max_step_reward],dtype=torch.double))
        torch.manual_seed(8)
        state=tuple(torch.randn_like(x).mul(.03).requires_grad_() for x in zero)
        self.assertTrue(torch.autograd.gradcheck(lambda *s:reward(s,frames,variants),state))


if __name__=='__main__':
    unittest.main()
