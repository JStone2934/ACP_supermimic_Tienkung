import unittest
import torch
from supermimic.acp.posture_transfer import reference_reward,retention_cost


class TransferTests(unittest.TestCase):
    def test_reference_moves_body_toward_target_without_arm_gradient(self):
        q=torch.ones(2,29,requires_grad=True);ref=torch.zeros(29)
        r=reference_reward(q,ref,.35);r.sum().backward()
        self.assertTrue((q.grad[:,:15]<0).all())
        self.assertEqual(int(torch.count_nonzero(q.grad[:,15:])),0)
        self.assertTrue((reference_reward(torch.zeros_like(q),ref,.35)==1).all())

    def test_teacher_is_detached_and_policy_pulled_back(self):
        a=torch.ones(2,29,requires_grad=True);b=torch.zeros(2,29,requires_grad=True)
        loss=retention_cost(a,b,torch.ones(29),2).sum();loss.backward()
        self.assertIsNone(b.grad);self.assertTrue((a.grad>0).all())
        self.assertEqual(float(retention_cost(b,b,torch.ones(29),2).sum().detach()),0)


if __name__=='__main__':unittest.main()
