import unittest
import torch
from supermimic.acp.collision_objective import pair_collision_cost

class CollisionPairTests(unittest.TestCase):
    def test_both_independent_collisions_receive_gradient(self):
        gaps=torch.tensor([-.004,-.007],dtype=torch.float64,requires_grad=True)
        pair_collision_cost(gaps,torch.tensor([0,1]),2,10).backward()
        self.assertTrue(bool((gaps.grad<0).all()))
        self.assertTrue(bool(torch.isfinite(gaps.grad).all()))

    def test_repeated_historical_planes_do_not_change_penalty(self):
        gaps=torch.tensor([-.003,-.006,.002],dtype=torch.float64)
        a=pair_collision_cost(gaps,torch.tensor([0,0,1]),2,100)
        b=pair_collision_cost(gaps.repeat(5),torch.tensor([0,0,1]).repeat(5),2,100)
        self.assertEqual(float(a),float(b))
        self.assertAlmostEqual(float(a),.0062**2/100)

    def test_clear_pairs_have_no_force(self):
        gaps=torch.tensor([.001,.01],dtype=torch.float64,requires_grad=True)
        value=pair_collision_cost(gaps,torch.tensor([0,1]),2,10)
        self.assertEqual(float(value),0.)
        value.backward();self.assertTrue(bool((gaps.grad==0).all()))

if __name__=='__main__':unittest.main()
