import unittest
import torch
from supermimic.acp.aiming_fullbody_env import perturb_initial_state


class ResetNoiseTests(unittest.TestCase):
    def test_reproducible_distinct_worlds_bounded_and_valid_rotations(self):
        n=100;p=torch.zeros(n,3);p[:,2]=1.
        state=(p,torch.eye(3).expand(n,3,3).clone(),torch.zeros_like(p),torch.zeros_like(p),
               torch.zeros(n,29),torch.zeros(n,29))
        cfg=dict(reset_position_noise_m=.01,reset_rotation_noise_rad=.015,reset_joint_noise_rad=.03)
        def sample(seed):
            torch.manual_seed(seed)
            return perturb_initial_state(state,cfg,torch.full((29,),-.02),torch.full((29,),.02))
        a,b,c=sample(12),sample(12),sample(13)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
        self.assertFalse(torch.equal(a[4],c[4]));self.assertEqual(len(torch.unique(a[4],dim=0)),n)
        self.assertLessEqual(float(a[0][:,:2].abs().max()),.01)
        self.assertLessEqual(float(a[4].abs().max()),.020001)
        torch.testing.assert_close(a[1]@a[1].transpose(-1,-2),torch.eye(3).expand(n,3,3),atol=1e-6,rtol=1e-6)
        for index in (2,3,5):torch.testing.assert_close(a[index],state[index],atol=0,rtol=0)
        unchanged=perturb_initial_state(state,{},torch.full((29,),-1),torch.full((29,),1))
        for x,y in zip(state,unchanged):self.assertIs(x,y)


if __name__=='__main__':unittest.main()
