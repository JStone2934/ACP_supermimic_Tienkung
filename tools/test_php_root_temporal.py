"""Physical-time root regularization must preserve offsets and penalize jumps."""
import unittest
import torch
from supermimic.acp.php_conversion import root_correction_cost

class RootTemporalTests(unittest.TestCase):
    def test_constant_morphology_translation_is_unpenalized(self):
        dp=torch.tensor([[.1,-.3,.4]]*100,dtype=torch.float64)
        self.assertEqual(float(root_correction_cost(dp)),0.)

    def test_physical_velocity_is_independent_of_sampling_rate(self):
        # One metre/second correction costs .25 / 3 at either sampling rate.
        for fps in (25.,50.,100.):
            dp=torch.zeros((int(fps)+1,3),dtype=torch.float64)
            dp[:,0]=torch.arange(len(dp),dtype=torch.float64)/fps
            self.assertAlmostEqual(float(root_correction_cost(dp, fps)),.25/3,places=10)

    def test_release_jump_cost_and_gradient_exceed_smooth_transition(self):
        jump=torch.zeros((100,3),dtype=torch.float64);jump[50:,0]=.13
        smooth=jump.clone();smooth[40:61,0]=torch.linspace(0,.13,21,dtype=torch.float64)
        self.assertGreater(float(root_correction_cost(jump)),10*float(root_correction_cost(smooth)))
        jump.requires_grad_();root_correction_cost(jump).backward()
        self.assertTrue(torch.isfinite(jump.grad).all())
        self.assertGreater(float(jump.grad[50,0]),0.)
        self.assertLess(float(jump.grad[49,0]),0.)

if __name__=='__main__':unittest.main()
