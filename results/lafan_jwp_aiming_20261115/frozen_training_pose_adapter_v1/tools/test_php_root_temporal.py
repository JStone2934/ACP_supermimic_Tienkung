"""Physical-time root regularization must preserve offsets and penalize jumps."""
import unittest
import torch
from supermimic.acp.php_conversion import root_correction_cost,root_excess_cost

class RootTemporalTests(unittest.TestCase):
    def test_constant_morphology_translation_is_unpenalized(self):
        dp=torch.tensor([[.1,-.3,.4]]*100,dtype=torch.float64)
        self.assertEqual(float(root_correction_cost(dp)),0.)

    def test_physical_velocity_is_independent_of_sampling_rate(self):
        # One metre/second adds the same .9m/s tail penalty at every sampling rate.
        for fps in (25.,50.,100.):
            dp=torch.zeros((int(fps)+1,3),dtype=torch.float64)
            dp[:,0]=torch.arange(len(dp),dtype=torch.float64)/fps
            self.assertAlmostEqual(float(root_correction_cost(dp, fps)),.25/3+.5*.1**2,places=10)

    def test_release_jump_cost_and_gradient_exceed_smooth_transition(self):
        jump=torch.zeros((100,3),dtype=torch.float64);jump[50:,0]=.13
        smooth=jump.clone();smooth[40:61,0]=torch.linspace(0,.13,21,dtype=torch.float64)
        self.assertGreater(float(root_correction_cost(jump)),10*float(root_correction_cost(smooth)))
        jump.requires_grad_();root_correction_cost(jump).backward()
        self.assertTrue(torch.isfinite(jump.grad).all())
        self.assertGreater(float(jump.grad[50,0]),0.)
        self.assertLess(float(jump.grad[49,0]),0.)

    def test_peak_guard_is_not_diluted_by_stationary_prefix(self):
        short=torch.zeros((100,3),dtype=torch.float64);short[50:,0]=.13
        long=torch.cat((torch.zeros((900,3),dtype=torch.float64),short),dim=0)
        # The upper-5% average legitimately changes with clip duration.
        # The local 6.5 m/s, 325 m/s² peak contribution must not be diluted.
        peak_floor=.1*(6.5-2.25)**2+.001*(325.-90.)**2
        for dp in (short,long):
            self.assertGreaterEqual(float(root_excess_cost(dp)),peak_floor-1e-9)

    def test_peak_gradient_remains_finite_at_zero_speed_frames(self):
        dp=torch.zeros((100,3),dtype=torch.float64);dp[50:,0]=.13;dp.requires_grad_()
        root_excess_cost(dp).backward()
        self.assertTrue(torch.isfinite(dp.grad).all())
        self.assertGreater(float(dp.grad[50,0]),0.)
        self.assertLess(float(dp.grad[49,0]),0.)

    def test_upper_tail_is_not_averaged_over_stationary_frames(self):
        # Twenty 1.1 m/s corrections in 400 intervals: the entire upper 5%.
        # Acceleration stays below the peak guidance, isolating the tail term.
        speed=torch.zeros(400,dtype=torch.float64);speed[190:210]=1.1
        dp=torch.zeros((401,3),dtype=torch.float64)
        dp[1:,0]=speed.cumsum(0)/50.
        self.assertAlmostEqual(float(root_excess_cost(dp)),.5*.2**2,places=10)

if __name__=='__main__':unittest.main()
