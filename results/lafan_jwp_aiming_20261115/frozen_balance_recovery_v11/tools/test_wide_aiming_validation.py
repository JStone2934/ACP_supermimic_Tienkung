import unittest
import numpy as np
from supermimic.acp.wide_aiming_validation import episode_metrics


class WideValidationTests(unittest.TestCase):
    def arrays(self):
        return [np.zeros((1501,3)), np.zeros((1501,2,3)), np.zeros(1501),
                np.full(1501,.01), np.ones(1501,dtype=bool)]

    def test_adjustment_before_capture_is_allowed_but_later_drift_is_not(self):
        root,feet,tilt,error,valid=self.arrays()
        error[:100]=.2
        feet[:100,:,0]=np.linspace(0.,.2,100)[:,None]
        feet[100:,:,0]=.2
        self.assertTrue(episode_metrics(root,feet,tilt,error,valid,False)['passed'])
        feet[500:,:,0]=.3
        self.assertFalse(episode_metrics(root,feet,tilt,error,valid,False)['passed'])
        feet[500:,:,0]=.2;feet[500:,:,2]=.06
        self.assertFalse(episode_metrics(root,feet,tilt,error,valid,False)['passed'])

    def test_each_switch_requires_recapture(self):
        args=self.arrays()
        self.assertTrue(episode_metrics(*args,False,switching=True)['passed'])
        args[3][500:701]=.2
        self.assertFalse(episode_metrics(*args,False,switching=True)['passed'])

    def test_fall_nonfinite_short_run_and_drift_cannot_pass(self):
        args=self.arrays()
        self.assertFalse(episode_metrics(*args,True)['passed'])
        self.assertFalse(episode_metrics(*(a[:100] for a in args),False)['passed'])
        args[4][2]=False
        self.assertFalse(episode_metrics(*args,False)['passed'])
        args=self.arrays();args[0][-1,0]=.51
        self.assertFalse(episode_metrics(*args,False)['passed'])


if __name__=='__main__':unittest.main()
