import unittest
import numpy as np
from supermimic.acp.moving_aiming_validation import moving_episode_metrics


class MovingMetricsTests(unittest.TestCase):
    def test_motion_stops_aiming_and_physical_limits_are_jointly_required(self):
        n=1501;command=np.zeros((n,3));command[250:1000,0]=.3
        root=np.concatenate((np.zeros((1,3)),np.cumsum(command[:-1]*.02,axis=0)))
        tilt=np.zeros(n);error=np.zeros(n);finite=np.ones(n,bool)
        check=lambda v,e=error,c=command: moving_episode_metrics(root,v,c,root,tilt,e,finite,False)
        self.assertTrue(check(command)['passed'])
        self.assertFalse(check(np.zeros_like(command))['passed'])
        self.assertFalse(check(np.zeros_like(command),c=np.zeros_like(command))['passed'])
        bad=error.copy();bad[600:900]=.051
        self.assertFalse(check(command,bad)['passed'])
        tilt[-1]=15.1;self.assertFalse(check(command)['passed'])


if __name__=='__main__':unittest.main()
