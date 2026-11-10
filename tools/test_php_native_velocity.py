import unittest
import numpy as np
from supermimic.acp.native_pose_polish import velocity_bounds

class NativeVelocityTests(unittest.TestCase):
    def test_sequential_local_changes_preserve_all_measured_velocities(self):
        rng=np.random.default_rng(51);fps=50.;limits=np.array([1.,2.,7.])
        for n in (2,3,4,17):
            q=np.cumsum(rng.uniform(-.3,.3,(n,3))*limits/fps,axis=0)
            for frame in list(range(n))+list(reversed(range(n))):
                lo,hi=velocity_bounds(q,frame,limits,fps)
                self.assertTrue(np.all(lo<=hi))
                for trial in (lo,hi,(lo+hi)/2):
                    candidate=q.copy();candidate[frame]=trial
                    self.assertLessEqual(np.max(np.abs(np.gradient(candidate,axis=0)*fps)/limits),1+1e-12)
                q[frame]=lo+.4*(hi-lo)

    def test_endpoints_are_not_treated_as_central_differences(self):
        q=np.zeros((5,1));limits=np.array([2.])
        lo,hi=velocity_bounds(q,0,limits,50.)
        self.assertAlmostEqual(float(hi[0]),.04)
        candidate=q.copy();candidate[0]=.041
        self.assertGreater(np.abs(np.gradient(candidate,axis=0)*50).max(),2.)
        lo,hi=velocity_bounds(q,2,limits,50.)
        self.assertAlmostEqual(float(hi[0]),.08)

    def test_single_frame_has_no_temporal_constraint(self):
        lo,hi=velocity_bounds(np.zeros((1,2)),0,np.ones(2),50.)
        self.assertTrue(np.isneginf(lo).all() and np.isposinf(hi).all())
        with self.assertRaises(ValueError):velocity_bounds(np.zeros((2,2)),2,np.ones(2),50.)

if __name__=='__main__':unittest.main()
