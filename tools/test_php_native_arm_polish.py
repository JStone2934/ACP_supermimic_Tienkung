import unittest
import numpy as np
from supermimic.acp.native_pose_polish import polish,violating_arm_phases
from supermimic.acp.retarget_fidelity import native_poses,score_arm_trajectories,ARM_POINTS
from supermimic.acp.robots import get_robot

class NativeArmPolishTests(unittest.TestCase):
    def fixture(self,n=12):
        robot=get_robot('tienkung3_29dof');motion=np.tile(np.r_[[0.,0.,2.],[1.,0.,0.,0.],robot.default_q],(n,1))
        P,R=native_poses('tienkung3_29dof',motion[:,:3],motion[:,3:7],motion[:,7:])
        actual=(R[:,7,None].transpose(0,1,3,2)@(P[:,ARM_POINTS]-P[:,7,None])[...,None]).squeeze(-1)
        return robot,motion,actual

    def test_matching_reference_keeps_every_frame_exact(self):
        robot,motion,actual=self.fixture();out,stats=polish(robot,motion,np.empty((0,8)),np.zeros((len(motion),9),bool),np.zeros((len(motion),9,3)),arm_target=actual)
        np.testing.assert_array_equal(out,motion);self.assertFalse(stats['arm_rows'])

    def test_supporting_arms_are_never_adjusted_to_free_reference(self):
        robot,motion,actual=self.fixture();contact=np.zeros((len(motion),9),bool);contact[:,4:6]=True
        out,stats=polish(robot,motion,np.empty((0,8)),contact,np.zeros((len(motion),9,3)),arm_target=actual+.2)
        np.testing.assert_array_equal(out,motion);self.assertFalse(stats['arm_rows'])

    def test_free_arm_repair_preserves_body_and_velocity_limits(self):
        robot,motion,actual=self.fixture(16);reference=actual.copy();reference[:,:2,0]+=.08
        out,stats=polish(robot,motion,np.empty((0,8)),np.zeros((len(motion),9),bool),np.zeros((len(motion),9,3)),arm_target=reference)
        self.assertTrue(any(r['applied'] for r in stats['arm_rows']))
        np.testing.assert_array_equal(out[:,:22],motion[:,:22])
        np.testing.assert_array_equal(out[:,29:],motion[:,29:])
        self.assertLessEqual(float((np.abs(np.gradient(out[:,7:],axis=0)*50)/robot.velocity).max()),1.)

    def test_amplitude_violation_selects_only_affected_arm_phase(self):
        t=np.linspace(0,2*np.pi,32);reference=np.zeros((32,4,3));reference[:,:2,0]=.06*np.sin(t)[:,None]
        actual=reference.copy();actual[:,:2]*=1.5
        report=score_arm_trajectories(reference,actual,np.ones((32,4),bool))
        self.assertEqual(violating_arm_phases(report),[(0,0,32)])

if __name__=='__main__':unittest.main()
