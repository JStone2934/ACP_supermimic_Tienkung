import unittest
import numpy as np
from supermimic.acp.retarget_fidelity import contact_transition_weights,score_arm_trajectories
from supermimic.acp.source_contacts import source_validation_required

class TransitionTests(unittest.TestCase):
    def test_only_real_contact_boundaries_fade(self):
        free=np.ones((100,4),bool);free[30:50,:2]=False
        weight=contact_transition_weights(free)
        np.testing.assert_array_equal(weight[:20],1)
        np.testing.assert_array_equal(weight[70:],1)
        np.testing.assert_array_equal(weight[:,2:],1)
        np.testing.assert_array_equal(weight[30:50,:2],0)
        self.assertTrue(np.all(np.diff(weight[50:58,0])>0))
        self.assertTrue(np.all(np.diff(weight[22:30,0])<0))
        self.assertTrue(np.all(weight[50:58,0]>0))
        np.testing.assert_array_equal(contact_transition_weights(np.ones((100,4),bool)),1)

    def test_short_phase_retains_core_and_fidelity_still_rejects_frozen_arm(self):
        free=np.zeros((32,4),bool);free[10:18]=True
        weight=contact_transition_weights(free)
        np.testing.assert_array_equal(weight[12:16],1)
        self.assertTrue(np.all(weight[10:18]>0))
        t=np.arange(150)/50
        ref=np.repeat(np.stack((.18*np.sin(2*np.pi*t),.1*np.cos(2*np.pi*t),np.zeros_like(t)),-1)[:,None],4,axis=1)
        self.assertFalse(score_arm_trajectories(ref,np.zeros_like(ref),np.ones((150,4),bool))['passed'])

    def test_future_solver_versions_keep_source_validation(self):
        for version in ('php-segment-v18','php-segment-v19','php-segment-v20'):
            self.assertTrue(source_validation_required(version))
        self.assertFalse(source_validation_required('php-segment-v17'))

if __name__=='__main__':unittest.main()
