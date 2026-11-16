"""Posture diagnostics must measure anatomical lean, not model axes or yaw."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.acp.retarget_fidelity import neutral_poses
from supermimic.acp.retarget_fidelity import (POLICY, POSTURE_POLICY, score_posture,
                                            verify_fidelity_report)
from tools.diagnose_php_posture import anatomical_rotations, heading_from_root, lean_deg, statistics


class PostureTests(unittest.TestCase):
    def test_known_forward_lean_survives_heading_change_and_model_calibration(self):
        for robot in ('g1', 'tienkung3_29dof'):
            _, neutral = neutral_poses(robot)
            pitch = Rotation.from_euler('y', 12, degrees=True).as_matrix()
            for yaw in (0, 90, 170, -90):
                turn = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
                raw = (turn @ pitch @ neutral)[None]
                calibrated = anatomical_rotations(robot, raw)
                measured = lean_deg(calibrated, heading_from_root(calibrated))
                np.testing.assert_allclose(measured, 12, atol=1e-10)

    def test_side_lean_does_not_count_as_forward(self):
        side = Rotation.from_euler('x', 15, degrees=True).as_matrix()
        rotations = np.broadcast_to(side, (5, 14, 3, 3))
        np.testing.assert_allclose(lean_deg(rotations, heading_from_root(rotations)), 0, atol=1e-10)

    def test_added_forward_bias_is_separate_from_absolute_source_lean(self):
        source = np.array([2., 4., 20., -5.])
        result = statistics(source, source+8)
        self.assertAlmostEqual(result['added_forward_mean_deg'], 8)
        self.assertAlmostEqual(result['error_abs_p95_deg'], 8)
        self.assertAlmostEqual(result['error_rmse_deg'], 8)

    def test_extra_torso_pitch_roll_or_yaw_rejected(self):
        reference = np.tile(np.eye(3), (100, 2, 1, 1))
        for axis in ('x', 'y', 'z'):
            actual = reference.copy()
            actual[:, 1] = Rotation.from_euler(axis, 8, degrees=True).as_matrix()
            self.assertFalse(score_posture(reference, actual)['passed'])
        self.assertTrue(score_posture(reference, reference.copy())['passed'])

    def test_natural_source_lean_is_preserved_and_short_spike_rejected(self):
        reference = np.tile(Rotation.from_euler('y', 35, degrees=True).as_matrix(), (100, 2, 1, 1))
        self.assertTrue(score_posture(reference, reference.copy())['passed'])
        actual = reference.copy()
        actual[50, 1] = actual[50, 1] @ Rotation.from_euler('y', 15, degrees=True).as_matrix()
        self.assertIn('torso:max', score_posture(reference, actual)['rejection_reasons'])

    def test_pose_failure_cannot_be_exported_as_quality_pass(self):
        report = dict(fidelity_policy=POLICY, posture_policy=POSTURE_POLICY,
                      variants=[dict(geometry_pass=True, fidelity_pass=True,
                                     fidelity=dict(policy=POLICY, passed=True), posture_pass=False,
                                     posture=dict(policy=POSTURE_POLICY, passed=False), quality_pass=True)])
        with self.assertRaisesRegex(ValueError, 'omits required'):
            verify_fidelity_report(report, POLICY, POSTURE_POLICY)
        report['variants'][0]['quality_pass'] = False
        verify_fidelity_report(report, POLICY, POSTURE_POLICY)


if __name__ == '__main__':
    unittest.main()
