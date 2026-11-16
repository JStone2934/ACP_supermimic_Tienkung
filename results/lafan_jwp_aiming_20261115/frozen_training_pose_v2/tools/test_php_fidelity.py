"""Fidelity gates must reject frozen, attenuated and delayed moving arms."""
import unittest
import numpy as np

from supermimic.acp.retarget_fidelity import (
    PARENTS, free_arm_mask, native_poses, neutral_poses, score_arm_trajectories, segment_targets)
from supermimic.acp.robots import get_robot
from supermimic.acp.retarget_fidelity import POLICY, POSTURE_POLICY
from tools.audit_php_variant_coverage import coverage
from tools.plan_php_variant_retries import selections
from supermimic.acp.contact_intent import (HAND_BASIS, corresponding_patch, distal_shank_alias,
                                         is_brief_graze, material_tangent_speed)


class FidelityTests(unittest.TestCase):
    def motion(self):
        t = np.arange(150)/50
        one = np.stack((.18*np.sin(2*np.pi*1.1*t), .08*np.cos(2*np.pi*.7*t),
                        -.2+.1*np.sin(2*np.pi*.9*t+.3)), -1)
        return np.repeat(one[:, None], 4, axis=1), np.ones((len(t), 4), bool)

    def test_exact_trajectory_passes(self):
        ref, free = self.motion()
        self.assertTrue(score_arm_trajectories(ref, ref.copy(), free)['passed'])

    def test_frozen_arms_rejected_even_when_body_can_walk(self):
        ref, free = self.motion()
        got = np.broadcast_to(ref.mean(0), ref.shape)
        result = score_arm_trajectories(ref, got, free)
        self.assertFalse(result['passed'])
        self.assertIn('left_hand:amplitude', result['rejection_reasons'])
        self.assertIn('right_elbow:trajectory', result['rejection_reasons'])

    def test_weak_swing_and_temporal_shift_are_rejected(self):
        ref, free = self.motion()
        low = ref.mean(0)+.25*(ref-ref.mean(0))
        self.assertIn('left_hand:amplitude', score_arm_trajectories(ref, low, free)['rejection_reasons'])
        delayed = np.concatenate((ref[:1].repeat(6, 0), ref[:-6]), 0)
        self.assertIn('right_hand:timing', score_arm_trajectories(ref, delayed, free)['rejection_reasons'])

    def test_hand_support_masks_its_own_arm_only(self):
        contact = np.zeros((30, 9), bool)
        contact[10:20, 4] = True
        mask = free_arm_mask(contact)
        self.assertFalse(mask[10:20, :2].any())
        self.assertTrue(mask[:, 2:].all())

    def test_segment_lengths_and_global_frame_equivariance(self):
        robot = get_robot('g1')
        q = np.array([robot.default_q]*3)
        q[:, 15] += [-.5, 0., .5]
        P, R = native_poses('g1', np.zeros((3, 3)), np.array([[1, 0, 0, 0]]*3), q)
        root = np.array([[0., 0., 1.]]*3)
        target, arm = segment_targets(P, R, root)
        tp, _ = neutral_poses('tienkung3_29dof')
        for child in range(1, 14):
            parent = PARENTS[child]
            np.testing.assert_allclose(np.linalg.norm(target[:, child]-target[:, parent], axis=-1),
                                       np.linalg.norm(tp[child]-tp[parent]), atol=1e-8)
        # Forearm and upper arm have different ratios; neither uses one global scale.
        sp, _ = neutral_poses('g1')
        ratios = [np.linalg.norm(tp[c]-tp[p])/np.linalg.norm(sp[c]-sp[p]) for p, c in ((8, 9), (9, 10))]
        self.assertGreater(abs(ratios[0]-ratios[1]), .1)
        turn = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        shift = np.array([2., 3., .1])
        transformed, same_arm = segment_targets(P @ turn.T+shift, turn @ R, root @ turn.T+shift)
        np.testing.assert_allclose(transformed, target @ turn.T+shift, atol=1e-8)
        np.testing.assert_allclose(same_arm, arm, atol=1e-8)
        self.assertGreater(np.ptp(arm[:, 1, 0]), .1)


class CoverageTests(unittest.TestCase):
    def fixture(self):
        inventory = dict(records=[dict(source='source/category/walk_motion.npz', source_sha256='s',
                         terrain_sha256='t', labels_sha256='l', frames=100, variants=3, variant_aliases=[0, 1, 1])])
        index = dict(fidelity_policy=POLICY, posture_policy=POSTURE_POLICY, source_identities={'category/walk': dict(source_hash='s', terrain_hash='t',
                     labels_hash='l', source_frames=[0, 100], variants=3)},
                     entries=[dict(source_key='category/walk', source_variant=v) for v in range(3)])
        return index, inventory

    def test_complete_original_slots_required(self):
        index, inv = self.fixture()
        self.assertTrue(coverage(index, inv)['all_original_variants_pass'])
        index['entries'].pop()
        result = coverage(index, inv)
        self.assertFalse(result['all_original_variants_pass'])
        self.assertEqual(result['missing_variants'], 1)

    def test_retries_do_not_double_count_and_old_geometry_only_is_rejected(self):
        index, inv = self.fixture()
        index['entries'].append(index['entries'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            coverage(index, inv)
        index['entries'].pop()
        index['fidelity_policy'] = None
        with self.assertRaisesRegex(ValueError, 'fidelity'):
            coverage(index, inv)

    def test_trimmed_source_cannot_fill_full_motion_slot(self):
        index, inv = self.fixture()
        index['source_identities']['category/walk']['source_frames'] = [0, 50]
        with self.assertRaisesRegex(ValueError, 'frames'):
            coverage(index, inv)

    def test_pre_posture_results_cannot_fill_current_coverage(self):
        index, inv = self.fixture()
        index.pop('posture_policy')
        with self.assertRaisesRegex(ValueError, 'posture'):
            coverage(index, inv)

    def test_specialized_retries_deduplicate_geometry_but_keep_full_frames(self):
        index, inv = self.fixture()
        index['entries'] = []
        rows = selections(coverage(index, inv), inv)
        self.assertEqual([r['solve_variant'] for r in rows], [0, 1])
        self.assertTrue(all(r['source_key'] == 'category/walk' and r['end'] == 100 for r in rows))
        self.assertEqual(len({r['key'] for r in rows}), 2)


class SourceContactTests(unittest.TestCase):
    def test_distal_shank_requires_existing_same_surface_foot_support(self):
        knee = np.zeros((5, 3)); ankle = np.tile([0., 0., -.3], (5, 1))
        point = ankle + [.02, 0, 0]
        self.assertTrue(distal_shank_alias(knee, ankle, point, np.ones(5, bool))[0])
        self.assertFalse(distal_shank_alias(knee, ankle, point, np.zeros(5, bool))[0])
        self.assertFalse(distal_shank_alias(knee, ankle, knee+[.02, 0, 0], np.ones(5, bool))[0])

    def test_hand_patch_mapping_preserves_fingers_and_palm_axes(self):
        from itertools import product
        source = np.array(list(product([0., 1.], [-.2, .2], [-.1, .1])))
        target = (source*[.5, 1, .5]) @ HAND_BASIS.T
        patch = corresponding_patch(np.array([1., 0., -.1]), source, target, HAND_BASIS[:, 2], HAND_BASIS)
        np.testing.assert_allclose(patch, HAND_BASIS @ [.5, 0., -.05], atol=1e-8)

    def test_stationary_long_and_hand_support_are_preserved(self):
        self.assertTrue(is_brief_graze('right_knee', .14, .55))
        self.assertFalse(is_brief_graze('right_knee', .14, .02))
        self.assertFalse(is_brief_graze('right_knee', .5, .55))
        self.assertFalse(is_brief_graze('right_hand', .14, .55))
        self.assertFalse(is_brief_graze('right_foot', .14, .55))

    def test_material_slip_is_not_closest_point_motion(self):
        P = np.zeros((8, 1, 3)); P[:, 0, 0] = np.arange(8)*.01
        R = np.broadcast_to(np.eye(3), (8, 1, 3, 3))
        terrain_points = np.zeros_like(P)
        normals = np.zeros_like(P); normals[..., 2] = 1
        # A fixed nearest terrain witness cannot hide a body sliding at .5 m/s.
        speed = material_tangent_speed(P, R, np.zeros((8, 1)), terrain_points, normals, 50)
        np.testing.assert_allclose(speed, .5)
        P[:] = 0
        np.testing.assert_allclose(material_tangent_speed(P, R, np.zeros((8, 1)), terrain_points, normals, 50), 0)


if __name__ == '__main__':
    unittest.main()
