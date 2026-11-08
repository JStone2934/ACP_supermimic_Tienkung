"""Meaningful rejection, mapping, provenance and geometry tests for full PHP conversion."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.php_conversion import (SolverConfig, atomic_json, atomic_npz, poses,
                                         project_surface, unique_variants)
from supermimic.acp.robots import get_robot, sha256
from tools.check_php_conversion import assess_variant, collision_evidence
from tools.convert_php import aggregate, verify_task
from tools.retarget_php_pilot import sole


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = get_robot('tienkung3_29dof')
        cls.metadata = cls.robot.metadata()
        cls.kin = RobotKinematics('cpu', cls.robot, dtype=torch.float64)
        cls.centers = np.array([sole(cls.robot, n) for n in ('left_foot', 'right_foot')])

    def fixture(self, folder, *, height=0., slide=False, fast=False):
        robot = self.robot
        motion = np.zeros((8, 36))
        motion[:, 2] = robot.stand_height + height
        motion[:, 3] = 1
        motion[:, 7:] = robot.default_q
        if slide:
            motion[:, 0] = np.arange(8)*.01
        if fast:
            motion[3, 7+robot.joint_names.index('waist_yaw_joint')] += .4
        path = Path(folder)/'test_motion.npz'
        atomic_npz(path, joint_pos=motion, joint_names=robot.joint_names, fps=[50], robot_metadata=json.dumps(self.metadata))
        np.save(Path(folder)/'test_terrain.npy', np.zeros((0, 1, 10)))
        mask = np.zeros((1, 8, 9), bool)
        mask[:, :, :2] = True
        offsets = np.zeros((1, 8, 9, 3))
        offsets[:, :, :2] = self.centers
        normals = np.zeros_like(offsets)
        normals[..., 2] = 1
        return path, dict(contact=mask, offsets=offsets, normals=normals,
                          anchors=np.zeros_like(offsets), surfaces=np.zeros((1, 8, 9), int))

    def test_pose_only_fk_matches_native_mujoco_for_all_bodies(self):
        kin = self.kin
        m, d = kin.model, mujoco.MjData(kin.model)
        rng = np.random.default_rng(7)
        q = np.array(self.robot.default_q) + rng.normal(0, .15, self.robot.num_joints)
        root = np.array([.2, -.1, 1.1])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        P, R = poses(kin, torch.tensor(root)[None], torch.tensor(rotation)[None], torch.tensor(q)[None])
        qa, _ = self.robot.joint_addresses(m)
        d.qpos[:3], d.qpos[3:7], d.qpos[qa] = root, [2**-.5, 0, 0, 2**-.5], q
        mujoco.mj_kinematics(m, d)
        np.testing.assert_allclose(P[0], d.xpos[kin.body_ids], atol=1e-10)
        np.testing.assert_allclose(R[0], d.xmat[kin.body_ids].reshape(-1, 3, 3), atol=1e-10)

    def test_yawed_vertical_face_is_not_projected_to_box_top(self):
        box = np.array([[1., 2., .8, 0., 1., .6, .2, .4]])
        point, normal = project_surface([1., 2.8, .3], [0., 1., 0.], 1, box)
        np.testing.assert_allclose(point, [1., 2.6, .3])
        np.testing.assert_allclose(normal, [0., 1., 0.])
        point, normal = project_surface([3., 5., .02], [0, 0, 1], 0, box)
        np.testing.assert_allclose(point, [3, 5, 0])
        with self.assertRaises(ValueError):
            project_surface([0, 0, 0], [0, 0, 1], 2, box)

    def test_variant_aliases_preserve_all_original_indices(self):
        t = np.zeros((1, 4, 10))
        t[:, 1, 0] = t[:, 3, 0] = .1
        self.assertEqual(unique_variants(t), ([0, 1], [0, 1, 0, 1]))
        self.assertEqual(unique_variants(np.zeros((0, 10, 10))), ([0], [0]*10))

    def test_standing_passes_policy_but_raw_overlap_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            path, intent = self.fixture(folder)
            report = assess_variant(self.robot, path, intent, 0)
        self.assertTrue(report['geometry_pass'], report['rejection_reasons'])
        self.assertFalse(report['strict_raw_geometry_pass'])
        self.assertGreater(report['self_penetration_raw_max_m'], .03)

    def test_terrain_penetration_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, intent = self.fixture(folder, height=-.025)
            report = assess_variant(self.robot, path, intent, 0)
        self.assertFalse(report['geometry_pass'])
        self.assertIn('terrain_penetration', report['rejection_reasons'])

    def test_sliding_and_joint_overspeed_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, intent = self.fixture(folder, slide=True, fast=True)
            report = assess_variant(self.robot, path, intent, 0)
        self.assertIn('left_foot:support_slip', report['rejection_reasons'])
        self.assertIn('joint_velocity', report['rejection_reasons'])

    def test_hand_contact_to_wrong_surface_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, intent = self.fixture(folder)
            intent['contact'][:, :, 4] = True  # hand far above floor cannot be accepted as supporting it
            report = assess_variant(self.robot, path, intent, 0)
        self.assertIn('left_hand:support_distance', report['rejection_reasons'])

    def test_missing_contact_intent_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, intent = self.fixture(folder)
            intent['contact'][:] = False
            report = assess_variant(self.robot, path, intent, 0)
        self.assertIn('no_contact_intent', report['rejection_reasons'])

    def test_stale_artifacts_cannot_resume_as_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'labels.npz'
            p.write_bytes(b'original labels')
            state = dict(labels_path='labels.npz', labels_hash=sha256(p))
            verify_task(folder, state)
            p.write_bytes(b'stale labels')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_task(folder, state)

    def test_failed_tasks_and_empty_accepted_subset_are_explicit(self):
        plan = dict(run_id='fixture', identity='x', robot_metadata=self.metadata,
                    tasks=[dict(id='a', record=dict(variants=10, frames=8))])
        with tempfile.TemporaryDirectory() as folder:
            report = aggregate(folder, plan, {'a': dict(status='failed')})
            self.assertEqual(report['status'], 'completed_with_failures')
            self.assertEqual(report['converted_sources'], 0)
            self.assertEqual(report['accepted_variants'], 0)
            self.assertFalse((Path(folder)/'accepted').exists())

    def test_collision_policy_matches_actual_serial_ankles(self):
        report = collision_evidence()
        self.assertEqual(len(report['chains']), 2)
        self.assertTrue(all(min(r['pair_distance_m'].values()) < -.02 for r in report['representative_poses']))

    def test_invalid_window_settings_rejected(self):
        for kwargs in (dict(iterations=0), dict(window_frames=64, overlap=32), dict(overlap=1)):
            with self.assertRaises(ValueError):
                SolverConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
