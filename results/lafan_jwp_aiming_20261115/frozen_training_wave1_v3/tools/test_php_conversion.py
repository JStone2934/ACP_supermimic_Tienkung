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
from tools.retarget_php_pilot import link_cloud, sole


class ConversionTests(unittest.TestCase):
    def test_separation_filter_preserves_real_penetration(self):
        from unittest.mock import patch
        from supermimic.acp.make_labels import geom_distance
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <geom type="box" size="1 1 1"/>
            <geom type="box" size="1 1 1" pos="1.5 0 0"/>
            </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        before = data.geom_xpos.copy()
        native = mujoco.mj_geomDistance
        calls = 0
        def first_query_degenerate(m, d, g, tg, limit, witness):
            nonlocal calls
            calls += 1
            if calls == 1:
                witness[:] = [0, 0, 0, 1, 0, 0]
                return 0.
            return native(m, d, g, tg, limit, witness)
        with patch.object(mujoco, 'mj_geomDistance', side_effect=first_query_degenerate):
            distance = geom_distance(model, data, 0, 1, 1., np.zeros(6), strict_reference=True)
        self.assertAlmostEqual(distance, -.5, places=6)
        np.testing.assert_array_equal(data.geom_xpos, before)

    def test_false_penetration_is_rejected_by_separating_plane(self):
        from supermimic.acp.make_labels import build_model, geom_distance
        case = json.loads((Path(__file__).resolve().parents[1] /
                           'docs/php_conversion/evidence/distance_false_penetration.json').read_text())
        model = build_model(np.array(case['boxes']), case['robot'])
        data = mujoco.MjData(model)
        data.qpos[:] = case['qpos']
        mujoco.mj_kinematics(model, data)
        before = data.geom_xpos.copy()
        witness = np.zeros(6)
        distance = geom_distance(model, data, case['g'], case['tg'], case['max_dist'],
                                 witness, strict_reference=True)
        self.assertGreater(distance, .0015)
        self.assertLess(abs(distance-case['expected_distance_m']), 2e-6)
        np.testing.assert_array_equal(data.geom_xpos, before)

    def test_inconsistent_signed_witness_is_excluded_from_retry_consensus(self):
        from supermimic.acp.make_labels import build_model, geom_distance
        case = json.loads((Path(__file__).resolve().parents[1] /
                           'docs/php_conversion/evidence/distance_signed_witness.json').read_text())
        model = build_model(np.array(case['boxes']), case['robot'])
        data = mujoco.MjData(model)
        data.qpos[:] = case['qpos']
        mujoco.mj_kinematics(model, data)
        before = data.geom_xpos.copy()
        witness = np.zeros(6)
        distance = geom_distance(model, data, case['g'], case['tg'], case['max_dist'],
                                 witness, strict_reference=True)
        self.assertGreater(distance, .002)
        self.assertLess(abs(distance-case['expected_distance_m']), 2e-6)
        np.testing.assert_array_equal(data.geom_xpos, before)

    def test_zero_uninitialized_witness_does_not_poison_retry_consensus(self):
        from supermimic.acp.make_labels import build_model, geom_distance
        case = json.loads((Path(__file__).resolve().parents[1] /
                           'docs/php_conversion/evidence/distance_zero_witness.json').read_text())
        model = build_model(np.array(case['boxes']), case['robot'])
        data = mujoco.MjData(model)
        data.qpos[:] = case['qpos']
        mujoco.mj_kinematics(model, data)
        before = data.geom_xpos.copy()
        witness = np.zeros(6)
        distance = geom_distance(model, data, case['g'], case['tg'], case['max_dist'],
                                 witness, strict_reference=True)
        # Expected separation was independently solved between the two convex
        # polytopes, not copied from the native retry being tested.
        self.assertLess(abs(distance-case['expected_distance_m']), 2e-6)
        self.assertGreater(distance, .04)
        self.assertLess(abs(np.linalg.norm(witness[:3]-witness[3:])-distance), 2e-6)
        np.testing.assert_array_equal(data.geom_xpos, before)

        # The adjacent pose returns zero with two zero witnesses on the FIRST
        # query, so the initial guard must reject it as well as retry samples.
        data.geom_xpos[case['g'], 0] += 1e-7
        before = data.geom_xpos.copy()
        distance = geom_distance(model, data, case['g'], case['tg'], case['max_dist'],
                                 witness, strict_reference=True)
        self.assertLess(abs(distance-case['expected_distance_m']), 2e-6)
        np.testing.assert_array_equal(data.geom_xpos, before)

    def test_native_retry_agreement_across_scales_preserves_geometry(self):
        from supermimic.acp.make_labels import build_model, geom_distance
        case = json.loads((Path(__file__).resolve().parents[1] /
                           'docs/php_conversion/evidence/distance_cross_scale.json').read_text())
        model = build_model(np.array(case['boxes']), case['robot'])
        data = mujoco.MjData(model)
        data.qpos[:] = case['qpos']
        mujoco.mj_kinematics(model, data)
        before = data.geom_xpos.copy()
        witness = np.zeros(6)
        distance = geom_distance(model, data, case['g'], case['tg'], case['max_dist'],
                                 witness, strict_reference=True)
        self.assertLess(abs(distance-case['expected_distance_m']), case['expected_tolerance_m'])
        self.assertLess(abs(np.linalg.norm(witness[:3]-witness[3:])-distance), 2e-6)
        self.assertGreater(distance, .08)  # erroneous zero would invent contact
        np.testing.assert_array_equal(data.geom_xpos, before)

    def test_native_near_touch_retry_preserves_geometry(self):
        from supermimic.acp.make_labels import geom_distance
        from tools.check_php_pilot import scene
        case = json.loads((Path(__file__).resolve().parents[1] /
                           'docs/php_conversion/evidence/distance_near_touch.json').read_text())
        model = scene(self.robot, np.array(case['boxes']), self_collision=True)
        data = mujoco.MjData(model)
        data.qpos[:] = case['qpos']
        mujoco.mj_forward(model, data)
        before = data.geom_xpos.copy()
        witness = np.zeros(6)
        distance = geom_distance(model, data, case['g'], case['tg'], case['max_dist'],
                                 witness, strict_reference=True)
        # Mesh bottom is within one micron of the box top. The old retry
        # rejected every consistent zero and raised on this real full-run pose.
        mesh = int(model.geom_dataid[case['g']])
        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[a:a+n] @ data.geom_xmat[case['g']].reshape(3, 3).T + data.geom_xpos[case['g']]
        separation = vertices[:, 2].min() - (data.geom_xpos[case['tg'], 2] + model.geom_size[case['tg'], 2])
        self.assertLess(abs(distance-separation), 2e-6)
        self.assertLess(np.linalg.norm(witness[:3]-witness[3:]), 2e-6)
        np.testing.assert_array_equal(data.geom_xpos, before)

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

    def test_native_witness_gradient_separates_ground_and_preserves_fixed_frames(self):
        from supermimic.acp.php_conversion import FullBodyRetargeter, STRUCTURAL_PAIRS
        solver = FullBodyRetargeter('cpu')
        motion = np.zeros((8, 36), np.float32)
        motion[:, 2], motion[:, 3], motion[:, 7:] = self.robot.stand_height-.025, 1, self.robot.default_q
        times, owners, local, normals = solver.collision_witnesses(motion, np.zeros((0, 8)), first=3)
        self.assertGreaterEqual(int(times.min()), 3)
        root = torch.tensor(motion[:, :3], requires_grad=True)
        P, R = poses(solver.tk, root, torch.eye(3).repeat(8, 1, 1), torch.tensor(motion[:, 7:]))
        world = P[times[:, None], owners.clamp(min=0)] + (R[times[:, None], owners.clamp(min=0)] @ local[..., None]).squeeze(-1)
        world = torch.where((owners < 0)[..., None], local, world)
        gap = ((world[:, 1]-world[:, 0])*normals).sum(-1)
        ground = (owners < 0).any(-1)
        self.assertLess(float(gap[ground].detach().min()), -.02)
        torch.relu(-gap[ground]).square().mean().backward()
        self.assertLess(float(root.grad[3:, 2].sum()), 0)  # descent raises the robot
        for pair in owners.tolist():
            names = frozenset(solver.tk.body_names[b] for b in pair if b >= 0)
            self.assertNotIn(names, STRUCTURAL_PAIRS)
        lifted, info = solver.terrain_clearance(motion, np.zeros((0, 8)))
        np.testing.assert_array_equal(lifted[:, :2], motion[:, :2])
        np.testing.assert_array_equal(lifted[:, 3:], motion[:, 3:])
        self.assertLessEqual(info['max_lift_m'], solver.config.clearance_max_m)
        self.assertGreater(info['max_lift_m'], 0)
        # A deep violation cannot be hidden by an unbounded root shift.
        self.assertLess(float(lifted[0, 2]), self.robot.stand_height-.01)

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

    def test_real_hand_on_vertical_box_face_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            path, intent = self.fixture(folder)
            with np.load(path) as original:
                fields = dict(original)
            fields['joint_pos'][:, 7+self.robot.joint_names.index('shoulder_pitch_l_joint')] = -.9
            atomic_npz(path, **fields)
            m, d = self.kin.model, mujoco.MjData(self.kin.model)
            qa, _ = self.robot.joint_addresses(m)
            d.qpos[:7], d.qpos[qa] = fields['joint_pos'][0, :7], fields['joint_pos'][0, 7:]
            mujoco.mj_kinematics(m, d)
            bid = m.body(self.robot.contact_bodies['left_hand'][0]).id
            local = link_cloud(self.robot, 'left_hand')
            world = d.xpos[bid] + local @ d.xmat[bid].reshape(3, 3).T
            index = int(world[:, 0].argmax())
            point = world[index]
            top = float(point[2]+.05)
            raw = np.array([[[point[0]+.06, point[1], top/2, 1, 0, 0, 0, .12, .08, top]]])
            np.save(Path(folder)/'test_terrain.npy', raw)
            intent['contact'][:, :, 4] = True
            intent['surfaces'][:, :, 4] = 1
            intent['offsets'][:, :, 4] = local[index]
            intent['anchors'][:, :, 4] = point
            intent['normals'][:, :, 4] = [-1, 0, 0]
            report = assess_variant(self.robot, path, intent, 0)
        self.assertTrue(report['geometry_pass'], report['rejection_reasons'])
        self.assertLess(report['contacts']['left_hand']['distance_p95_m'], 1e-5)

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

    def test_refinement_preserves_intent_and_requires_new_quality(self):
        from supermimic.acp.refine_motion import write_clip
        with tempfile.TemporaryDirectory() as folder:
            source, intent = self.fixture(folder)
            original_intent = Path(folder)/'test_intent.npz'
            atomic_npz(original_intent, **intent)
            with np.load(source) as z:
                motion = z['joint_pos'].copy()
            motion[:, 2] += .001
            dest = Path(folder)/'refined'/'test_motion.npz'
            lineage = write_clip(self.kin.model, str(source), str(dest), motion[:, :3], motion[:, 3:7], motion[:, 7:], self.robot)
            self.assertEqual(lineage['parent_intent_hash'], lineage['intent_hash'])
            self.assertEqual(lineage['quality_status'], 'requires_new_quality_and_labels')
            self.assertEqual(lineage['parent_motion_hash'], sha256(source))
            self.assertNotEqual(lineage['motion_hash'], lineage['parent_motion_hash'])
            with self.assertRaises(ValueError):
                write_clip(self.kin.model, str(source), str(dest), motion[:, :3], motion[:, 3:7], motion[:, 7:], self.robot)

    def test_failed_tasks_and_empty_accepted_subset_are_explicit(self):
        plan = dict(run_id='fixture', identity='x', robot_metadata=self.metadata,
                    tasks=[dict(id='a', record=dict(variants=10, frames=8))])
        with tempfile.TemporaryDirectory() as folder:
            report = aggregate(folder, plan, {'a': dict(status='failed')})
            self.assertEqual(report['status'], 'completed_with_failures')
            self.assertEqual(report['converted_sources'], 0)
            self.assertEqual(report['accepted_variants'], 0)
            self.assertFalse((Path(folder)/'accepted').exists())

    def test_killed_worker_records_interruption_without_claiming_terminal_tasks(self):
        from queue import Queue
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from tools.convert_php import execute
        process = MagicMock(pid=1234, exitcode=-9)
        process.is_alive.return_value = False
        context = SimpleNamespace(Queue=Queue, Process=lambda **kwargs: process)
        plan = dict(run_id='interrupted-fixture', identity='x', robot_metadata=self.metadata,
                    solver={}, tasks=[dict(id='a', record=dict(variants=10, frames=8))])
        args = SimpleNamespace(retry_failed=False, devices='cpu', workers_per_gpu=1, cpu_workers=1)
        with tempfile.TemporaryDirectory() as folder:
            with patch('tools.convert_php.mp.get_context', return_value=context), \
                 patch('tools.convert_php.ProcessPoolExecutor', return_value=MagicMock()):
                with self.assertRaisesRegex(RuntimeError, 'exited -9'):
                    execute(Path(folder), plan, args)
            report = json.loads((Path(folder)/'manifest.json').read_text())
            self.assertEqual(report['status'], 'interrupted')
            self.assertFalse(report['all_terminal'])
            self.assertEqual(report['counts']['pending'], 1)

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
