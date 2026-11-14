"""Contract regressions for fullbody v2; --physics additionally checks real reset/transitions."""
import argparse
import unittest
from unittest.mock import patch

import numpy as np
import torch

from supermimic.acp.aiming_fullbody import *
from supermimic.acp.aiming_fullbody_curriculum import Curriculum, fixed_parameters
from supermimic.acp.buffer import ReplayBuffer
from supermimic.acp.motion import MotionLib
from supermimic.acp.standing import ARM_Q
from supermimic.rotations import so3_exp
from tools.evaluate_aiming_fullbody import metric_report


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.task = FullbodyTask('cpu')

    def setUp(self):
        self.S = self.task.initial_state(3)
        self.ctx = self.task.context(self.S, torch.tensor([[3., 0., 0., .2], [2., .2, -.1, .1], [4., -.2, .1, .05]]))
        self.xi = torch.zeros(3, 33); self.xi[:, 0] = 1

    def test_reference_free(self):
        with patch.object(MotionLib, 'from_dir', side_effect=AssertionError('reference construction')), \
             patch.object(MotionLib, 'ref_state', side_effect=AssertionError('reference read')):
            task = FullbodyTask('cpu')
            task.command(self.S, self.ctx)
            task.reward(self.S, torch.zeros(3, 29), torch.zeros(3, 29), self.ctx)
            task.termination(self.S, torch.zeros(3), self.ctx)
            task.critic_features(self.S, torch.zeros(3, 10, 99), torch.zeros(3, 29), self.xi, torch.zeros(3), self.ctx)
            self.assertIsNone(task.mdp.motion)

    def test_all_action_dimensions_and_offsets(self):
        self.xi[:, 1:30] = .01
        mid = (self.task.mdp.soft_lo + self.task.mdp.soft_hi) / 2
        raw = ((mid - self.task.mdp.default_q - .01) / self.task.alpha).requires_grad_()
        jac = torch.autograd.functional.jacobian(lambda a: self.task.compose_action(a, self.xi[0])[1], raw)
        torch.testing.assert_close(jac, torch.diag(self.task.alpha))
        a, q = self.task.compose_action(raw + 1e4, self.xi[0])
        torch.testing.assert_close(q, self.task.mdp.soft_hi)
        torch.testing.assert_close(self.task.mdp.default_q + .01 + a * self.task.alpha, q)

    def test_world_target_and_command(self):
        before = {k: v.clone() for k, v in self.ctx.items()}
        command = self.task.command(self.S, self.ctx)
        self.assertEqual(command.shape, (3, 107))
        self.assertEqual(int(torch.count_nonzero(command[:, 6:27])), 0)
        torch.testing.assert_close(command[:, 30:93], self.S[0][:, 2:3].expand(-1, 63))
        torch.testing.assert_close(command[:, 102:104], self.ctx['aim_tolerance'] / .2)
        S = list(self.S); S[0] = S[0] + torch.tensor([.2, -.1, 0.])
        S[1] = so3_exp(torch.tensor([[0., 0., .3]])).expand(3, -1, -1) @ S[1]
        changed = self.task.command(S, self.ctx)
        self.assertFalse(torch.allclose(command[:, 93:96], changed[:, 93:96]))
        for k in before: torch.testing.assert_close(before[k], self.ctx[k])

    def test_ray_validity(self):
        h = torch.zeros(4, 3); g = torch.tensor([[3., 0, 0.]]).repeat(4, 1)
        d = torch.tensor([[1., 0, 0], [-1., 0, 0], [0., 1., 0], [1., 0, 0]])
        g[-1, 0] = -3
        e, _, valid, _ = ray_metrics(h, d, g, torch.tensor([[1., 0, 0.]]).repeat(4, 1))
        self.assertEqual(valid.tolist(), [True, False, False, False])
        self.assertTrue(torch.isinf(e[1:]).all())

    def test_reward_gradient_and_timeout(self):
        q = (self.S[4] + torch.randn_like(self.S[4]) * .03).requires_grad_()
        S = (*self.S[:4], q, self.S[5])
        a = torch.zeros(3, 29)
        raw, contributions = self.task.reward_terms(S, a, a, self.ctx)
        torch.testing.assert_close(self.task.reward(S, a, a, self.ctx), sum(contributions.values()))
        grad = torch.autograd.grad(raw['aim'].sum(), q)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad[:, self.task.arm_ids[:4]].abs().sum()), 0)
        term, timeout = self.task.termination(self.S, torch.tensor([1499, 1500, 1501]), self.ctx)
        self.assertEqual(timeout.tolist(), [False, True, True]); self.assertFalse(term.any())
        torch.testing.assert_close(self.task.soft_termination(self.S, torch.tensor([1500] * 3), self.ctx), torch.ones(3))
        invalid = list(self.S); invalid[4] = invalid[4].clone(); invalid[4][0, 0] = float('nan')
        self.assertTrue(self.task.termination(invalid, torch.zeros(3), self.ctx)[0][0])

    def test_replay_context_history_and_terminal_mask(self):
        torch.manual_seed(3)
        buf = FullbodyReplay(12, 1, 'cpu', extra=REPLAY_FIELDS, robot=self.task.robot)
        for i in range(20):
            tr = {k: torch.zeros_like(v[0]) for k, v in buf.data.items()}
            ep = i // 4; step = i % 4
            tr['ep_step'][:] = step; tr['done'][:] = step == 3
            tr['terminated'][:] = ep % 2 == 0 and step == 3
            tr['timeout'][:] = ep % 2 == 1 and step == 3
            tr['o'][:] = ep * 10 + step; tr['aim_target_w'][:] = ep
            buf.add(tr)
        starts, h = buf.sample_starts(100, 10)
        ep = starts['aim_target_w'][:, 0]
        torch.testing.assert_close((h['o'][:, :, 0] / 10).floor(), ep[:, None].expand(-1, 10))
        self.assertTrue(torch.equal(h['o'][:, -1, 0], ep * 10 + starts['ep_step']))
        expected_steps = (starts['ep_step'][:, None] - torch.arange(9, -1, -1)[None]).clamp_min(0)
        torch.testing.assert_close(h['o'][:, :, 0], ep[:, None] * 10 + expected_steps)
        windows = buf.sample_windows(100, 8)
        for done, mask in zip(windows['done'], windows['mask']):
            first = int(done.nonzero()[0])
            self.assertTrue((mask[:first + 1] == 1).all()); self.assertTrue((mask[first + 1:] == 0).all())

    def test_migration_both_known_layouts_and_rejection(self):
        pi = FullbodyPolicy()
        for width in (93, 110):
            state = {k: v.clone() for k, v in pi.state_dict().items()}
            w = torch.arange(width).float()[None].expand(state['enc.0.weight'].shape[0], -1).clone()
            state['enc.0.weight'] = w
            meta = dict(config=dict(name='standing_aiming'), arm_names=list(ARM_NAMES), arm_q=list(ARM_Q)) if width == 93 else dict(
                name='standing_aiming_curriculum', contact_dim=93, fixed_names=list(ARM_NAMES[:4]), axis=[0, 0, -1],
                features=['target_xyz/3', 'direction_xyz', 'plane_xyz', 'velocity_xyz', 'rho/.2', 'sigma/.2', 'previous_terminal/soft_range'])
            ck = dict(pi=state, task_config=meta, robot_metadata=self.task.robot.metadata())
            result, report = migrate_policy(ck, self.task)
            mapped = result.enc[0].weight
            torch.testing.assert_close(mapped[:, :93], w[:, :93])
            self.assertTrue((mapped[:, 104:] == 0).all())
            if width == 110: torch.testing.assert_close(mapped[:, 102:104], w[:, 105:107])
            else: self.assertTrue((mapped[:, 93:] == 0).all())
            ids = self.task.arm_ids[:4 if width == 110 else 7]
            self.assertTrue((result.dec[-1].weight[ids] == 0).all())
            self.assertTrue(result.dec[-1].weight.requires_grad)
            ck['task_config'] = {}
            with self.assertRaises(ValueError): migrate_policy(ck, self.task)

    def test_acceptance_no_warmup_deletion(self):
        x = np.zeros((1501, 8)); x[:, 6:] = 1; x[:, 4] = 1
        x[150:401, 4] = 0  # exactly 3s to 8s: passes
        r = metric_report(x, .05, False, True)
        self.assertTrue(r['passed'])
        x[0, 1] = .051  # Initial drift must not be removed by later windowing.
        self.assertFalse(metric_report(x, .05, False, True)['passed'])
        x[0, 1] = 0; x[150, 4] = 1
        self.assertFalse(metric_report(x, .05, False, True)['passed'])
        x[:, 4] = np.inf; x[:, 6] = 0
        r = metric_report(x, .05, False, True)
        self.assertFalse(r['error_p95_finite']); self.assertEqual(r['valid_ray_fraction'], 0)

    def test_static_curriculum(self):
        self.assertEqual(len(fixed_parameters(8, 'cpu')), 27)
        cur = Curriculum('cpu')
        good = dict(level=0, stage='A', passed=True, scenarios=[dict(standing_pass=True)])
        cur.settle(good); self.assertEqual(cur.level, 0)
        cur.settle(good); self.assertEqual(cur.level, 1)
        p, w = cur.sample(1000)
        self.assertTrue((w == 0).any() and (w == 2).any())

    def test_training_schedule_and_exploration(self):
        from supermimic.acp.train_aiming_fullbody import training_phase, exploration_fraction
        cfg = dict(wm_only_iters=1000, critic_warmup_iters=100, resume_wm_iters=100,
                   noise_decay_updates=10000, noise_min_fraction=.1)
        self.assertEqual(training_phase(999, 0, cfg, 0), 'wm')
        self.assertEqual(training_phase(1000, 0, cfg, 0), 'critic')
        self.assertEqual(training_phase(1099, 0, cfg, 99), 'critic')
        self.assertEqual(training_phase(1100, 0, cfg, 100), 'joint')
        self.assertEqual(training_phase(1500, 1500, cfg, 100, resume=True), 'wm')
        self.assertEqual(training_phase(1600, 1500, cfg, 100, resume=True), 'joint')
        self.assertEqual(training_phase(1100, 1050, cfg, 50, resume=True), 'wm')
        self.assertEqual(training_phase(1150, 1050, cfg, 50, resume=True), 'critic')
        self.assertEqual(exploration_fraction(0, cfg), 1.)
        self.assertEqual(exploration_fraction(5000, cfg), .5)
        self.assertEqual(exploration_fraction(30000, cfg), .1)


def physics_contract(checkpoint_path=None):
    from supermimic.acp.aiming_fullbody_env import FullbodySim
    import warp as wp
    wp.config.quiet = True
    with patch.object(MotionLib, 'from_dir', side_effect=AssertionError('reference construction')), \
         patch.object(MotionLib, 'ref_state', side_effect=AssertionError('reference read')):
        task = FullbodyTask('cuda:0')
        count = 0
        def sampler(n):
            nonlocal count
            count += 1
            return task.alpha.new_tensor([[3., .01 * count, 0, .2]]).repeat(n, 1), 2.
        env = FullbodySim(task, 2, sampler)
        old = {k: x.clone() for k, x in env.context.items()}
        env.ep_step[:] = 1499
        tr = env.step(env.a_prev)
        assert tr['timeout'].all() and tr['done'].all() and (env.ep_step == 0).all()
        for k in old: torch.testing.assert_close(tr[k], old[k])
        assert not torch.allclose(env.context['aim_target_w'], old['aim_target_w'])
        torch.testing.assert_close(tr['r'], task.reward(unflatten_state(tr['s1']), tr['a'], tr['a_prev'], old))
        torch.testing.assert_close(tr['q_des'], task.mdp.default_q + tr['xi'][:, 1:30] + task.alpha * tr['a'])
        if checkpoint_path:
            from supermimic.acp.networks import WorldModel, Critic
            from supermimic.acp.train_aiming_fullbody import update_actor
            ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            pi = FullbodyPolicy().cuda(); pi.load_state_dict(ck['pi'])
            wm = WorldModel(task.mdp.scan_dim, robot=task.robot).cuda(); wm.load_state_dict(ck['wm'])
            critic = Critic(task.critic_dim).cuda()
            buf = FullbodyReplay(12, 2, env.device, REPLAY_FIELDS, robot=task.robot)
            hist = env.observe()[:, None].repeat(1, 10, 1)
            with torch.no_grad():
                for _ in range(12):
                    row = env.step(pi(hist, task.command(env.state(), env.context))); buf.add(row)
                    hist = torch.cat((hist[:, 1:], env.observe()[:, None]), 1)
            cfg = dict(pi_batch=2, horizon=32, gamma=.99, lambda_return=.95, critic_epochs=1, critic_minibatches=1)
            op = torch.optim.Adam(pi.parameters(), lr=1e-5, betas=(.9, .95))
            ov = torch.optim.Adam(critic.parameters(), lr=1e-4)
            pi_before = {k: x.clone() for k, x in pi.state_dict().items()}
            wm_before = {k: x.clone() for k, x in wm.state_dict().items()}
            critic_before = [x.clone() for x in critic.parameters()]
            warm = update_actor(pi, wm, critic, task, buf, op, ov, cfg, train_actor=False)
            assert all(torch.equal(x, pi_before[k]) for k, x in pi.state_dict().items())
            assert all(torch.equal(x, wm_before[k]) for k, x in wm.state_dict().items())
            assert any(not torch.equal(x, old) for x, old in zip(critic.parameters(), critic_before))
            assert all(p.grad is None for p in pi.parameters()) and warm['actor_grad'] == 0
            stats = update_actor(pi, wm, critic, task, buf, op, ov, cfg)
            assert all(np.isfinite(v) for v in stats.values()) and stats['actor_grad'] > 0
            assert all(p.grad is None for p in wm.parameters())
            assert any(not torch.equal(x, pi_before[k]) for k, x in pi.state_dict().items())
            assert all(float(p.grad.abs().max()) <= .100001 for p in pi.parameters() if p.grad is not None)
            print('nonzero aiming reward imagination gradient:', stats)
    print('physics reference-free reset/timeout/reward/action contracts passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--physics', action='store_true')
    parser.add_argument('--checkpoint'); args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful(): raise SystemExit(1)
    if args.physics:
        from supermimic.acp.mdp import unflatten_state
        physics_contract(args.checkpoint)
