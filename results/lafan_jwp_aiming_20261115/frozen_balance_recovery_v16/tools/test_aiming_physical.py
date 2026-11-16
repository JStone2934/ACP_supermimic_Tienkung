"""Contracts for physical-MC training; GPU integration is a separate finite run."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
import time
from unittest.mock import Mock, patch

import numpy as np
import torch

from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy
from supermimic.acp.aiming_physical import (physical_returns, policy_digest, snapshot_actor, restore_actor,
                                         initialize_standing_policy, compare_candidate, fit_critic, cpu_tree, PhysicalWorker,
                                         RolloutStore, actor_batches, check_baseline, to_device)
from supermimic.acp.aiming_physical import execution_setup, execution_precision
from supermimic.acp.networks import Critic
from supermimic.acp.train_aiming_physical import validate_config, apply_precision_override, official_due, DEFAULT


def metrics(**overrides):
    values = dict(root_max_m=.02, feet_max_m=.01, tilt_max_deg=2., survival_s=30.,
                  return_=10., undiscounted_return=100., fell=0., standing_pass=1.,
                  hit_rate=.5, angle_deg=5.)
    values.update(overrides)
    return dict(values, finite=True, complete=True,
                per_environment={k: [v] * 8 for k, v in values.items()})


class PhysicalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_terminal_reward_and_reset_isolation(self):
        r = torch.tensor([[1., 10.], [2., 20.], [1000., 30.]])
        d = torch.tensor([[False, False], [True, False], [True, True]])
        g = physical_returns(r, d, .5)
        torch.testing.assert_close(g[:, 0], torch.tensor([2., 2., 1000.]))
        torch.testing.assert_close(g[:, 1], torch.tensor([27.5, 35., 30.]))

    def test_digest_includes_normalization(self):
        state = {'weight': torch.zeros(3), 'norm.mean': torch.zeros(1)}
        initial = policy_digest(state)
        copied = cpu_tree(state)
        state['norm.mean'].add_(1)
        self.assertNotEqual(initial, policy_digest(state))
        self.assertEqual(initial, policy_digest(copied))

    def test_random_initialization_has_reset_pose_bias(self):
        task = FullbodyTask('cpu')
        torch.manual_seed(10)
        pi = FullbodyPolicy(freeze_norm=False)
        initialize_standing_policy(pi, task)
        xi = torch.zeros(1, task.robot.xi_dim); xi[:, 0] = 1
        _, target = task.compose_action(pi.dec[-1].bias[None], xi)
        torch.testing.assert_close(target[0], task.initial_q)
        self.assertGreater(pi.obs_norm.max_count, 0)
        self.assertGreater(int(torch.count_nonzero(pi.dec[-1].weight)), 0)
        torch.manual_seed(10)
        second = FullbodyPolicy(freeze_norm=False)
        initialize_standing_policy(second, task)
        self.assertEqual(policy_digest(pi.state_dict()), policy_digest(second.state_dict()))

    def test_exact_parameter_and_adam_rollback(self):
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
        x = torch.randn(7, 3)
        model(x).square().mean().backward(); optimizer.step(); optimizer.zero_grad()
        saved = snapshot_actor(model, optimizer)
        model(x).square().mean().backward(); optimizer.step()
        restore_actor(model, optimizer, saved)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, saved[0][k]))
        after = optimizer.state_dict()
        self.assertEqual(after['param_groups'], saved[1]['param_groups'])
        for i, item in saved[1]['state'].items():
            for k, v in item.items():
                self.assertTrue(torch.equal(after['state'][i][k], v))

    def test_survival_improvement_can_increase_drift(self):
        base = metrics(survival_s=1., fell=1., standing_pass=0., feet_max_m=.02)
        better = metrics(survival_s=5., fell=1., standing_pass=0., feet_max_m=.10)
        self.assertTrue(compare_candidate(base, better)['accepted'])
        self.assertFalse(compare_candidate(better, base)['accepted'])

    def test_standing_protection_and_best_anchor(self):
        base = metrics()
        bad = metrics(feet_max_m=.02)
        self.assertFalse(compare_candidate(base, bad, protected=True)['accepted'])
        current = metrics(feet_max_m=.015)
        tiny = metrics(feet_max_m=.016)
        self.assertTrue(compare_candidate(current, tiny, protected=True)['accepted'])
        self.assertFalse(compare_candidate(current, tiny, protected=True, anchor=base)['accepted'])

    def test_incomplete_and_nonfinite_never_accepted(self):
        candidate = metrics(); candidate['complete'] = False
        self.assertFalse(compare_candidate(metrics(), candidate)['accepted'])
        candidate['complete'] = True; candidate['finite'] = False
        self.assertFalse(compare_candidate(metrics(), candidate)['accepted'])

    def test_fixed_h8_and_separate_gpu_validation(self):
        cfg = json.loads(Path(DEFAULT).read_text())
        validate_config(cfg)
        validate_config(dict(cfg, horizon=16))
        for changes in [dict(horizon=4), dict(hist=8), dict(sampler_device=cfg['learner_device']),
                        dict(critic_train_envs=cfg['critic_envs'])]:
            with self.assertRaises(ValueError):
                validate_config(dict(cfg, **changes))

    def test_reward_migration_preserves_dynamics_and_resets_old_objective(self):
        from tools.migrate_aiming_physical_checkpoint import reconfigure_checkpoint
        cfg = json.loads(Path(DEFAULT).read_text())
        cfg['reward_weights'].update(aim=2., feet=6.)
        cfg.pop('feet_position_scale_m', None)
        task = FullbodyTask('cpu', cfg)
        meta = task.metadata(); meta.pop('feet_position_scale_m')
        opt = dict(state={0: {'exp_avg': torch.ones(2)}}, param_groups=[dict(lr=1e-7)])
        original = dict(training_backend='physical_mc_dual_v1', stage='training', smoke=False,
                        config=cfg, task_config=meta, robot_metadata=task.robot.metadata(),
                        pi={'weight': torch.ones(2)}, wm={'weight': torch.ones(3)},
                        critic={'old': torch.ones(1)}, optimizers=[copy.deepcopy(opt) for _ in range(3)],
                        physical_state=dict(completed=False, critic_ready=True, accepted_blocks=22,
                                            actor_updates=110, wm_updates=45000, anchor={'old': 1}),
                        iteration=4500, curriculum=dict(level=0, consecutive=1, mastered=[]),
                        exploration=dict(best_score=1., stale_evaluations=3))
        target = json.loads(Path('docs/config/standing_aiming_fullbody_v2/h16_physical_dual_aggressive.json').read_text())
        requested = dict(cfg, **{k:target[k] for k in ('horizon','critic_envs','critic_train_envs',
            'validation_horizons','reward_weights','feet_position_scale_m')})
        changed, record = reconfigure_checkpoint(original, requested)
        self.assertTrue(record['reward_changed'])
        self.assertEqual(changed['config']['horizon'],16)
        self.assertEqual(changed['config']['critic_envs'],4096)
        self.assertEqual(changed['config']['critic_train_envs'],3072)
        self.assertEqual(changed['task_config']['feet_position_scale_m'],.1)
        self.assertEqual(changed['optimizers'][0]['param_groups'][0]['lr'],1e-7)
        self.assertFalse(changed['optimizers'][0]['state'])
        self.assertFalse(changed['optimizers'][2]['state'])
        torch.testing.assert_close(changed['optimizers'][1]['state'][0]['exp_avg'],torch.ones(2))
        torch.testing.assert_close(changed['pi']['weight'],original['pi']['weight'])
        torch.testing.assert_close(changed['wm']['weight'],original['wm']['weight'])
        self.assertEqual(changed['physical_state']['actor_updates'],110)
        self.assertFalse(changed['physical_state']['critic_ready'])
        self.assertTrue(changed['physical_state']['official_pending'])
        self.assertIsNone(changed['physical_state']['anchor'])
        self.assertTrue(original['optimizers'][0]['state'])
        self.assertTrue(original['physical_state']['critic_ready'])
        with self.assertRaisesRegex(ValueError,'Unsupported configuration'):
            reconfigure_checkpoint(original,dict(requested,gamma=.995))

    def test_critic_training_and_validation_are_separate(self):
        torch.manual_seed(5)
        critic = Critic(3)
        op = torch.optim.Adam(critic.parameters(), lr=1e-3)
        data = dict(x=torch.cat((torch.zeros(16, 3), torch.ones(8, 3) * 20)),
                    y=torch.ones(24), t=torch.zeros(24), env_id=torch.tensor([0] * 16 + [1] * 8))
        cfg = dict(critic_train_envs=1, critic_initial_updates=20, critic_refresh_updates=10,
                   critic_batch=8, critic_check_every=5, critic_patience=2,
                   gamma=.99, bound_value_bootstrap=True, max_step_reward=.3)
        result = fit_critic(critic, op, data, cfg, first=True)
        self.assertEqual(result['train_samples'], 16)
        self.assertEqual(result['validation_samples'], 8)
        torch.testing.assert_close(critic.norm.mean, torch.zeros(3))
        self.assertEqual(critic.norm.max_count, 0)
        self.assertLessEqual(result['best_mae'], result['before_mae'])
        self.assertTrue(np.isfinite(result['best_mae']))

    def test_worker_timeout_and_exit_are_visible(self):
        worker = PhysicalWorker.__new__(PhysicalWorker)
        worker.connection = Mock(); worker.connection.poll.return_value = False
        worker.process = Mock(); worker.process.is_alive.return_value = True
        worker.cfg = {'worker_timeout_s': 1}
        worker.pending = {'began': time.monotonic() - 2}
        with self.assertRaises(TimeoutError): worker.ready()
        worker.process.is_alive.return_value = False
        with self.assertRaises(RuntimeError): worker.ready()

    def test_worker_rejects_stale_policy_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = PhysicalWorker.__new__(PhysicalWorker)
            worker.connection = Mock(); worker.connection.poll.return_value = True
            worker.connection.recv.return_value = dict(job_id=1, ok=True)
            worker.pending = dict(job_id=1, directory=tmp, version=3, digest='policy', level=0, began=time.monotonic())
            torch.save(dict(job_id=1, version=2, digest='policy', level=0), Path(tmp) / 'result.pt')
            with self.assertRaisesRegex(ValueError, 'Stale physical result: version'):
                worker.receive()
            self.assertIsNotNone(worker.pending)

    def test_rollout_storage_reuses_buffers_without_stale_rows(self):
        store = RolloutStore(3)
        value = dict(x=torch.tensor([[1.], [2.]]), ctx=dict(a=torch.ones(2, 3)),
                     valid=torch.tensor([True, False]))
        store.append(value)
        value['x'].add_(10)
        store.append(value)
        view = store.view()
        torch.testing.assert_close(view['x'][:, 0], torch.tensor([1., 2., 11., 12.]))
        pointer = store.buffers['x'].data_ptr()
        store.reset(); store.append(value)
        self.assertEqual(store.buffers['x'].data_ptr(), pointer)
        self.assertEqual(len(store.view()['x']), 2)
        self.assertEqual(store.view()['valid'].tolist(), [True, False])

    def test_actor_payload_contains_only_valid_training_starts(self):
        marker = torch.arange(6.)[:, None]
        data = {k: marker.clone() for k in ('s', 'history', 'ap', 'xi', 't')}
        data.update(ctx=dict(marker=marker), env_id=torch.tensor([0, 0, 1, 1, 2, 2]),
                    valid=torch.tensor([False, True, True, False, True, True]), x=marker)
        batches = actor_batches(data, dict(critic_train_envs=2, pi_batch=64), 3)
        self.assertEqual(len(batches), 3)
        for b in batches:
            self.assertEqual(len(b['s']), 64)
            self.assertTrue(((b['s'] == 1) | (b['s'] == 2)).all())
            self.assertNotIn('x', b)
            self.assertNotIn('env_id', b)
            torch.testing.assert_close(b['ctx']['marker'], b['s'])
        moved = to_device(batches, 'cpu')
        self.assertIsInstance(moved, list)
        self.assertEqual(len(moved), 3)

    def test_overlapped_baseline_requires_identical_identity_and_seed(self):
        base = dict(job_id=3, digest='old', version=2, level=1, seed=99)
        payload = dict(baseline_job_id=3, baseline_digest='old', version=2, level=1, seed=99)
        check_baseline(base, payload)
        for key, value in [('baseline_job_id', 4), ('baseline_digest', 'other'), ('version', 3), ('level', 2), ('seed', 100)]:
            with self.assertRaisesRegex(ValueError, 'Stale paired baseline'):
                check_baseline(base, dict(payload, **{key: value}))
        with self.assertRaises(ValueError): check_baseline(None, payload)

    def test_invalid_rollout_rows_do_not_train_or_validate_critic(self):
        critic = Critic(3)
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        data = dict(x=torch.tensor([[0., 0., 0.], [999., 999., 999.], [0., 0., 0.], [999., 999., 999.]]),
                    y=torch.tensor([1., float('nan'), 1., float('nan')]), t=torch.zeros(4),
                    env_id=torch.tensor([0, 0, 1, 1]), valid=torch.tensor([True, False, True, False]))
        cfg = dict(critic_train_envs=1, critic_initial_updates=5, critic_refresh_updates=5,
                   critic_batch=8, critic_check_every=5, critic_patience=2,
                   gamma=.99, bound_value_bootstrap=True, max_step_reward=.3)
        result = fit_critic(critic, optimizer, data, cfg, first=True)
        self.assertEqual((result['train_samples'], result['validation_samples']), (1, 1))
        torch.testing.assert_close(critic.norm.mean, torch.zeros(3))
        self.assertTrue(np.isfinite(result['best_mae']))

    def test_aggressive_configuration_preserves_protocol(self):
        cfg = json.loads(Path('docs/config/standing_aiming_fullbody_v2/h8_physical_dual_aggressive.json').read_text())
        validate_config(cfg)
        self.assertEqual((cfg['num_envs'], cfg['critic_envs']), (4096, 1024))
        self.assertEqual((cfg['wm_batch'], cfg['pi_batch'], cfg['critic_batch']), (4096, 2048, 4096))
        self.assertEqual((cfg['iterations'], cfg['actor_budget']), (0, 0))
        self.assertTrue(cfg['critic_on_worker'] and cfg['overlap_baseline'] and cfg['compile_training'])
        self.assertFalse(cfg['actor_recompute'])
        self.assertEqual(cfg['amp'], 'none')
        self.assertFalse(cfg['allow_tf32'])

    def test_fp32_override_preserves_resumed_configuration(self):
        cfg = json.loads(Path(DEFAULT).read_text())
        cfg.update(amp='bf16', allow_tf32=True, actor_lr=1.5e-6)
        expected = dict(cfg, amp='none', allow_tf32=False)
        apply_precision_override(cfg, 'fp32')
        self.assertEqual(cfg, expected)
        apply_precision_override(cfg, None)
        self.assertEqual(cfg, expected)

    def test_resume_precision_routes_to_physical_entrypoint(self):
        from supermimic.acp.train_aiming_fullbody import main
        for precision_args in (['--precision', 'fp32'], ['--precision=fp32']):
            with patch('sys.argv', ['train_aiming_fullbody', '--resume', 'checkpoint.pt', *precision_args]):
                with patch('supermimic.acp.train_aiming_physical.main') as physical_main:
                    main()
                    physical_main.assert_called_once_with()

    def test_standard_fp32_disables_autocast_and_tf32(self):
        from supermimic.acp import networks
        old_amp = networks.AMP_DTYPE
        old_precision = torch.get_float32_matmul_precision()
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            execution_setup(dict(amp='bf16', allow_tf32=True))
            actual = execution_setup(dict(amp='none', allow_tf32=False))
            self.assertIsNone(networks.AMP_DTYPE)
            self.assertEqual(actual, execution_precision())
            self.assertEqual(actual['float32_matmul_precision'], 'highest')
            self.assertFalse(actual['matmul_allow_tf32'])
            self.assertFalse(actual['cudnn_allow_tf32'])
            net = torch.nn.Linear(3, 2)
            self.assertEqual(networks.run_mlp(net, torch.ones(4, 3)).dtype, torch.float32)
        finally:
            networks.AMP_DTYPE = old_amp
            torch.set_float32_matmul_precision(old_precision)
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def test_interrupted_formal_evaluation_is_retried_on_resume(self):
        cfg = dict(formal_eval_blocks=10)
        legacy = dict(initial_evaluated=True, accepted_blocks=20)
        self.assertTrue(official_due(legacy, cfg))
        done = dict(legacy, last_official_accepted_blocks=20, official_pending=False)
        self.assertFalse(official_due(done, cfg))
        self.assertTrue(official_due(dict(done, official_pending=True), cfg))
        self.assertFalse(official_due(dict(done, accepted_blocks=21), cfg))
        self.assertTrue(official_due(dict(done, initial_evaluated=False), cfg))

    def test_service_restart_selects_latest_checkpoint_without_a_budget(self):
        from tools.run_aiming_physical_service import resume_command
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'before_fp32.pt').write_bytes(b'old')
            with self.assertRaises(FileNotFoundError): resume_command(root)
            (root / 'last.pt').write_bytes(b'latest')
            command = resume_command(root, '/python')
            self.assertEqual(command[command.index('--resume') + 1], str(root / 'last.pt'))
            self.assertEqual(command[command.index('--precision') + 1], 'fp32')
            self.assertNotIn('--iterations', command)
            self.assertNotIn('--actor-budget', command)

    def test_host_memory_monitor_records_phase_and_cgroup(self):
        from supermimic.acp.aiming_runtime import MemoryMonitor
        with tempfile.TemporaryDirectory() as directory:
            monitor = MemoryMonitor(directory, lambda: dict(phase='official_evaluation_pending', iteration=1595))
            monitor.sample()
            value = json.loads((Path(directory) / 'memory_latest.json').read_text())
            self.assertEqual(value['iteration'], 1595)
            self.assertEqual(value['phase'], 'official_evaluation_pending')
            self.assertGreater(value['process']['VmRSS'], 0)
            self.assertEqual(len((Path(directory) / 'memory.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
