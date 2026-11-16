"""Check live log ingestion and rejected-policy chart semantics."""
import json
from pathlib import Path
import tempfile
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tools.aiming_tensorboard import MetricsFollower, scalars


class TensorBoardContracts(unittest.TestCase):
    def test_backfill_partial_append_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / 'metrics.jsonl'; events = root / 'events'
            (root / 'config.json').write_text(json.dumps(dict(iterations=0, actor_budget=0, horizon=8)))
            first = json.dumps(dict(iteration=25, phase='wm_warmup', elapsed_s=10, last_wm_loss=.5)) + '\n'
            second = json.dumps(dict(iteration=50, phase='wm_warmup', elapsed_s=20, last_wm_loss=.2)) + '\n'
            metrics.write_text(first + second[:15])
            follower = MetricsFollower(metrics, events)
            self.assertEqual(follower.poll(), 1)
            self.assertEqual(follower.poll(), 0)
            with metrics.open('a') as stream: stream.write(second[15:])
            self.assertEqual(follower.poll(), 1)
            follower.close()
            resumed = MetricsFollower(metrics, events)
            self.assertEqual(resumed.poll(), 0)
            resumed.close()
            read = EventAccumulator(str(events)).Reload()
            points = read.Scalars('loss/world_model')
            self.assertEqual([p.step for p in points], [25, 50])
            self.assertAlmostEqual(points[1].value, .2)
            self.assertEqual(read.Scalars('config/unlimited_actor_updates')[0].value, 1.)

    def test_rejected_candidate_is_not_current_policy(self):
        row = dict(phase='candidate_rejected', baseline=dict(feet_max_m=.01, standing_pass=1.),
                   candidate=dict(feet_max_m=.08, standing_pass=0.),
                   actor_batch=[dict(loss=-2., gradient_norm=3.), dict(loss=-4., gradient_norm=5.)])
        result = scalars(row)
        self.assertEqual(result['physical/candidate/feet_drift_cm'], 8.)
        self.assertEqual(result['physical/current_policy/feet_drift_cm'], 1.)
        self.assertEqual(result['physical/current_policy/standing_pass_percent'], 100.)
        self.assertEqual(result['loss/actor_batch_mean'], -3.)
        row['phase'] = 'candidate_accepted'
        self.assertEqual(scalars(row)['physical/current_policy/feet_drift_cm'], 8.)

    def test_unavailable_losses_are_not_fabricated(self):
        result = scalars(dict(phase='wm_warmup', last_wm_loss=float('nan'), actor_updates=0))
        self.assertNotIn('loss/world_model', result)
        self.assertNotIn('loss/actor_batch_mean', result)
        self.assertNotIn('critic/best_mae', result)
        self.assertEqual(result['progress/actor_updates'], 0.)

    def test_precision_marker_requires_tf32_to_be_disabled(self):
        precision = dict(network_amp_dtype=None, float32_matmul_precision='highest',
                         matmul_allow_tf32=False, cudnn_allow_tf32=False)
        self.assertEqual(scalars(dict(precision=precision))['precision/standard_fp32'], 1.)
        precision['matmul_allow_tf32'] = True
        self.assertEqual(scalars(dict(precision=precision))['precision/standard_fp32'], 0.)
        self.assertNotIn('precision/standard_fp32', scalars({}))

    def test_single_writer_and_log_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); metrics = root / 'metrics.jsonl'; events = root / 'events'
            metrics.write_text('{"iteration": 1}\n')
            follower = MetricsFollower(metrics, events)
            try:
                follower.poll()
                with self.assertRaises(BlockingIOError): MetricsFollower(metrics, events)
                metrics.write_text('')
                with self.assertRaisesRegex(ValueError, 'truncated'): follower.poll()
            finally:
                follower.close()


if __name__ == '__main__':
    unittest.main()
