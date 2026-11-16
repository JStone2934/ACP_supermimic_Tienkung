"""Follow fullbody metrics.jsonl without restarting or importing the GPU trainer.

Historical rows are backfilled, complete appended rows are followed, and a
durable byte cursor avoids re-importing history when this observer restarts.
All chart steps are learner iterations; rejected candidates remain separate
from the accepted policy's physical performance.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import math
from pathlib import Path
import signal
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


STAGES = ('A', 'B1', 'B2', 'C1a', 'C1b', 'C2a', 'C2b', 'C3a', 'C3b')
PHASES = ('initial_collection', 'wm_warmup', 'wm_validation', 'official_evaluation',
          'critic_calibration', 'candidate_evaluation', 'candidate_accepted',
          'candidate_rejected', 'model_recovery', 'stopped', 'completed', 'failed', 'official_evaluation_pending')


def scalars(row):
    values = {}
    def put(tag, value, scale=1.):
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[tag] = float(value) * scale
    put('loss/world_model', row.get('last_wm_loss'))
    put('optimizer/actor_learning_rate', row.get('actor_lr'))
    put('runtime/elapsed_seconds', row.get('elapsed_s'))
    if isinstance(row.get('precision'), dict):
        precision = row['precision']
        put('precision/standard_fp32', precision.get('network_amp_dtype') is None
            and precision.get('float32_matmul_precision') == 'highest'
            and precision.get('matmul_allow_tf32') is False and precision.get('cudnn_allow_tf32') is False)
        put('precision/tf32_enabled', precision.get('matmul_allow_tf32'))
    for key in ('actor_updates', 'attempted_actor_updates', 'critic_updates', 'wm_updates',
                'accepted_blocks', 'rejected_blocks', 'consecutive_rejections', 'warmup_done', 'version'):
        put('progress/' + key, row.get(key))
    if row.get('stage') in STAGES:
        put('curriculum/stage_index', STAGES.index(row['stage']))
    if row.get('phase') in PHASES:
        put('runtime/phase_index', PHASES.index(row['phase']))
    for key in ('physical_transitions', 'worker_transitions', 'valid_worker_transitions'):
        put('data/' + key, row.get(key))
    if 'physical_transitions' in row and 'worker_transitions' in row:
        put('data/total_simulated_transitions', row['physical_transitions'] + row['worker_transitions'])
    if row.get('blocks', 0):
        put('policy/accepted_block_fraction', row['accepted_blocks'] / row['blocks'])
    if row.get('phase') == 'critic_calibration':
        fit = row.get('last_calibration') or {}
        for name in ('before_mae', 'best_mae', 'updates', 'train_samples', 'validation_samples'):
            put('critic/' + name, fit.get(name))
        if fit.get('curve'):
            put('loss/critic_last_training', fit['curve'][-1].get('train_loss'))
    if row.get('actor_batch'):
        for field, tag in [('loss', 'loss/actor_batch_mean'), ('gradient_norm', 'actor/gradient_norm_mean')]:
            nums = [x[field] for x in row['actor_batch'] if isinstance(x.get(field), (float, int)) and math.isfinite(x[field])]
            if nums:
                put(tag, sum(nums) / len(nums))
    physical = {
        'root_max_m': ('root_drift_cm', 100.), 'feet_max_m': ('feet_drift_cm', 100.),
        'tilt_max_deg': ('tilt_degrees', 1.), 'survival_s': ('survival_seconds', 1.),
        'return_': ('discounted_return', 1.), 'undiscounted_return': ('episode_return', 1.),
        'fell': ('fall_percent', 100.), 'standing_pass': ('standing_pass_percent', 100.),
        'hit_rate': ('hit_percent', 100.), 'angle_deg': ('aim_angle_degrees', 1.)}
    for group in ('baseline', 'candidate'):
        if isinstance(row.get(group), dict):
            for field, (tag, scale) in physical.items():
                put(f'physical/{group}/{tag}', row[group].get(field), scale)
    if row.get('phase') in ('candidate_accepted', 'candidate_rejected'):
        accepted = row['phase'] == 'candidate_accepted'
        put('policy/candidate_accepted', accepted)
        current = row.get('candidate' if accepted else 'baseline', {})
        for field, (tag, scale) in physical.items():
            put('physical/current_policy/' + tag, current.get(field), scale)
    put('formal_evaluation/passed', row.get('formal_passed'))
    if row.get('phase') == 'wm_validation' and isinstance(row.get('wm_validation'), dict):
        report = row['wm_validation']
        put('wm_validation/passed', report.get('passed'))
        for horizon, result in report.get('horizons', {}).items():
            for key in ('root_m_p95', 'feet_m_p95', 'tcp_m_p95', 'tip_m_p95'):
                put(f'wm_validation/h{horizon}/' + key.replace('_m_', '_cm_'), result.get(key), 100.)
            put(f'wm_validation/h{horizon}/direction_deg_p95', result.get('direction_deg_p95'))
    return values


class MetricsFollower:
    def __init__(self, metrics, out):
        self.metrics, self.out = Path(metrics).resolve(), Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / '.writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.state_path = self.out / 'cursor.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(
            source=str(self.metrics), offset=0, rows=0, malformed_rows=0, identity=None,
            last_scalars={}, config_written=False, origin=None, last_elapsed=0., last_wall_time=0.)
        if self.state['source'] != str(self.metrics):
            self.lock.close()
            raise ValueError('This TensorBoard directory already belongs to another metrics file')
        launch = self.metrics.parent.parent / 'launch.json'
        if self.state['origin'] is None and launch.is_file():
            self.state['origin'] = datetime.datetime.fromisoformat(json.loads(launch.read_text())['created_at']).timestamp()
        self.writer = EventFileWriter(str(self.out), max_queue_size=100, flush_secs=5)

    def emit(self, values, step, wall_time):
        summary = []
        for tag, value in values.items():
            previous = self.state['last_scalars'].get(tag)
            if previous == [step, value]:
                continue
            summary.append(Summary.Value(tag=tag, simple_value=value))
            self.state['last_scalars'][tag] = [step, value]
        if summary:
            self.writer.add_event(Event(wall_time=wall_time, step=step, summary=Summary(value=summary)))

    def poll(self):
        if not self.metrics.exists():
            return 0
        stat = self.metrics.stat()
        identity = [stat.st_dev, stat.st_ino]
        if self.state['identity'] not in (None, identity) or stat.st_size < self.state['offset']:
            raise ValueError('Source metrics were replaced or truncated; use a new TensorBoard output directory')
        self.state['identity'] = identity
        config = self.metrics.parent / 'config.json'
        if not self.state['config_written'] and config.is_file():
            cfg = json.loads(config.read_text())
            values = {'config/unlimited_iterations': float(cfg.get('iterations') == 0),
                      'config/unlimited_actor_updates': float(cfg.get('actor_budget') == 0)}
            values.update({'config/' + k: float(cfg[k]) for k in ('horizon', 'num_envs', 'critic_envs', 'wm_batch', 'pi_batch', 'gamma') if k in cfg})
            self.emit(values, 0, self.state['origin'] or time.time())
            self.state['config_written'] = True
        count = 0
        with self.metrics.open('rb') as stream:
            stream.seek(self.state['offset'])
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break
                self.state['offset'] = stream.tell()
                try:
                    row = json.loads(line)
                    step = int(row['iteration'])
                    elapsed = float(row.get('elapsed_s', 0.))
                    if step < 0 or not math.isfinite(elapsed):
                        raise ValueError('Invalid log step/time')
                    values = scalars(row)
                except (ValueError, KeyError, TypeError, UnicodeError):
                    self.state['malformed_rows'] += 1
                    continue
                if self.state['origin'] is None:
                    self.state['origin'] = stat.st_mtime - elapsed
                if elapsed < self.state['last_elapsed']:
                    self.state['origin'] = max(time.time() - elapsed, self.state['last_wall_time'] - elapsed)
                wall_time = self.state['origin'] + elapsed
                self.emit(values, step, wall_time)
                self.state['last_elapsed'], self.state['last_wall_time'] = elapsed, wall_time
                self.state['rows'] += 1; count += 1
        self.writer.flush()
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.state_path)
        return count

    def close(self):
        self.writer.close()
        self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5.)
    parser.add_argument('--once', action='store_true', help='Backfill complete history and exit')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    follower = MetricsFollower(args.metrics, args.out)
    print(json.dumps(dict(metrics=str(follower.metrics), tensorboard=str(follower.out), axis='learner iteration')), flush=True)
    try:
        while not stopped:
            count = follower.poll()
            if count:
                print(f'imported={count} total_rows={follower.state["rows"]} malformed={follower.state["malformed_rows"]}', flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        follower.close()


if __name__ == '__main__':
    main()
