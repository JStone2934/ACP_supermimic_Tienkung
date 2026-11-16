"""Bounded, sequential full-data hyperparameter trials from one saved checkpoint."""
import argparse
import csv
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tools.prepare_stage1_dataset import sha
from tools.run_stage1_training import verify_checkpoint, write_json


def flag(command, key, value):
    key = '--' + key
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command.extend([key, str(value)])


def evaluate(cfg, checkpoint, output):
    if output.exists():
        raise RuntimeError(f'evaluation already exists: {output}')
    subset = Path(cfg['out'])/'eval_dataset'
    command = [sys.executable, '-m', 'supermimic.acp.evaluate', '--ckpt', str(checkpoint),
               '--motion_dir', str(subset/'motions'), '--contacts_dir', str(subset/'contacts'),
               '--out', str(output), '--seed', '0']
    write_json(output.with_suffix('.command.json'), command)
    # Trusted, locally generated full checkpoints contain optimizer/RNG state.
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    with output.with_suffix('.stdout.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True, timeout=600)
    result = json.loads(output.read_text())
    entries = json.loads((subset/'manifest.json').read_text())['entries']
    expected = {e['key']: e for e in entries}
    if set(result['per_clip']) != set(expected):
        raise ValueError('evaluation clip coverage mismatch')
    if result['checkpoint_data_hash'] != cfg['data_hash'] or result['term_window'] != 0:
        raise ValueError('evaluation checkpoint/termination changed')
    metrics = dict(norm_return=result['norm_return'], n_failed=result['n_failed'],
                   n_clips=len(expected), per_folder=result['per_folder'])
    for metric in ('root_rmse_m', 'joint_rmse_rad', 'torque_saturation_fraction', 'velocity_exceed_fraction'):
        metrics[metric] = sum(c[metric] for c in result['per_clip'].values())/len(expected)
    metrics['mean_alive_fraction'] = sum(c['alive_steps']/(expected[name]['frames']-1)
                                         for name, c in result['per_clip'].items())/len(expected)
    if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))):
        raise ValueError('nonfinite evaluation')
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    cfg = json.loads(ap.parse_args().config.read_text())
    root = Path(cfg['out'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'owner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha(cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('source checkpoint changed')
    if any(sha(p) != h for p, h in cfg['source_hashes'].items()):
        raise ValueError('bound sources changed')
    if subprocess.run(['systemctl', '--user', 'is-active', cfg['production_unit']],
                      capture_output=True, text=True).stdout.strip() == 'active':
        raise RuntimeError('production is still active')

    def sample_gpu():
        with (root/'gpu_samples.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time', 'candidate', 'stage', 'iteration', 'gpu', 'utilization', 'memory_used_mib'])
            while True:
                try:
                    state = json.loads((root/'status.json').read_text())
                    candidate = state.get('candidate')
                    log = root/str(candidate)/'train_log.csv'
                    rows = list(csv.DictReader(log.open())) if log.exists() else []
                    iteration = rows[-1]['iteration'] if rows else ''
                    data = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
                    for line in data.strip().splitlines():
                        writer.writerow([dt.datetime.now().astimezone().isoformat(), candidate,
                                         state['stage'], iteration, *[v.strip() for v in line.split(',')]])
                    stream.flush()
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                time.sleep(1)

    threading.Thread(target=sample_gpu, daemon=True).start()

    def status(stage, candidate=None):
        state = dict(time=dt.datetime.now().astimezone().isoformat(), stage=stage, candidate=candidate,
                     checkpoint=cfg['checkpoint'], run_dir=str(root))
        write_json(root/'status.json', state)
        Path('docs/stage1/progress.md').write_text('# Stage1 当前进度\n\n'+
            f"北京时间 {state['time']}，超参对照阶段 `{stage}`，候选 `{candidate}`。\n\n"+
            f"正式训练已保存在第 {cfg['start_iteration']} 轮，三个候选顺序运行，训练输入均为全部 2960 条。\n\n"+
            '以固定 60 条动作/地形组合检查短训表现；这不是独立测试集，也不代替最终全量验收。\n\n'+
            f"证据：`{root}`。\n")

    reports = {}
    status('source_evaluation')
    source_eval = evaluate(cfg, cfg['checkpoint'], root/'source_eval.json')
    write_json(root/'source_metrics.json', source_eval)
    for candidate in cfg['candidates']:
        name = candidate['name']
        out = root/name
        if out.exists():
            raise RuntimeError(f'candidate already exists; inspect before restarting: {out}')
        status('training', name)
        command = list(cfg['base_command'])
        for key, value in dict(**candidate['parameters'], log_dir=out, resume=cfg['checkpoint'],
                              iterations=cfg['start_iteration']+cfg['steps'],
                              benchmark_warmup=50, benchmark_steps=100,
                              log_every=25, ckpt_every=100).items():
            flag(command, key, value)
        write_json(root/(name+'.command.json'), command)
        print(name+': training', flush=True)
        start = time.monotonic()
        with (root/(name+'.stdout.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)
        elapsed = time.monotonic()-start
        params = candidate['parameters']
        increment = dict(wm=cfg['steps']*params['wm_batches'], pi=cfg['steps'], critic=cfg['steps']*20)
        expected = {k: cfg['source_updates'][k]+v for k, v in increment.items()}
        checkpoint = verify_checkpoint(out/'last.pt', cfg['start_iteration']+cfg['steps'],
                                       cfg['data_hash'], expected, 'spheres224')
        rows = list(csv.DictReader((out/'train_log.csv').open()))
        if not rows or not all(math.isfinite(float(v)) for row in rows for v in row.values()):
            raise ValueError('missing/nonfinite training metrics')
        bench = json.loads((out/'benchmark.json').read_text())
        if bench['updates'] != dict(wm=100*params['wm_batches'], pi=100, critic=2000):
            raise ValueError('benchmark update counts mismatch')
        seconds = bench['elapsed_s']/100
        report = dict(parameters=params, checkpoint=checkpoint, benchmark=bench,
                      seconds_per_iteration=seconds, process_elapsed_s=elapsed,
                      simulation_transitions_per_s=params['num_envs']/params['utd']/seconds,
                      wm_windows_per_s=params['wm_batch']*params['wm_batches']/seconds,
                      policy_trajectories_per_s=params['pi_batch']/seconds,
                      final_training_metrics=rows[-1])
        write_json(out/'training_verified.json', report)
        status('evaluation', name)
        report['evaluation'] = evaluate(cfg, out/'last.pt', out/'eval.json')
        write_json(out/'verified.json', report)
        reports[name] = report
        write_json(root/'results.json', dict(source_evaluation=source_eval, candidates=reports))
        print(name+f': verified {seconds:.4f} s/it; eval '+str(report['evaluation']), flush=True)
    status('comparisons_complete')


if __name__ == '__main__':
    main()
