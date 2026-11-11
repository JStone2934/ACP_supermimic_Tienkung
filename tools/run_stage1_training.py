"""Own one Stage1 job: full-data preflight, update/resume smoke, then training.

Use a server systemd unit so terminal disconnects do not affect its children.
Completion here means the iteration budget finished, not policy acceptance.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from tools.prepare_stage1_dataset import sha

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def verify_checkpoint(path, expected_iteration, data_hash, expected_updates=None, geometry=None):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['iteration'] != expected_iteration or ck['data_hash'] != data_hash:
        raise ValueError('checkpoint iteration/data identity mismatch')
    if geometry is not None and ck['ctc_cfg'].get('contact_geometry', 'legacy') != geometry:
        raise ValueError('checkpoint reward geometry mismatch')
    dist = ck['distributed']
    if any(dist['skipped'].values()) or not dist.get('state_sha256'):
        raise ValueError('skipped optimizer updates or no verified distributed state')
    for name in ('wm', 'pi', 'critic'):
        if dist['updates'].get(name, 0) < 1:
            raise ValueError(f'no actual {name} updates')
        for value in ck[name].values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite {name} checkpoint tensor')
    if expected_updates and dist['updates'] != expected_updates:
        raise ValueError(f'wrong resumed update counts: {dist["updates"]}')
    return dict(iteration=ck['iteration'], data_hash=ck['data_hash'], distributed=dist,
                contact_geometry=ck['ctc_cfg'].get('contact_geometry', 'legacy'),
                geometry_transition=ck.get('geometry_transition'), checkpoint=str(path), checkpoint_sha256=sha(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    root, dataset = Path(config['run_dir']), Path(config['dataset'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'owner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(config, root, dataset)


def execute(config, root, dataset):
    for path, expected in config['source_hashes'].items():
        if sha(REPO/path) != expected:
            raise ValueError(f'code changed since launch binding: {path}')
    if sha(dataset/'manifest.json') != config['dataset_manifest_sha256']:
        raise ValueError('dataset manifest changed')
    evidence = REPO/'docs/stage1/evidence'
    progress = REPO/'docs/stage1/progress.md'
    status_path = root/'status.json'
    target = config['iterations']
    geometry = config.get('contact_geometry', 'legacy')
    preflight_path = Path(config.get('preflight_evidence', evidence/'dataset_preflight.json'))
    bootstrap = config.get('bootstrap_checkpoint')
    if bootstrap and sha(bootstrap) != config['bootstrap_checkpoint_sha256']:
        raise ValueError('migration source checkpoint changed')

    def status(stage, state, **extra):
        now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
        value = dict(time=now.isoformat(), stage=stage, state=state, run_dir=str(root), **extra)
        csv_path = root/'train'/'train_log.csv'
        rows = []
        if csv_path.exists():
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
        line = '正式训练尚未产生首条迭代日志。'
        if rows:
            last = rows[-1]
            value['training'] = last
            iteration = int(last['iteration'])
            line = f"正式训练：{iteration}/{target} 轮（{100*iteration/target:.2f}%）；最近窗口 {last['it_per_s']} 轮/秒。"
            # Only use windows entirely after actor warmup and initial compilation.
            rates = [float(r['it_per_s']) for r in rows[-8:] if int(r['iteration']) >= config.get('eta_min_iteration', 600) and float(r['it_per_s']) > 0]
            if rates:
                rate = sorted(rates)[len(rates)//2]
                remaining = max(0, target-iteration)/rate
                value['eta'] = dict(rate_it_s=rate, remaining_hours=remaining/3600,
                                    estimated_finish=(now+dt.timedelta(seconds=remaining)).isoformat(),
                                    scope='iteration budget only; evaluation extra')
                line += f" 按近期 actor 训练速度预计剩余 {remaining/3600:.1f} 小时；此估计不含评估，不代表收敛保证。"
            else:
                line += ' 尚未积累足够的稳定策略训练窗口，暂不据此估计全程。'
        write_json(status_path, value)
        progress.write_text('# Stage1 当前进度\n\n'+f"北京时间 {now.isoformat()}，阶段 `{stage}`，状态 `{state}`。\n\n"+line+'\n\n'
                            +'完整输入：296 条原动作、2960 个原始地形编号、1812360 个展开帧。\n\n'
                            +f"接触奖励几何：`{geometry}`。\n\n"
                            +f"运行目录：`{root}`。状态由后台控制器每 60 秒刷新；日志及检查点写入该目录。\n\n"
                            +'本页的训练进度不代表策略已通过完整动作仿真评估。\n')

    def run(stage, command, done=None):
        marker = root/(stage+'.done.json')
        if marker.exists():
            return
        write_json(root/(stage+'.command.json'), command)
        with (root/(stage+'.stdout.log')).open('a') as log:
            process = subprocess.Popen([str(x) for x in command], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            status(stage, 'running', pid=process.pid, command=[str(x) for x in command])
            print(f'{stage}: pid {process.pid}', flush=True)
            try:
                while True:
                    try:
                        code = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        status(stage, 'running', pid=process.pid)
                if code:
                    raise RuntimeError(f'{stage} failed with exit {code}; see {root/(stage+".stdout.log")}')
                result = done() if done else dict(passed=True)
                write_json(marker, result)
                status(stage, 'completed')
            except BaseException as exc:
                if process.poll() is None:
                    process.terminate()
                status(stage, 'failed', error=str(exc))
                raise

    run('preflight', [sys.executable, '-m', 'tools.check_stage1_dataset', '--dataset', str(dataset),
                      '--out', str(preflight_path), '--contact-geometry', geometry])
    preflight = json.loads(preflight_path.read_text())
    if bootstrap:
        import torch
        source = torch.load(bootstrap, map_location='cpu', weights_only=False)
        if (source['iteration'] != 500 or source['data_hash'] != preflight['data_hash']
                or source['distributed']['updates'] != dict(wm=5000) or source['distributed']['skipped']):
            raise ValueError('expected intact iteration-500 WM-only migration source')
        del source
    common = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
              '--master-addr=127.0.0.1', f"--master-port={config['master_port']}", '--max-restarts=0',
              '--module', 'supermimic.acp.train_stage1', '--robot', 'tienkung3_29dof',
              '--motion_dir', str(dataset/'motions'), '--contacts_dir', str(dataset/'contacts'),
              '--num_envs', '512', '--wm_batch', '256', '--pi_batch', '256', '--buffer_steps', '1000',
              '--horizon', '32', '--hist', '10', '--wm_window', '8', '--wm_batches', '10',
              '--critic_epochs', '5', '--critic_minibatches', '4', '--reward_chunk_size', str(config.get('reward_chunk_size', 64)),
              '--amp', 'bf16', '--reward', 'paper', '--seed', '0', '--verify_sync', '--save_full_checkpoints']
    common += ['--contact_geometry', geometry]
    smoke = root/'smoke'
    run('smoke', common+['--iterations', '3', '--wm_only_iters', '0', '--ckpt_every', '3',
                         '--log_every', '1', '--log_dir', str(smoke)],
        lambda: verify_checkpoint(smoke/'last.pt', 3, preflight['data_hash'], dict(wm=30, pi=3, critic=60), geometry))
    run('resume_smoke', common+['--iterations', '4', '--wm_only_iters', '0', '--ckpt_every', '4',
                                '--log_every', '1', '--log_dir', str(smoke), '--resume', str(smoke/'last.pt')],
        lambda: verify_checkpoint(smoke/'last.pt', 4, preflight['data_hash'], dict(wm=40, pi=4, critic=80), geometry))
    if bootstrap:
        # This migration starts at the saved WM-only boundary; independent smoke weights are never promoted.
        migration = root/'migration_smoke'
        run('migration_smoke', common+['--iterations', '502', '--wm_only_iters', '500', '--ckpt_every', '502',
                                      '--log_every', '1', '--log_dir', str(migration), '--resume', bootstrap,
                                      '--allow_contact_geometry_change'],
            lambda: verify_checkpoint(migration/'last.pt', 502, preflight['data_hash'], dict(wm=5020, pi=2, critic=40), geometry))
        run('migration_resume_smoke', common+['--iterations', '503', '--wm_only_iters', '500', '--ckpt_every', '503',
                                             '--log_every', '1', '--log_dir', str(migration), '--resume', str(migration/'last.pt')],
            lambda: verify_checkpoint(migration/'last.pt', 503, preflight['data_hash'], dict(wm=5030, pi=3, critic=60), geometry))
    train = root/'train'
    command = common+['--iterations', str(target), '--wm_only_iters', '500', '--ckpt_every', '500',
                     '--log_every', '25', '--benchmark_warmup', '25', '--benchmark_steps', '50', '--log_dir', str(train)]
    if (train/'last.pt').exists():
        command += ['--resume', str(train/'last.pt')]
    elif bootstrap:
        command += ['--resume', bootstrap, '--allow_contact_geometry_change']
    run('training', command, lambda: verify_checkpoint(train/'last.pt', target, preflight['data_hash'], geometry=geometry))
    status('training_finished_evaluation_pending', 'completed')


if __name__ == '__main__':
    main()
