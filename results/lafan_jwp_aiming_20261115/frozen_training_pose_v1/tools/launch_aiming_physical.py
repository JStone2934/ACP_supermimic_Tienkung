"""Launch one reviewed physical-critic run from an immutable source snapshot."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--config', default='docs/config/standing_aiming_fullbody_v2/h8_physical_dual_scratch.json')
    parser.add_argument('--unit', default='aiming-fullbody-h8-physical.service')
    parser.add_argument('--estimated-capacity', action='store_true',
                        help='Use explicitly authorized estimated capacity; still require functional integration checks')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    owner = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'php-codex.service']).returncode == 0
    if owner and os.environ.get('PHP_OVERNIGHT_OWNER') != '1':
        raise RuntimeError('The PHP background owner is active; coordinate ownership before launch')
    for process in Path('/proc').glob('[0-9]*'):
        try:
            command = (process / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if '-m' in command and command[command.index('-m') + 1] in (
                'supermimic.acp.train_aiming_fullbody', 'supermimic.acp.train_aiming_physical'):
            raise RuntimeError(f'Another fullbody trainer exists: PID {process.name}')
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.unit]).returncode == 0:
        raise RuntimeError('The requested training service is already active')
    validation_path = Path(args.validation).resolve()
    validation = json.loads(validation_path.read_text())
    cfg = json.loads(Path(args.config).read_text())
    if not validation['tests']['passed'] or cfg['horizon'] not in (8, 16) or cfg['source'] is not None:
        raise ValueError('Require passed tests and a from-scratch H8/H16 configuration')
    if args.estimated_capacity:
        if (validation.get('capacity_basis') != 'estimated_no_benchmark_user_requested'
                or not validation.get('integration', {}).get('passed')
                or validation.get('config_sha256') != hashlib.sha256(Path(args.config).read_bytes()).hexdigest()):
            raise ValueError('Estimated capacity requires matching configuration and passed functional checks')
    else:
        capacity = validation['runs']['capacity']
        if (capacity['envs'], capacity['worker_envs'], capacity['horizon']) != (cfg['num_envs'], cfg['critic_envs'], cfg['horizon']):
            raise ValueError('Requested capacity does not match its verification')
        if not all(r['actor_parameters_exactly_restored'] and r['adam_exactly_restored'] for r in validation['runs'].values()):
            raise ValueError('Rollback verification did not pass')
    data = (root / 'data').resolve()
    if not (data / 'tienkung3_29dof/manifests/robot.json').is_file():
        raise FileNotFoundError('Robot assets are missing')
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    runner = run / 'runner'
    runner.mkdir()
    for directory in ('supermimic', 'tools', 'docs/config/standing_aiming_fullbody_v2'):
        shutil.copytree(root / directory, runner / directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (runner / 'data').symlink_to(data, target_is_directory=True)
    config_path = run / 'config.json'
    config_path.write_text(json.dumps(cfg, indent=2) + '\n')
    shutil.copyfile(validation_path, run / 'validation.json')
    files = sorted(p for p in runner.rglob('*') if p.is_file() and p.suffix in ('.py', '.json'))
    identity = {str(p.relative_to(runner)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (run / 'source_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    command = ['systemd-run', '--user', '--unit=' + args.unit, f"--description=H{cfg['horizon']} from-scratch fullbody physical critic, two GPUs",
               '--working-directory=' + str(runner), '--property=Type=exec', '--property=CPUQuota=1600%',
               '--property=Nice=5', '--property=Restart=no', '--property=OOMPolicy=stop',
               '--property=KillMode=control-group', '--property=TimeoutStopSec=60',
               '--property=StandardOutput=append:' + str(run / 'stdout.log'),
               '--property=StandardError=append:' + str(run / 'stdout.log'),
               '--setenv=CUDA_VISIBLE_DEVICES=0,1', '--setenv=MUJOCO_GL=egl',
               '--setenv=OMP_NUM_THREADS=8', '--setenv=MKL_NUM_THREADS=8', '--setenv=OPENBLAS_NUM_THREADS=1',
               '--setenv=TORCHINDUCTOR_COMPILE_THREADS=2',
               sys.executable, '-u', '-m', 'supermimic.acp.train_aiming_fullbody', '--from-scratch',
               '--config', str(config_path), '--log-dir', str(run / 'train')]
    receipt = dict(created_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command,
                   service=args.unit, source_root=str(root), frozen_runner=str(runner), training_dir=str(run / 'train'),
                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                   source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   source_changes=subprocess.check_output(['git', 'status', '--short'], text=True),
                   initialization='all networks random, reset-pose actor bias; no test checkpoint loaded',
                   horizon=cfg['horizon'], num_envs=cfg['num_envs'], critic_envs=cfg['critic_envs'],
                   capacity_basis='estimated_no_benchmark_user_requested' if args.estimated_capacity else 'verified',
                   acceptance='Original 30-second fullbody curriculum; final independent reload', launched=False)
    (run / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    subprocess.run(command, check=True)
    receipt['launched'] = True
    receipt['service_status'] = subprocess.check_output(['systemctl', '--user', 'show', args.unit,
        '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'MemoryHigh', '-p', 'MemoryMax'], text=True)
    (run / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(run_dir=str(run), service=args.unit, status=receipt['service_status']), indent=2))


if __name__ == '__main__':
    main()
