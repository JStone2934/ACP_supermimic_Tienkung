"""Launch the reviewed H8 experiment once; never starts alongside another fullbody trainer."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

root = Path('/home/js/robot/locomotion/ACP_supermimic_Tienkung')
run = root / 'logs/tienkung3_29dof/standing_aiming_fullbody_v2/h8_request_stage_20261114'
for process in Path('/proc').glob('[0-9]*'):
    try:
        args = (process / 'cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError):
        continue
    if '-m' in args and args[args.index('-m') + 1] == 'supermimic.acp.train_aiming_fullbody':
        raise RuntimeError(f'Another fullbody trainer exists: {process.name}')
if (run / 'train').exists() and any((run / 'train').iterdir()):
    raise RuntimeError('Formal output is not empty; inspect before any resume')
identity = json.loads((run / 'source_identity_final.json').read_text())
for name, digest in identity.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Source changed after review: {name}')
pilot = json.loads((run / 'checks/pilot_conservative/evaluation/000300.json').read_text())
pilot_status = json.loads((run / 'checks/pilot_conservative/status.json').read_text())
if pilot_status['iteration'] != 300 or not all(math.isfinite(pilot_status[k]) for k in ('actor_loss', 'critic_loss', 'wm_loss')):
    raise RuntimeError('Conservative pilot did not finish with finite training metrics')
if pilot_status['bootstrap_max'] > 28.401:
    raise RuntimeError('Bootstrap exceeded the derived task reward upper bound')
unit = 'aiming-fullbody-h8.service'
cmd = ['systemd-run', '--user', '--unit=' + unit, '--description=Fullbody aiming H8 request penalty and stage exploration',
       '--working-directory=' + str(root), '--property=Type=exec', '--property=MemoryHigh=16G',
       '--property=MemoryMax=24G', '--property=MemorySwapMax=2G', '--property=CPUQuota=800%',
       '--property=Nice=10', '--property=OOMPolicy=stop', '--property=Restart=no',
       '--property=KillMode=control-group', '--property=TimeoutStopSec=30',
       '--property=StandardOutput=append:' + str(run / 'stdout.log'),
       '--property=StandardError=append:' + str(run / 'stdout.log'),
       '--setenv=PYTHONNOUSERSITE=1', '--setenv=MUJOCO_GL=egl', '--setenv=OMP_NUM_THREADS=8',
       '--setenv=MKL_NUM_THREADS=8', '--setenv=TORCHINDUCTOR_COMPILE_THREADS=4',
       '/home/js/miniconda3/envs/acp_supermimic/bin/python', '-u', '-m', 'supermimic.acp.train_aiming_fullbody',
       '--init', str(run / 'preparation/recovery_init.pt'), '--config', str(run / 'config.json'),
       '--log-dir', str(run / 'train'), '--iterations', '0', '--horizon', '8']
subprocess.run(cmd, check=True)
status = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
                                  '-p', 'MemoryHigh', '-p', 'MemoryMax', '-p', 'MemorySwapMax'], text=True)
receipt = dict(started_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), service=unit,
               command=cmd, service_status=status, source_iteration=20000, new_iteration_origin=0,
               initialization_sha256=hashlib.sha256((run / 'preparation/recovery_init.pt').read_bytes()).hexdigest(),
               initial_curriculum='A recertification before B1', actor_lr=1e-6,
               pilot_standing_pass=all(s['standing_pass'] for s in pilot['scenarios']),
               requested_envs=512, wm_batch=256, actor_batch=256, horizon=8, iterations=0,
               acceptance='Original fullbody v2 physical curriculum criteria unchanged')
(run / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(status)
