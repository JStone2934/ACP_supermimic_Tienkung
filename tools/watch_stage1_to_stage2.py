"""Durable single-owner training monitor and sequential Stage1 -> Stage2 handoff.

This is a bounded job controller, not another autonomous coding agent. Heavy jobs
remain in separate systemd units in php-compute.slice with unchanged memory caps.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
GIB = 2**30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n');temporary.replace(path)


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip() in ('active', 'activating', 'deactivating')


def health(unit=None):
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    value = dict(time=dt.datetime.now().astimezone().isoformat(), available_gib=int(mem['MemAvailable'].split()[0])*1024/GIB,
                 disk_free_gib=shutil.disk_usage(REPO).free/GIB)
    value['ssh_service'] = subprocess.run(['systemctl', 'is-active', 'ssh.service'], capture_output=True, text=True, timeout=5).stdout.strip()
    try:
        with socket.create_connection(('127.0.0.1', 22), timeout=3) as sock:
            value['ssh_banner'] = sock.recv(200).decode(errors='replace').strip()
        value['ssh_ok'] = value['ssh_service'] == 'active' and value['ssh_banner'].startswith('SSH-2.0-')
    except OSError as exc:
        value.update(ssh_ok=False, ssh_error=str(exc))
    if unit and active(unit):
        group = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ControlGroup', '--value'], text=True).strip()
        if group:
            root = Path('/sys/fs/cgroup')/group.lstrip('/')
            value['unit'] = unit
            for key in ('memory.current', 'memory.peak', 'memory.high', 'memory.max', 'memory.events'):
                path = root/key
                if path.exists():value[key] = path.read_text().strip()
    return value


def change_flags(command, **values):
    result = list(command)
    for name, value in values.items():
        key = '--'+name
        if key in result:result[result.index(key)+1] = str(value)
        else:result.extend([key, str(value)])
    return result


def check_evaluation(report, names, data_hash):
    if set(report['per_clip']) != set(names) or report['checkpoint_data_hash'] != data_hash:
        raise ValueError('evaluation coverage/checkpoint identity mismatch')
    if report['term_window'] != 0 or not math.isfinite(report['norm_return']):
        raise ValueError('changed termination or nonfinite evaluation')
    if report['n_failed'] != sum(bool(c['failed']) for c in report['per_clip'].values()):
        raise ValueError('inconsistent evaluation failure accounting')
    for row in report['per_clip'].values():
        if not all(math.isfinite(float(v)) for v in row.values()):raise ValueError('nonfinite rollout metric')


class Watcher:
    def __init__(self, cfg):
        self.cfg = cfg;self.root = Path(cfg['run_dir']);self.root.mkdir(parents=True, exist_ok=True)
        self.control = self.root/'control.json'
        self.retry_path = self.root/'retries.json'
        self.lock = (self.root/'owner.lock').open('w');fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stage = 'waiting_stage1';self.current_unit = None

    def status(self, state, **extra):
        value = dict(time=dt.datetime.now().astimezone().isoformat(), stage=self.stage, state=state, **extra)
        write(self.root/'status.json', value)
        Path(self.cfg['progress']).write_text('# Stage1 → Stage2 后台进度\n\n'+
            f"北京时间 {value['time']}；阶段 `{self.stage}`；状态 `{state}`。\n\n"+
            f"当前任务：`{self.current_unit}`。\n\n"+
            'Stage1 完成迭代后进行全部2960条分批评估，再进入离线规划器和Stage2联合训练。阶段完成不代表最终策略验收通过。\n\n'+
            '内存与SSH健康、异常原因和下一步详见 status.json、health.jsonl 和 events.jsonl。\n')

    def event(self, kind, **extra):
        with (self.root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=dt.datetime.now().astimezone().isoformat(), kind=kind, stage=self.stage, **extra))+'\n')

    def sample(self):
        value = health(self.current_unit)
        with (self.root/'health.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        return value

    def controls(self):
        control = json.loads(self.control.read_text()) if self.control.exists() else {}
        if control.get('pause') or control.get('cancel'):
            if self.current_unit and active(self.current_unit):
                subprocess.run(['systemctl', '--user', 'stop', self.current_unit], check=True)
                self.event('user_control_stop', unit=self.current_unit)
            if control.get('cancel'):raise RuntimeError('user cancelled continuation')
            self.status('paused_by_user')
            while json.loads(self.control.read_text()).get('pause'):
                if json.loads(self.control.read_text()).get('cancel'):raise RuntimeError('user cancelled continuation')
                self.sample();time.sleep(self.cfg['poll_seconds'])
            return True
        return False

    def wait_resources(self):
        good = 0
        while good < 3:
            self.controls();sample = self.sample()
            okay = sample['ssh_ok'] and sample['available_gib'] >= self.cfg['start_min_available_gib'] and sample['disk_free_gib'] >= 20
            good = good+1 if okay else 0
            self.status('checking_resources' if okay else 'waiting_resources', health=sample)
            if good < 3:time.sleep(self.cfg['poll_seconds'])

    def bindings(self):
        for path, expected in self.cfg['source_hashes'].items():
            if sha(REPO/path) != expected:raise ValueError('bound source changed: '+path)
        if sha(Path(self.cfg['dataset'])/'manifest.json') != self.cfg['manifest_sha256']:
            raise ValueError('dataset manifest changed')

    def retry(self, reason):
        history = json.loads(self.retry_path.read_text()) if self.retry_path.exists() else []
        now = time.time();recent = [x for x in history if now-x['time'] < 1800]
        if len(recent) >= 3:raise RuntimeError('three recovery attempts within 30 minutes; '+reason)
        history.append(dict(time=now, stage=self.stage, reason=reason));write(self.retry_path, history)
        self.event('recovery', reason=reason)

    def monitor(self, unit, output):
        self.current_unit = unit;bad = 0;started = time.time()
        while active(unit):
            if self.controls():return False
            sample = self.sample()
            unsafe = not sample['ssh_ok'] or sample['available_gib'] < self.cfg['critical_available_gib'] or sample['disk_free_gib'] < 10
            bad = bad+1 if unsafe else 0
            paths = [Path(output), Path(output).parent/'train_log.csv', Path(output).parent/'last.pt']
            latest = max([started]+[p.stat().st_mtime for p in paths if p.exists()])
            stale = time.time()-latest > self.cfg['stale_seconds']
            if bad >= 3 or stale:
                reason = 'health protection' if bad >= 3 else 'no log/checkpoint progress'
                self.event('protective_stop', unit=unit, reason=reason, health=sample)
                subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
                self.status('protective_stop', reason=reason, health=sample)
                self.retry(reason);self.wait_resources();return False
            self.status('running', health=sample)
            time.sleep(self.cfg['poll_seconds'])
        return True

    def systemd(self, unit, command, output):
        self.bindings();output = Path(output);output.parent.mkdir(parents=True, exist_ok=True)
        launch = ['systemd-run', '--user', '--unit='+unit.removesuffix('.service'), '--slice=php-compute.slice',
                  '--property=WorkingDirectory='+str(REPO), '--property=MemoryHigh=64G', '--property=MemoryMax=80G',
                  '--property=Restart=no', '--property=KillMode=control-group', '--property=TimeoutStopSec=90',
                  '--property=StandardOutput=append:'+str(output), '--property=StandardError=append:'+str(output),
                  '--setenv=CUDA_VISIBLE_DEVICES=0,1', '--setenv=OMP_NUM_THREADS=4', '--setenv=OPENBLAS_NUM_THREADS=1',
                  '--setenv=TORCHINDUCTOR_COMPILE_THREADS=4', '--setenv=PYTHONUNBUFFERED=1', '--setenv=MUJOCO_GL=egl',
                  '--setenv=TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1', *command]
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], capture_output=True)
        subprocess.run(launch, check=True)

    def job(self, name, command, *, checkpoint=None, verify=None):
        self.stage = name;unit = self.cfg['worker_prefix']+'-'+name+'.service';self.current_unit = unit
        folder = self.root/'jobs'/name;folder.mkdir(parents=True, exist_ok=True)
        identity = folder/'command.json';done = folder/'done.json';output = folder/'stdout.log'
        if identity.exists() and json.loads(identity.read_text()) != command:raise ValueError('existing job identity changed: '+name)
        write(identity, command)
        if done.exists():return json.loads(done.read_text())
        while True:
            if not active(unit):
                # Never overlap the adopted production trainer with a successor job.
                if active(self.cfg['stage1_unit']):raise RuntimeError('Stage1 is still active')
                self.wait_resources()
                actual = list(command)
                if checkpoint and Path(checkpoint).exists():actual = change_flags(actual, resume=checkpoint)
                write(folder/'actual_command.json', actual)
                self.systemd(unit, actual, output)
            if not self.monitor(unit, output):continue
            result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'Result', '--value'], capture_output=True, text=True).stdout.strip()
            # Successful transient services may already have been unloaded; verify artifacts either way.
            try:
                if result not in ('', 'success'):raise RuntimeError('worker result '+result)
                report = verify() if verify else {'completed': True}
                write(done, report);self.event('job_complete', unit=unit);return report
            except Exception as exc:
                tail = output.read_text(errors='replace')[-20000:] if output.exists() else ''
                if 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail:
                    raise RuntimeError('worker memory failure; retained checkpoint, no limit relaxation') from exc
                self.retry(str(exc));self.wait_resources()

    def torchrun(self, module, arguments):
        return [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
                '--master-addr=127.0.0.1', '--master-port='+str(self.cfg['master_port']), '--max-restarts=0',
                '--module', module, *map(str, arguments)]

    def check_checkpoint(self, path, target, *, offline=False, previous=None, planner_required=True):
        import torch
        ck = torch.load(path, map_location='cpu', weights_only=False)
        counter = 'step' if offline else 'iteration'
        if ck[counter] != target or ck['data_hash'] != self.cfg['data_hash']:raise ValueError('checkpoint target/data mismatch')
        if ck['ctc_cfg']['contact_geometry'] != 'spheres224':raise ValueError('checkpoint geometry changed')
        dist = ck['distributed'];updates = dist['updates']
        if any(dist['skipped'].values()) or not dist.get('state_sha256'):raise ValueError('skipped or unsynchronized updates')
        models = ['planner'] if offline else ['wm', 'pi', 'critic']+(['planner'] if 'planner' in ck else [])
        for name in models:
            for tensor in ck[name].values():
                if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():raise ValueError('nonfinite '+name)
            if (name != 'planner' or planner_required) and updates.get(name, 0) <= (previous or {}).get(name, 0):
                raise ValueError('no actual update growth: '+name)
        return dict(checkpoint=str(path), sha256=sha(path), target=target, updates=updates, data_hash=ck['data_hash'])

    def wait_stage1(self):
        self.stage = 'stage1';self.current_unit = self.cfg['stage1_unit']
        root = Path(self.cfg['stage1_run']);checkpoint = root/'train'/f"ckpt_{self.cfg['stage1_iterations']:06d}.pt"
        while True:
            if active(self.current_unit):
                if not self.monitor(self.current_unit, root/'training.stdout.log'):continue
            if (root/'training.done.json').exists() and checkpoint.exists():
                result = self.check_checkpoint(checkpoint, self.cfg['stage1_iterations'], planner_required=False)
                write(self.root/'stage1_complete.json', result);return checkpoint
            self.controls();self.retry('Stage1 stopped before target');self.wait_resources();self.bindings()
            subprocess.run(['systemctl', '--user', 'reset-failed', self.current_unit], capture_output=True)
            subprocess.run(self.cfg['stage1_systemd_command'], check=True)

    def evaluate_all(self, checkpoint, tag):
        dataset = Path(self.cfg['dataset']);manifest = json.loads((dataset/'manifest.json').read_text())
        entries = sorted(manifest['entries'], key=lambda e:e['key']);combined = {};weighted_return = 0
        for index, start in enumerate(range(0, len(entries), self.cfg['evaluation_batch'])):
            selected = entries[start:start+self.cfg['evaluation_batch']]
            subset = self.root/'evaluation_sets'/f'{index:03d}'
            for entry in selected:
                for key in ('motion', 'terrain', 'labels'):
                    source = Path(entry[key])
                    if sha(source) != entry[key+'_sha256']:raise ValueError('evaluation input changed')
                    target = subset/source.relative_to(dataset);target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():target.symlink_to(source)
                    if target.resolve() != source.resolve():raise ValueError('evaluation symlink changed')
            write(subset/'selection.json', selected)
            output = self.root/'evaluations'/tag/f'{index:03d}.json';output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'supermimic.acp.evaluate', '--ckpt', str(checkpoint),
                       '--motion_dir', str(subset/'motions'), '--contacts_dir', str(subset/'contacts'),
                       '--out', str(output), '--seed', '0']
            if tag == 'stage2':command += ['--planner', str(checkpoint)]
            def verify():
                report = json.loads(output.read_text());check_evaluation(report, [e['key'] for e in selected], self.cfg['data_hash'])
                return dict(output=str(output), sha256=sha(output), clips=len(selected), failed=report['n_failed'])
            self.job(f'{tag}-eval-{index:03d}', command, verify=verify)
            report = json.loads(output.read_text());check_evaluation(report, [e['key'] for e in selected], self.cfg['data_hash'])
            if combined.keys() & report['per_clip'].keys():raise ValueError('duplicate evaluated clips')
            combined.update(report['per_clip']);weighted_return += report['norm_return']*len(selected)
        if set(combined) != {e['key'] for e in entries}:raise ValueError('incomplete evaluation')
        failed = [name for name,c in combined.items() if c['failed']]
        summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), clips=len(combined),
                       n_failed=len(failed), failed=failed, norm_return=weighted_return/len(combined), per_clip=combined,
                       acceptance='evaluation complete, not automatic final policy acceptance',
                       return_precision='weighted aggregation of rounded per-batch means')
        write(self.root/'evaluations'/tag/'summary.json', summary)
        return summary

    def execute(self):
        self.bindings();tracker = self.wait_stage1();evaluation = self.evaluate_all(tracker, 'stage1')
        if evaluation['n_failed'] == evaluation['clips']:raise RuntimeError('all Stage1 rollouts failed; do not hand over an unusable tracker')
        write(self.root/'selected_tracker.json', dict(checkpoint=str(tracker), sha256=sha(tracker),
              reason='final full Stage1 checkpoint after complete rollout accounting; not a final capability acceptance',
              evaluation=str(self.root/'evaluations/stage1/summary.json')))
        prepared = self.cfg['prepared']
        offline = self.root/'offline';joint = self.root/'stage2';smoke = self.root/'smoke'
        def materialize(values):
            mapping = dict(STAGE1_ACCEPTED_CHECKPOINT=str(tracker), PLANNER_CHECKPOINT=str(offline/'last.pt'),
                           STAGE2_OFFLINE_RUN_DIR=str(offline), STAGE2_JOINT_RUN_DIR=str(joint))
            return [mapping.get(v,v) for v in values]
        offline_command = self.torchrun('supermimic.acp.train_planner_offline', materialize(prepared['offline_arguments']))
        previous = None
        for target in (2000, 4000, 20000):
            report = self.job('offline-'+str(target), change_flags(offline_command, steps=target), checkpoint=offline/'last.pt',
                verify=lambda target=target,previous=previous:self.check_checkpoint(offline/'last.pt', target, offline=True, previous=previous))
            previous = report['updates']
        joint_command = self.torchrun('supermimic.acp.train_stage2', materialize(prepared['joint_arguments']))
        # Exercise nonzero planned reward and restore independently; never promote diagnostic weights.
        smoke_command = change_flags(joint_command, log_dir=smoke, u_obs=0, u_rew=0, n_ann=1, log_every=1, ckpt_every=1,
                                     benchmark_steps=0)
        report = self.job('smoke-3', change_flags(smoke_command, iterations=3), checkpoint=smoke/'last.pt',
                         verify=lambda:self.check_checkpoint(smoke/'last.pt', 3, planner_required=False))
        self.job('smoke-4', change_flags(smoke_command, iterations=4), checkpoint=smoke/'last.pt',
                 verify=lambda:self.check_checkpoint(smoke/'last.pt', 4, previous=report['updates'], planner_required=False))
        report = self.job('joint-1000', change_flags(joint_command, iterations=1000), checkpoint=joint/'last.pt',
                         verify=lambda:self.check_checkpoint(joint/'last.pt', 1000))
        self.job('joint-40000', joint_command, checkpoint=joint/'last.pt',
                 verify=lambda:self.check_checkpoint(joint/'last.pt', 40000, previous=report['updates']))
        self.evaluate_all(joint/'ckpt_040000.pt', 'stage2')
        self.stage = 'stage2_evaluation_complete';self.status('completed_pending_policy_review')


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true');args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.check_only:
        for path,digest in cfg['source_hashes'].items():
            if sha(REPO/path) != digest:raise ValueError('bound source changed: '+path)
        print(json.dumps(health(cfg['stage1_unit']), indent=2));return
    watcher = Watcher(cfg)
    try:watcher.execute()
    except Exception as exc:
        watcher.event('blocked', error=str(exc));watcher.status('blocked', error=str(exc))
        # Do not loop on algorithmic failures, silently weaken thresholds or start a duplicate job.
        raise


if __name__ == '__main__':main()
