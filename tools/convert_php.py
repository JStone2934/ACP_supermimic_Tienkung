"""Scan, retarget, check and label the complete PHP library with a resumable GPU/CPU queue.

Examples:
  python -m tools.convert_php --scan --out .cache/php-plan
  python -m tools.convert_php --out data/tienkung3_29dof/motions/php_full_v1 --devices 0,1
  python -m tools.convert_php --out data/tienkung3_29dof/motions/php_full_v1 --resume
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from supermimic.acp.php_conversion import (COLLISION_POLICY, VERSION, FullBodyRetargeter, SolverConfig,
                                         atomic_json, atomic_npz, digest, discover)
from supermimic.acp.robots import ROOT, get_robot, sha256
from supermimic.acp.source_contacts import SOURCE_CONTACT_POLICY
from supermimic.acp.root_continuity import POLICY as ROOT_POLICY
from supermimic.acp.terrain import terrain_path
from supermimic.acp.retarget_fidelity import POLICY as FIDELITY_POLICY, POSTURE_POLICY
from supermimic.acp.contact_intent import POLICY as CONTACT_INTENT_POLICY, PATCH_POLICY
from tools.check_php_conversion import collision_evidence, evaluate_task

TERMINAL = ('accepted', 'rejected', 'failed')


def source_fingerprint():
    paths = sorted((ROOT/'supermimic').rglob('*.py')) + [ROOT/'tools'/n for n in
                  ('convert_php.py', 'check_php_conversion.py', 'check_php_pilot.py', 'retarget_php_pilot.py', 'polish_php_conversion.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def load_json(path):
    return json.loads(Path(path).read_text())


def output_hashes(root, task):
    key = task['record']['key']
    paths = [Path(root)/'candidates'/(key+suffix) for suffix in ('_motion.npz', '_terrain.npy', '_intent.npz')]
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def verify_task(root, state):
    expected = dict(state.get('candidate_files', {}))
    expected.update(state.get('exported', {}))
    for key in ('quality', 'labels'):
        if key+'_path' in state:
            expected[state[key+'_path']] = state[key+'_hash']
    if not expected:
        raise ValueError('completion marker has no output hashes')
    for relative, checksum in expected.items():
        path = Path(root)/relative
        if not path.is_file() or sha256(path) != checksum:
            raise ValueError(f'output missing or changed: {path}')


def gpu_worker(device, config, tasks, events, root):
    """Persistent model per GPU; only this worker writes its current task's candidates."""
    torch.set_num_threads(4)
    try:
        solver = FullBodyRetargeter(device, SolverConfig(**config))
        metadata = solver.dst.metadata()
        events.put(('ready', device, os.getpid()))
        while True:
            task = tasks.get()
            if task is None:
                return
            events.put(('started', task['id'], device))
            started = time.monotonic()
            try:
                record = task['record']
                if (sha256(record['source']) != record['source_hash'] or
                    sha256(terrain_path(record['source'])) != record['terrain_hash'] or
                    sha256(record['labels']) != record['labels_hash']):
                    raise ValueError('source changed after planning')
                first, last = record['source_frames']
                prepared = solver.prepare(record, task['solve_variant'], first, last)
                if device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(device)
                motion, stats = solver.solve(prepared, Path(root)/'checkpoints'/(task['id']+'.npz'), task['id'])
                # Reuse the solved trajectory, but retain the intent of EVERY source terrain.
                variants = {task['solve_variant']: prepared}
                for v in record['unique_variants']:
                    if v not in variants:
                        variants[v] = solver.prepare(record, v, first, last)
                intents = {k: np.stack([variants[v][k] for v in record['variant_aliases']])
                           for k in ('contact', 'offsets', 'normals', 'anchors', 'surfaces', 'foot_target')}
                dest = Path(root)/'candidates'/(record['key']+'_motion.npz')
                atomic_npz(dest, joint_pos=motion, joint_names=np.array(solver.dst.joint_names), fps=[50],
                           robot_metadata=json.dumps(metadata), data_version=VERSION, source=record['source'],
                           source_frames=record['source_frames'], task_id=task['id'])
                terrain_dest = Path(terrain_path(str(dest)))
                temp = terrain_dest.with_name(terrain_dest.name+'.tmp')
                temp.write_bytes(Path(terrain_path(record['source'])).read_bytes())
                os.replace(temp, terrain_dest)
                atomic_npz(dest.with_name(dest.name.replace('_motion.npz', '_intent.npz')),
                           **intents, body_names=np.array(list(solver.dst.contact_bodies)),
                           segments=json.dumps({str(v): p['segments'] for v, p in variants.items()}),
                           incidental_segments=json.dumps({str(v): p['incidental_segments'] for v, p in variants.items()}),
                           contact_aliases=json.dumps({str(v): p['contact_aliases'] for v, p in variants.items()}),
                           source_contact_validation=json.dumps({str(v): p['source_contact_validation'] for v, p in variants.items()}),
                           source_contact_validation_policy=json.dumps(SOURCE_CONTACT_POLICY),
                           contact_patch_policy=json.dumps(PATCH_POLICY),
                           contact_intent_policy=json.dumps(CONTACT_INTENT_POLICY),
                           source_variants=np.arange(record['variants']), task_id=task['id'])
                events.put(('solved', task['id'], dict(device=device, worker_pid=os.getpid(), windows=stats,
                            solve_seconds=time.monotonic()-started, candidate_files=output_hashes(root, task))))
            except Exception:
                events.put(('solve_failed', task['id'], traceback.format_exc()))
    except BaseException:
        events.put(('worker_failed', device, traceback.format_exc()))
        raise


def aggregate(root, plan, states):
    counts = {s: 0 for s in ('pending', 'solving', 'solved', 'checking', *TERMINAL)}
    accepted = []
    variants_done, variants_pass, frames_done = 0, 0, 0
    for task in plan['tasks']:
        state = states[task['id']]
        counts[state['status']] += 1
        if state['status'] in ('accepted', 'rejected'):
            variants_done += task['record']['variants']
            frames_done += task['record']['frames']
            variants_pass += len(state['accepted_variants'])
        if state['status'] == 'accepted':
            accepted.append(dict(task_id=task['id'], key=task['record']['key'], family=task['record']['family'],
                                 source_variants=state['accepted_variants'], files=state['exported']))
    # PHP's primitive motion sources are shared across parameterized clips. No source mocap
    # family map is distributed with these files, so don't invent independent test families.
    splits = dict(seed=0, requested_ratios=[.8, .1, .1], train=[], validation=[], test=[],
                  development=[r['key'] for r in accepted],
                  families={r['key']: r['family'] for r in accepted},
                  status='development_only: primitive source family provenance is insufficient for independent evaluation')
    atomic_json(Path(root)/'splits.json', splits)
    summary = dict(run_id=plan['run_id'], identity=plan['identity'], counts=counts,
                   sources=len(plan['tasks']), converted_sources=counts['accepted']+counts['rejected'],
                   checked_variants=variants_done, accepted_variants=variants_pass, converted_frames=frames_done,
                   expected_variants=sum(t['record']['variants'] for t in plan['tasks']),
                   all_terminal=all(states[t['id']]['status'] in TERMINAL for t in plan['tasks']),
                   accepted=accepted, status='running', robot_metadata=plan['robot_metadata'],
                   collision_policy=COLLISION_POLICY, fidelity_policy=plan.get('fidelity_policy'),
                   posture_policy=plan.get('posture_policy'), pd_status='not_run', learned_tracking_status='not_run')
    if summary['all_terminal']:
        summary['status'] = 'completed_with_failures' if counts['failed'] else 'completed'
    atomic_json(Path(root)/'manifest.json', summary)
    if accepted:
        atomic_json(Path(root)/'accepted'/'manifest.json', dict(run_id=plan['run_id'], parent='../manifest.json',
                    robot_metadata=plan['robot_metadata'], collision_policy=COLLISION_POLICY, clips=accepted,
                    split_manifest='../splits.json', policy='all exported variants pass every declared independent quality gate',
                    fidelity_policy=plan.get('fidelity_policy'), posture_policy=plan.get('posture_policy')))
    return summary


def execute(root, plan, args):
    states, pending = {}, []
    for task in plan['tasks']:
        path = root/'tasks'/(task['id']+'.json')
        state = load_json(path) if path.exists() else dict(task_id=task['id'], status='pending', attempts=0)
        if state['status'] in ('accepted', 'rejected', 'solved', 'checking'):
            verify_task(root, state)
        if state['status'] == 'checking':
            state['status'] = 'solved'
        if state['status'] == 'solving' or (state['status'] == 'failed' and args.retry_failed):
            state['status'] = 'pending'
        states[task['id']] = state
        if state['status'] == 'pending':
            pending.append(task)
    ctx = mp.get_context('spawn')
    task_queue, events = ctx.Queue(), ctx.Queue()
    # Longest first reduces the tail while keeping task identity independent of scheduling.
    for task in sorted(pending, key=lambda t: t['record']['frames'], reverse=True):
        task_queue.put(task)
    workers = []
    if pending:
        for device in args.devices.split(','):
            device = device if device == 'cpu' or device.startswith('cuda:') else 'cuda:'+device
            for _ in range(args.workers_per_gpu):
                proc = ctx.Process(target=gpu_worker, args=(device, plan['solver'], task_queue, events, str(root)))
                proc.start()
                workers.append(proc)
    tasks_by_id = {t['id']: t for t in plan['tasks']}
    futures, finished_solves, stopped_workers = {}, 0, False
    pool = ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=ctx)
    active_workers = {}

    def save(task_id):
        atomic_json(root/'tasks'/(task_id+'.json'), states[task_id])

    def check(task_id):
        states[task_id]['status'] = 'checking'
        save(task_id)
        futures[pool.submit(evaluate_task, tasks_by_id[task_id], str(root))] = task_id

    try:
        for task_id, state in states.items():
            if state['status'] == 'solved':
                check(task_id)
        last_report = 0
        while True:
            try:
                event, task_id, payload = events.get(timeout=.5)
                if event == 'ready':
                    print(f'GPU worker ready: {task_id} pid={payload}', flush=True)
                elif event == 'worker_failed':
                    raise RuntimeError(f'{task_id}: {payload}')
                elif event == 'started':
                    states[task_id].update(status='solving', device=payload,
                                          attempts=states[task_id]['attempts']+1)
                    active_workers[task_id] = payload
                    save(task_id)
                elif event == 'solved':
                    finished_solves += 1
                    active_workers.pop(task_id, None)
                    states[task_id].update(payload, status='solved')
                    save(task_id)
                    check(task_id)
                elif event == 'solve_failed':
                    active_workers.pop(task_id, None)
                    errors = states[task_id].setdefault('errors', [])
                    errors.append(dict(stage='solve', traceback=payload))
                    if states[task_id]['attempts'] < args.max_attempts:
                        states[task_id]['status'] = 'pending'
                        task_queue.put(tasks_by_id[task_id])
                    else:
                        finished_solves += 1
                        states[task_id]['status'] = 'failed'
                    save(task_id)
            except queue.Empty:
                pass
            for future in list(futures):
                if not future.done():
                    continue
                task_id = futures.pop(future)
                try:
                    states[task_id].update(future.result())
                except Exception:
                    states[task_id].update(status='failed')
                    states[task_id].setdefault('errors', []).append(dict(stage='assessment', traceback=traceback.format_exc()))
                save(task_id)
                print(f"{states[task_id]['status']}: {tasks_by_id[task_id]['record']['key']}", flush=True)
            if finished_solves >= len(pending) and not stopped_workers:
                for _ in workers:
                    task_queue.put(None)
                stopped_workers = True
            for proc in workers:
                if proc.exitcode is not None and proc.exitcode != 0:
                    raise RuntimeError(f'GPU worker pid {proc.pid} exited {proc.exitcode}; resume the same run after diagnosis')
            if time.monotonic()-last_report >= 15:
                summary = aggregate(root, plan, states)
                print(json.dumps(summary['counts']), flush=True)
                last_report = time.monotonic()
            if finished_solves >= len(pending) and not futures:
                break
        for proc in workers:
            proc.join(timeout=30)
            if proc.is_alive() or proc.exitcode:
                raise RuntimeError(f'GPU worker {proc.pid} did not exit cleanly')
    finally:
        interrupted = sys.exc_info()[1]
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        pool.shutdown(wait=True, cancel_futures=True)
        summary = aggregate(root, plan, states)
        if interrupted is not None:
            summary.update(status='interrupted', interruption_reason=str(interrupted))
            atomic_json(root/'manifest.json', summary)
    for state in states.values():
        if state['status'] in ('accepted', 'rejected'):
            verify_task(root, state)
    return aggregate(root, plan, states)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--polish-seeds', type=Path, help='Refine committed candidates with CPU workers through the normal protected controller entry')
    ap.add_argument('--source', default='data/php_parkour')
    ap.add_argument('--contacts', default='data/php_parkour_contacts')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--devices', default='0,1')
    ap.add_argument('--workers-per-gpu', type=int, default=1)
    ap.add_argument('--cpu-workers', type=int, default=12)
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--iterations', type=int, default=1000)
    ap.add_argument('--window-frames', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=48)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--collision-rounds', type=int, default=12)
    ap.add_argument('--collision-iterations', type=int, default=400)
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--selections', type=Path, help='JSON list: source_key, key, start, end, solve_variant (optional)')
    ap.add_argument('--solve-variant', type=int, default=0)
    args = ap.parse_args(argv)
    if args.polish_seeds is not None:
        if args.workers_per_gpu != 0 or args.cpu_workers < 1:
            ap.error('native seed refinement requires --workers-per-gpu 0 and positive --cpu-workers')
        if args.selections or args.only or args.retry_failed:
            ap.error('native seed refinement uses --polish-seeds, not selection or retry flags')
        from tools.polish_php_conversion import main as polish_main
        forwarded=['--seeds',str(args.polish_seeds),'--out',str(args.out),'--workers',str(args.cpu_workers)]
        if args.scan:forwarded.append('--scan')
        if args.resume:forwarded.append('--resume')
        return polish_main(forwarded)
    if min(args.workers_per_gpu, args.cpu_workers, args.max_attempts) < 1:
        ap.error('worker and attempt counts must be positive')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'controller.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another controller holds this run directory')
    records = discover(args.source, args.contacts, args.only)
    for record in records:
        record['source_frames'] = [0, record['frames']]
        # Conservative family grouping: parameterized source clips reuse primitive motions.
        record['family'] = 'php_locomotion' if 'locomotion' in record['category'] else 'php_'+re.split(r'[0-9]', Path(record['key']).name)[0].rstrip('_')
    if args.selections:
        by_key = {r['key']: r for r in records}
        selected = []
        for row in load_json(args.selections):
            record = dict(by_key[row['source_key']])
            a, b = row.get('start', 0), row.get('end') or record['frames']
            if not 0 <= a < b <= record['frames'] or b-a < 3:
                raise ValueError('selection has an invalid frame range')
            record.update(key=row.get('key', record['key']), source_frames=[a, b], frames=b-a,
                          solve_variant=row.get('solve_variant', args.solve_variant))
            selected.append(record)
        records = selected
    if len({r['key'] for r in records}) != len(records):
        raise ValueError('duplicate output keys')
    for record in records:
        key = Path(record['key'])
        if key.is_absolute() or '..' in key.parts or len(key.parts) != 2:
            raise ValueError('output keys must be category/name without path traversal')
        if not 0 <= record.get('solve_variant', args.solve_variant) < record['variants']:
            raise ValueError('solve variant out of range')
    config = SolverConfig(iterations=args.iterations, window_frames=args.window_frames, overlap=args.overlap, graph=not args.no_graph,
                          collision_rounds=args.collision_rounds, collision_iterations=args.collision_iterations)
    metadata = get_robot('tienkung3_29dof').metadata()
    identity = dict(version=VERSION, solver=asdict(config), code=source_fingerprint(), robot_metadata=metadata,
                    source_robot_files={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'data/g1_description').rglob('*')) if p.is_file()},
                    collision_policy=COLLISION_POLICY, fidelity_policy=FIDELITY_POLICY, posture_policy=POSTURE_POLICY,
                    contact_intent_policy=CONTACT_INTENT_POLICY, contact_patch_policy=PATCH_POLICY,
                    source_contact_validation_policy=SOURCE_CONTACT_POLICY, root_continuity_policy=ROOT_POLICY)
    tasks = []
    for record in records:
        variant = record.get('solve_variant', args.solve_variant)
        stable_record = {k: v for k, v in record.items() if k not in ('source', 'labels')}
        tasks.append(dict(id=digest([identity, stable_record, variant])[:24], record=record, solve_variant=variant))
    plan = dict(run_id=root.name, identity=digest([identity, [t['id'] for t in tasks]]), **identity,
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                command=sys.argv, tasks=tasks, execution=dict(devices=args.devices, workers_per_gpu=args.workers_per_gpu,
                                                            cpu_workers=args.cpu_workers, simulation_environments=0))
    path = root/'plan.json'
    if path.exists():
        previous = load_json(path)
        if not args.resume:
            raise ValueError('run directory exists; use --resume with unchanged code/inputs/configuration')
        if previous['identity'] != plan['identity']:
            raise ValueError('run identity changed (code, inputs, model, policy or solver); choose a new output directory')
        plan = previous
    else:
        atomic_json(path, plan)
    if args.scan:
        print(json.dumps(dict(sources=len(tasks), frames=sum(r['frames'] for r in records),
                              terrain_slots=sum(r['variants'] for r in records), plan=str(path))), flush=True)
        return
    atomic_json(root/'collision_policy.json', collision_evidence())
    summary = execute(root, plan, args)
    print(json.dumps({k: v for k, v in summary.items() if k != 'accepted'}), flush=True)
    if summary['counts']['failed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
