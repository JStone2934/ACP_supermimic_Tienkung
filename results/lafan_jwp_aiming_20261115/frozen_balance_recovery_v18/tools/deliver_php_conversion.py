"""Assemble a read-only delivery from a full run and separately identified failed-task retries.

Original attempts remain immutable. Only failed tasks may be replaced; every
replacement must cover the identical source, frames, terrain, labels and model.
Each imported task retains its original ID and its parent plan/configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from supermimic.acp.php_conversion import atomic_json, digest
from supermimic.acp.robots import sha256
from tools.audit_php_conversion import audit
from tools.convert_php import aggregate, load_json, verify_task


def assemble(base, retries, out, inventory):
    base, out = Path(base).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('choose a new delivery directory')
    roots = [base] + [Path(r).resolve() for r in retries]
    if len({r.name for r in roots}) != len(roots):
        raise ValueError('parent run names must be unique')
    plans = [load_json(r/'plan.json') for r in roots]
    if any(not load_json(r/'manifest.json')['all_terminal'] for r in roots):
        raise ValueError('all parent runs must be terminal before delivery')
    chosen, attempts = {}, []
    for index, (root, plan) in enumerate(zip(roots, plans)):
        for key in ('robot_metadata', 'collision_policy', 'code', 'source_robot_files', 'version'):
            if plan[key] != plans[0][key]:
                raise ValueError(f'retry changes {key}; this assembler only supports solver-configuration retries')
        for task in plan['tasks']:
            record = task['record']
            state = load_json(root/'tasks'/(task['id']+'.json'))
            key = record['key']
            attempts.append(dict(run=str(root), task_id=task['id'], key=key,
                                 status=state['status'], errors=state.get('errors', []),
                                 candidate_files=state.get('candidate_files', {})))
            if index:
                if key not in chosen:
                    raise ValueError(f'retry introduces an unplanned source: {key}')
                old_task, old_state, _ = chosen[key]
                if old_state['status'] != 'failed':
                    raise ValueError(f'only failed tasks may be replaced: {key}')
                if record != old_task['record'] or task['solve_variant'] != old_task['solve_variant']:
                    raise ValueError(f'retry changes input identity or frame range: {key}')
            chosen[key] = (task, state, root)
    for key, (task, state, root) in chosen.items():
        if state['status'] not in ('accepted', 'rejected'):
            raise ValueError(f'unresolved task: {key}')
        verify_task(root, state)
    if inventory:
        expected = {str(Path(r['source']).resolve()): (r['source_sha256'], [0, r['frames']])
                    for r in load_json(inventory)['records']}
        actual = {str(Path(t['record']['source']).resolve()):
                  (t['record']['source_hash'], t['record']['source_frames']) for t, _, _ in chosen.values()}
        if actual != expected:
            raise ValueError('delivery does not cover the full inventory at its original frame ranges')
    # Preflight above leaves no partial delivery for unresolved/changed inputs.
    out.mkdir(parents=True)
    origins = []
    for root, plan in zip(roots, plans):
        dest = out/'provenance'/root.name
        dest.mkdir(parents=True)
        for name in ('plan.json', 'manifest.json', 'collision_policy.json'):
            shutil.copy2(root/name, dest/name)
        for task in plan['tasks']:
            state_path = root/'tasks'/(task['id']+'.json')
            state = load_json(state_path)
            if state['status'] == 'failed':
                atomic_json(dest/'failed_tasks'/(task['id']+'.json'), state)
                if state.get('candidate_files'):
                    verify_task(root, state)
                    for relative in state['candidate_files']:
                        failed_dest = dest/'failed_artifacts'/relative
                        failed_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(root/relative, failed_dest)
        origins.append(dict(root=str(root), plan_sha256=sha256(root/'plan.json'),
                            manifest_sha256=sha256(root/'manifest.json'), solver=plan['solver']))
    tasks, states, imported = [], {}, {}
    for task, state, root in chosen.values():
        paths = set(state['candidate_files']) | set(state.get('exported', {}))
        paths.update([state['quality_path'], state['labels_path']])
        for relative in paths:
            dest = out/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/relative, dest)
        atomic_json(out/'tasks'/(task['id']+'.json'), state)
        tasks.append(task)
        states[task['id']] = state
        imported[task['id']] = dict(root=str(root), original_task_id=task['id'])
    identity = digest(dict(parents=origins, imported=imported))
    plan = dict(plans[0], run_id=out.name, identity=identity, tasks=tasks,
                kind='immutable_delivery', solver={'per_task': 'see provenance parent plans'},
                parents=origins, imported_tasks=imported,
                command=['tools.deliver_php_conversion'],
                assembly_inputs=dict(base=str(base), retries=[str(r) for r in roots[1:]],
                                     out=str(out), inventory=str(inventory) if inventory else None),
                assembly_code_hash=sha256(Path(__file__)),
                execution={'simulation_environments': 0, 'operation': 'verified artifact import; no new solve'})
    atomic_json(out/'plan.json', plan)
    atomic_json(out/'attempts.json', attempts)
    shutil.copy2(base/'collision_policy.json', out/'collision_policy.json')
    if (base/'source_snapshot').exists():
        shutil.copytree(base/'source_snapshot', out/'source_snapshot')
    (out/'source_snapshot'/'tools').mkdir(parents=True, exist_ok=True)
    for name in ('deliver_php_conversion.py', 'audit_php_conversion.py', 'report_php_conversion.py'):
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/'tools'/name)
    aggregate(out, plan, states)
    verified = audit(out, inventory)
    atomic_json(out/'audit.json', verified)
    return verified


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base', type=Path)
    ap.add_argument('--retries', nargs='*', type=Path, default=[])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    args = ap.parse_args()
    print(assemble(args.base, args.retries, args.out, args.inventory))


if __name__ == '__main__':
    main()
