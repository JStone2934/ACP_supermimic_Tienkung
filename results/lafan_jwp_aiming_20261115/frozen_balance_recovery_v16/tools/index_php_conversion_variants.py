"""Index verified accepted variants without overwriting earlier accepted trajectories.

Runs and immutable assessment-only recoveries are ordered by preference.
This is an asset index, not a MotionLib dataset:
each entry names its parent motion, labels and exported terrain slot explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from supermimic.acp.motion import load_php_npz
from supermimic.acp.php_conversion import atomic_json
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.terrain import terrain_path
from supermimic.acp.retarget_fidelity import POSTURE_POLICY, posture_required, verify_fidelity_report
from tools.audit_php_conversion import check_labels
from tools.convert_php import load_json, verify_task


def read_parent(root, allow_running=False):
    """Read a conversion run or a provenance-bound assessment-only recovery."""
    if (root/'plan.json').exists():
        plan_bytes = (root/'plan.json').read_bytes()
        manifest_bytes = (root/'manifest.json').read_bytes()
        plan, manifest = json.loads(plan_bytes), json.loads(manifest_bytes)
        if not manifest['all_terminal'] and not allow_running:
            raise ValueError('cannot index an unfinished run')
        receipt = dict(root=str(root), kind='conversion_run', plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                       manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), status=manifest['status'],
                       all_terminal=manifest['all_terminal'])
        return plan, manifest, receipt, None
    provenance_bytes = (root/'provenance.json').read_bytes()
    result_bytes = (root/'result.json').read_bytes()
    provenance, result = json.loads(provenance_bytes), json.loads(result_bytes)
    if provenance.get('kind') != 'assessment_only' or provenance.get('motion_changed') is not False:
        raise ValueError('recovery must preserve its original candidate motion')
    parent = Path(provenance['parent'])
    parent_plan_bytes = (parent/'plan.json').read_bytes()
    if hashlib.sha256(parent_plan_bytes).hexdigest() != provenance['parent_plan_sha256']:
        raise ValueError('recovery parent plan changed')
    parent_plan = json.loads(parent_plan_bytes)
    task = provenance['task']
    if next((t for t in parent_plan['tasks'] if t['id'] == task['id']), None) != task:
        raise ValueError('recovery task differs from its parent plan')
    original = provenance['original_state']
    if original['task_id'] != task['id'] or original['status'] != 'failed' or not original.get('candidate_files'):
        raise ValueError('recovery does not identify a failed original candidate')
    parent_state = load_json(parent/'tasks'/(task['id']+'.json'))
    if parent_state['candidate_files'] != original['candidate_files']:
        raise ValueError('recovery candidate identity differs from parent')
    for relative, checksum in original['candidate_files'].items():
        if sha256(parent/relative) != checksum or sha256(root/relative) != checksum:
            raise ValueError('recovery changed an original candidate asset')
    if not provenance.get('assessment_code'):
        raise ValueError('recovery has no assessment code provenance')
    for relative, checksum in provenance['assessment_code'].items():
        if sha256(root/'source_snapshot'/relative) != checksum:
            raise ValueError('recovery assessment code changed')
    if result['status'] not in ('accepted', 'rejected'):
        raise ValueError('recovery assessment is unfinished')
    state = dict(original, **result)
    plan = dict(parent_plan, tasks=[task])
    manifest = dict(all_terminal=True, status='assessment_complete')
    receipt = dict(root=str(root), kind='assessment_only', parent=str(parent),
                   parent_plan_sha256=provenance['parent_plan_sha256'],
                   provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
                   result_sha256=hashlib.sha256(result_bytes).hexdigest(),
                   all_terminal=True, status=result['status'])
    return plan, manifest, receipt, {task['id']: state}


def build_index(roots, *, allow_running=False, require_root_continuity=False):
    from supermimic.acp.root_continuity import POLICY as ROOT_POLICY, score_root_files, required as root_required
    root_reports = {}
    chosen, source_identities, parents = {}, {}, []
    robot = get_robot('tienkung3_29dof')
    metadata = policy = fidelity_policy = posture_policy = None
    for root in map(lambda p: Path(p).resolve(), roots):
        plan, manifest, parent_receipt, recovered_states = read_parent(root, allow_running)
        robot.validate_metadata(plan['robot_metadata'])
        if metadata is not None and (metadata != plan['robot_metadata'] or policy != plan['collision_policy'] or
                                     fidelity_policy != plan.get('fidelity_policy') or
                                     posture_policy != plan.get('posture_policy')):
            raise ValueError('parent runs use different models, collision, fidelity or posture policies')
        metadata, policy = plan['robot_metadata'], plan['collision_policy']
        fidelity_policy = plan.get('fidelity_policy')
        posture_policy = plan.get('posture_policy')
        if posture_required(plan['version']) and posture_policy != POSTURE_POLICY:
            raise ValueError('current solver requires posture acceptance')
        parents.append(parent_receipt)
        for task in plan['tasks']:
            record = task['record']
            key = record['key']
            source_path = Path(record['source'])
            source_key = record.get('source_key', source_path.parent.name+'/'+source_path.name.removesuffix('_motion.npz'))
            identity = {k: record[k] for k in ('source_hash', 'terrain_hash', 'labels_hash',
                                              'source_frames', 'variants')}
            if source_key in source_identities and source_identities[source_key] != identity:
                raise ValueError(f'parent source identity differs: {source_key}')
            source_identities[source_key] = identity
            state_path = root/'tasks'/(task['id']+'.json')
            if recovered_states is None and not state_path.exists() and allow_running and not manifest['all_terminal']:
                continue
            state = recovered_states[task['id']] if recovered_states is not None else load_json(state_path)
            if state['task_id'] != task['id']:
                raise ValueError('task state belongs to another task')
            if state['status'] != 'accepted':
                continue
            for path, expected in ((record['source'], record['source_hash']),
                                   (terrain_path(record['source']), record['terrain_hash']),
                                   (record['labels'], record['labels_hash'])):
                if sha256(path) != expected:
                    raise ValueError('original input changed since conversion')
            verify_task(root, state)
            quality = root/state['quality_path']
            report = load_json(quality)
            if report['task_id'] != task['id'] or report['source'] != record:
                raise ValueError('quality report belongs to another task/source')
            candidate = root/'candidates'/(key+'_motion.npz')
            version = plan['version'].removeprefix('php-segment-v')
            if plan['version'].startswith('php-segment-v') and version.isdigit() and int(version) >= 18:
                from supermimic.acp.source_contacts import SOURCE_CONTACT_POLICY, verify_source_receipts
                if (plan.get('source_contact_validation_policy') != SOURCE_CONTACT_POLICY or
                    report.get('source_contact_validation_policy') != SOURCE_CONTACT_POLICY or
                    report.get('source_contact_validation_verified') is not True):
                    raise ValueError('missing source contact validation')
                with np.load(str(candidate).replace('_motion.npz', '_intent.npz')) as intent:
                    if json.loads(str(intent['source_contact_validation_policy'].item())) != SOURCE_CONTACT_POLICY:
                        raise ValueError('source contact policy differs in intent')
                    verify_source_receipts(record, json.loads(str(intent['source_contact_validation'].item())))
            p, _, _, fps = load_php_npz(candidate, robot)
            if len(p) != record['frames'] or len(p) != record['source_frames'][1]-record['source_frames'][0] or fps != 50:
                raise ValueError('candidate frame range or sampling rate differs')
            source_terrain = np.load(terrain_path(record['source']))
            if not np.array_equal(source_terrain, np.load(terrain_path(str(candidate)))):
                raise ValueError('candidate terrain differs from original')
            if [v['variant'] for v in report['variants']] != list(range(record['variants'])):
                raise ValueError('incomplete or misordered original variant reports')
            for field, path in (('motion_hash', candidate), ('terrain_hash', terrain_path(str(candidate))),
                                ('intent_hash', str(candidate).replace('_motion.npz', '_intent.npz'))):
                if report[field] != sha256(path):
                    raise ValueError('quality report does not bind its candidate: '+field)
            if fidelity_policy is not None:
                verify_fidelity_report(report, fidelity_policy, posture_policy)
            approved = [v['variant'] for v in report['variants'] if v.get('quality_pass', v['geometry_pass'])]
            if approved != state['accepted_variants'] or approved != report['accepted_variants']:
                raise ValueError('quality and accepted state disagree')
            motion = root/'accepted'/(key+'_motion.npz')
            labels = root/'accepted_contacts'/(key+'.npz')
            if sha256(motion) != sha256(candidate):
                raise ValueError('exported motion differs from the checked candidate')
            if not np.array_equal(np.load(terrain_path(str(motion))), source_terrain[:, approved]):
                raise ValueError('exported terrain has unchecked or misordered variants')
            check_labels(labels, motion, quality, len(approved), robot)
            with np.load(labels) as z:
                if z['source_variants'].tolist() != approved or not z['quality_pass'].all():
                    raise ValueError('accepted label variant mapping differs')
            root_key = None
            if require_root_continuity or root_required(plan['version']):
                measured = score_root_files(record, candidate)
                if root_required(plan['version']):
                    if plan.get('root_continuity_policy') != ROOT_POLICY or report.get('root_continuity_policy') != ROOT_POLICY or report.get('root_continuity') != measured:
                        raise ValueError('missing or stale required root continuity evidence')
                root_key = measured['source_sha256']+':'+measured['motion_sha256']
                root_reports[root_key] = measured
                if not measured['passed']:
                    continue
            for slot, variant in enumerate(approved):
                chosen.setdefault((source_key, variant), dict(source_key=source_key, output_key=key, source_variant=variant,
                    export_slot=slot, parent=str(root), task_id=task['id'], root_continuity_key=root_key,
                    motion=str(motion), terrain=str(motion).replace('_motion.npz', '_terrain.npy'),
                    labels=str(labels), quality=str(quality),
                    motion_sha256=sha256(motion), labels_sha256=sha256(labels), quality_sha256=sha256(quality)))
    return dict(kind='accepted_variant_asset_index', directly_loadable_by_motionlib=False,
                snapshot_of_running_parents=any(not p['all_terminal'] for p in parents),
                preference='first parent with a passing variant; never replace an earlier passing variant',
                parents=parents, robot_metadata=metadata, collision_policy=policy, fidelity_policy=fidelity_policy,
                posture_policy=posture_policy,
                root_continuity_policy=ROOT_POLICY if require_root_continuity else None,
                root_continuity_reports=root_reports,
                source_identities=source_identities,
                accepted_sources=len({key for key, _ in chosen}), accepted_variants=len(chosen),
                entries=[chosen[k] for k in sorted(chosen)],
                pd_status='not_run', learned_tracking_status='not_run')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allow-running', action='store_true',
                    help='index only committed accepted tasks of running parents')
    ap.add_argument('--require-root-continuity', action='store_true', help='independently recompute full-frame root derivatives and exclude discontinuous candidates')
    args = ap.parse_args()
    result = build_index(args.runs, allow_running=args.allow_running, require_root_continuity=args.require_root_continuity)
    atomic_json(args.out, result)
    print({k: result[k] for k in ('accepted_sources', 'accepted_variants')})


if __name__ == '__main__':
    main()
