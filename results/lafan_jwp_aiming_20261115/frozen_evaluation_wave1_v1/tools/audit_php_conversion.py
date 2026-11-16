"""Audit complete conversion coverage and all candidate/export/label identities from disk."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from supermimic.acp.source_contacts import SOURCE_CONTACT_POLICY, verify_source_receipts, source_validation_required
from supermimic.acp.motion import load_php_npz
from supermimic.acp.root_continuity import POLICY as ROOT_POLICY, required as root_required, score_root_files
from supermimic.acp.php_conversion import atomic_json
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.terrain import terrain_path
from tools.convert_php import load_json, verify_task


def check_labels(path, motion, quality, variants, robot):
    p, _, _, fps = load_php_npz(motion, robot)
    with np.load(path) as z:
        robot.validate_metadata(json.loads(str(z['robot_metadata'].item())))
        for key, target in (('motion_hash', motion), ('terrain_hash', terrain_path(str(motion))), ('quality_hash', quality)):
            if str(z[key].item()) != sha256(target):
                raise ValueError(f'{path}: stale {key}')
        if z['contact_dist'].shape != (variants, len(p), 9) or float(z['fps']) != fps:
            raise ValueError(f'{path}: wrong frame/variant/fps contract')
        if not np.isfinite(z['contact_dist']).all():
            raise ValueError(f'{path}: nonfinite distances')
        if not np.array_equal(z['contact_info'][..., 0].astype(bool), z['contact_dist'] <= z['threshold']):
            raise ValueError(f'{path}: contact flags disagree with distances')


def audit(root, inventory=None):
    root = Path(root).resolve()
    plan, summary = load_json(root/'plan.json'), load_json(root/'manifest.json')
    robot = get_robot('tienkung3_29dof')
    robot.validate_metadata(plan['robot_metadata'])
    if plan.get('kind') == 'native_pose_polish':
        from tools.polish_php_conversion import audit_seed_provenance
        audit_seed_provenance(root, plan)
    states, accepted_keys, sources, variants, frames = Counter(), set(), {}, 0, 0
    accepted_variants, checked_hashes, families = 0, 0, {}
    for task in plan['tasks']:
        record = task['record']
        state = load_json(root/'tasks'/(task['id']+'.json'))
        if state['task_id'] != task['id'] or state['status'] not in ('accepted', 'rejected'):
            raise ValueError(f"{record['key']}: conversion/assessment not complete ({state['status']})")
        for path, expected in ((record['source'], record['source_hash']),
                               (record['labels'], record['labels_hash']),
                               (terrain_path(record['source']), record['terrain_hash'])):
            if sha256(path) != expected:
                raise ValueError(f'source asset changed: {path}')
        verify_task(root, state)
        checked_hashes += len(state['candidate_files']) + len(state.get('exported', {})) + 2
        candidate = root/'candidates'/(record['key']+'_motion.npz')
        p, _, _, fps = load_php_npz(candidate, robot)
        if len(p) != record['frames'] or len(p) != record['source_frames'][1]-record['source_frames'][0] or fps != 50:
            raise ValueError('missing frames or incorrect sampling rate')
        source_terrain = np.load(terrain_path(record['source']))
        if not np.array_equal(source_terrain, np.load(terrain_path(str(candidate)))):
            raise ValueError('candidate terrain altered the source benchmark')
        report_path = root/state['quality_path']
        report = load_json(report_path)
        if report['task_id'] != task['id'] or len(report['variants']) != record['variants']:
            raise ValueError('incomplete variant quality coverage')
        for key, path in (('motion_hash', candidate), ('terrain_hash', terrain_path(str(candidate))),
                          ('intent_hash', str(candidate).replace('_motion.npz', '_intent.npz'))):
            if report[key] != sha256(path):
                raise ValueError(f'stale quality report: {key}')
        if source_validation_required(plan['version']):
            if plan.get('source_contact_validation_policy') != SOURCE_CONTACT_POLICY or report.get('source_contact_validation_policy') != SOURCE_CONTACT_POLICY or not report.get('source_contact_validation_verified'):
                raise ValueError('missing source contact validation')
            with np.load(str(candidate).replace('_motion.npz', '_intent.npz')) as intent:
                if json.loads(str(intent['source_contact_validation_policy'].item())) != SOURCE_CONTACT_POLICY:
                    raise ValueError('intent source validation policy mismatch')
                verify_source_receipts(record, json.loads(str(intent['source_contact_validation'].item())))
        if root_required(plan['version']):
            if plan.get('root_continuity_policy') != ROOT_POLICY or report.get('root_continuity_policy') != ROOT_POLICY:
                raise ValueError('missing required root continuity policy')
            if report.get('root_continuity') != score_root_files(record, candidate):
                raise ValueError('root continuity evidence differs from original trajectory')
        if plan['version'].startswith('php-segment-'):
            from supermimic.acp.retarget_fidelity import (POLICY, POSTURE_POLICY, posture_required,
                                                        verify_fidelity_report)
            if report.get('fidelity_policy') != POLICY:
                raise ValueError('missing or mismatched fidelity policy')
            expected_posture = POSTURE_POLICY if posture_required(plan['version']) else None
            if plan.get('posture_policy') != expected_posture:
                raise ValueError('missing or mismatched plan posture policy')
            verify_fidelity_report(report, POLICY, expected_posture)
        approved = [v for v, r in enumerate(report['variants']) if r.get('quality_pass', r['geometry_pass'])]
        if approved != state['accepted_variants'] or approved != report['accepted_variants']:
            raise ValueError('export status disagrees with per-variant quality')
        if bool(approved) != (state['status'] == 'accepted'):
            raise ValueError('incorrect terminal status')
        check_labels(root/state['labels_path'], candidate, report_path, record['variants'], robot)
        if approved:
            accepted_keys.add(record['key'])
            accepted_motion = root/'accepted'/(record['key']+'_motion.npz')
            if sha256(accepted_motion) != sha256(candidate):
                raise ValueError('exported motion differs from the checked trajectory')
            if not np.array_equal(np.load(terrain_path(str(accepted_motion))), source_terrain[:, approved]):
                raise ValueError('exported terrain has unchecked/misordered variants')
            accepted_label = root/'accepted_contacts'/(record['key']+'.npz')
            check_labels(accepted_label, accepted_motion, report_path, len(approved), robot)
            with np.load(accepted_label) as z:
                if z['source_variants'].tolist() != approved or not z['quality_pass'].all():
                    raise ValueError('accepted label variant mapping mismatch')
        states[state['status']] += 1
        variants += record['variants']
        accepted_variants += len(approved)
        frames += len(p)
        sources[str(Path(record['source']).resolve())] = (record['source_hash'], record['source_frames'])
        families[record['key']] = record['family']
    splits = load_json(root/'splits.json')
    lists = [set(splits[s]) for s in ('train', 'validation', 'test', 'development')]
    if set.union(*lists) != accepted_keys or sum(map(len, lists)) != len(accepted_keys):
        raise ValueError('split lists omit or duplicate accepted clips')
    family_sets = [{families[key] for key in keys} for keys in lists]
    for i, group in enumerate(family_sets):
        if any(group & other for other in family_sets[:i]):
            raise ValueError('source family leakage between splits')
    if inventory:
        original = load_json(inventory)
        expected = {str(Path(r['source']).resolve()): (r['source_sha256'], [0, r['frames']]) for r in original['records']}
        if sources != expected:
            raise ValueError('full-library inventory not completely converted at the original frame ranges')
    if (summary['converted_sources'] != sum(states.values()) or summary['checked_variants'] != variants or
        summary['converted_frames'] != frames or summary['accepted_variants'] != accepted_variants or
        summary['status'] != 'completed' or summary['counts']['failed']):
        raise ValueError('summary disagrees with authoritative per-task files')
    return dict(run_id=plan['run_id'], identity=plan['identity'], verified=True, sources=len(plan['tasks']),
                unique_source_files=len(sources), frames=frames, checked_variants=variants,
                accepted_variants=accepted_variants, statuses=dict(states), checked_output_hashes=checked_hashes,
                inventory=str(inventory) if inventory else None, full_library=bool(inventory),
                split_status=splits['status'], motion_time_s=sum((t['record']['frames']-1)/50 for t in plan['tasks']),
                pd_status='not_run', learned_tracking_status='not_run')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--inventory', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    report = audit(args.root, args.inventory)
    if args.out:
        atomic_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
