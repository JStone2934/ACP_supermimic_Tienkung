"""Verify all ORIGINAL source/terrain slots across versioned specialized solves.

Aliases retain their source numbering. Extra retries never increase coverage;
only a distinct original pair passing geometry AND fidelity fills a slot.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from supermimic.acp.php_conversion import atomic_json
from supermimic.acp.retarget_fidelity import POLICY, POSTURE_POLICY
from tools.convert_php import load_json
from tools.index_php_conversion_variants import build_index


def coverage(index, inventory, *, require_root_continuity=False):
    from supermimic.acp.root_continuity import POLICY as ROOT_POLICY, verify_report
    if require_root_continuity:
        if index.get('root_continuity_policy') != ROOT_POLICY:
            raise ValueError('final coverage requires independent root continuity acceptance')
        for entry in index['entries']:
            measured = index.get('root_continuity_reports', {}).get(entry.get('root_continuity_key'))
            if not measured or not measured['passed'] or measured['policy'] != ROOT_POLICY:
                raise ValueError('accepted entry lacks passing root continuity evidence')
            verify_report(measured)
            if measured['motion_sha256'] != entry['motion_sha256'] or measured['source_sha256'] != index['source_identities'][entry['source_key']]['source_hash']:
                raise ValueError('root continuity evidence belongs to another source/motion')
    if index.get('fidelity_policy') != POLICY:
        raise ValueError('all-variant acceptance requires the current arm fidelity policy')
    if index.get('posture_policy') != POSTURE_POLICY:
        raise ValueError('all-variant acceptance requires the current torso posture policy')
    expected, source_rows = set(), {}
    for row in inventory['records']:
        path = Path(row['source'])
        key = path.parent.name+'/'+path.name.removesuffix('_motion.npz')
        if key in source_rows:
            raise ValueError('duplicate source in inventory')
        source_rows[key] = row
        expected.update((key, v) for v in range(row['variants']))
    if inventory.get('source_count', len(source_rows)) != len(source_rows) or inventory.get('terrain_slots', len(expected)) != len(expected):
        raise ValueError('inventory totals disagree with its source rows')
    for key, identity in index['source_identities'].items():
        row = source_rows[key]
        required = dict(source_hash=row['source_sha256'], terrain_hash=row['terrain_sha256'],
                        labels_hash=row['labels_sha256'], source_frames=[0, row['frames']], variants=row['variants'])
        if identity != required:
            raise ValueError(f'{key}: input/frames differ from the full original inventory')
    approved = {(e['source_key'], e['source_variant']) for e in index['entries']}
    if any(key not in index['source_identities'] for key, _ in approved):
        raise ValueError('accepted pair has no verified source identity')
    if len(approved) != len(index['entries']) or approved-expected:
        raise ValueError('duplicate or invented original terrain variant')
    missing = expected-approved
    per_source = [dict(source_key=key, expected_variants=row['variants'],
                       accepted_variants=[v for v in range(row['variants']) if (key, v) in approved],
                       missing_variants=[v for v in range(row['variants']) if (key, v) in missing])
                  for key, row in sorted(source_rows.items())]
    return dict(all_original_variants_pass=not missing, expected_sources=len(source_rows),
                expected_variants=len(expected), accepted_variants=len(approved), missing_variants=len(missing),
                fully_accepted_sources=sum(not r['missing_variants'] for r in per_source),
                fidelity_policy=POLICY, posture_policy=POSTURE_POLICY,
                root_continuity_policy=index.get('root_continuity_policy'),
                root_continuity_required=require_root_continuity, sources=per_source)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--inventory', type=Path, default=Path('docs/php_conversion/evidence/source_inventory.json'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--require-all', action='store_true')
    ap.add_argument('--allow-running', action='store_true',
                    help='audit committed accepted tasks while parents run; never count pending/checking tasks')
    ap.add_argument('--require-root-continuity', action='store_true', help='required for final delivery; also gate independently recomputed root derivatives')
    args = ap.parse_args()
    index = build_index(args.runs, allow_running=args.allow_running, require_root_continuity=args.require_root_continuity)
    result = coverage(index, load_json(args.inventory), require_root_continuity=args.require_root_continuity)
    index_path = args.out.with_suffix('.index.json')
    atomic_json(index_path, index)
    result['index'] = str(index_path.resolve())
    result['snapshot_of_running_parents'] = index['snapshot_of_running_parents']
    atomic_json(args.out, result)
    print({k: v for k, v in result.items() if k not in ('sources', 'fidelity_policy')})
    if args.require_all and not result['all_original_variants_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
