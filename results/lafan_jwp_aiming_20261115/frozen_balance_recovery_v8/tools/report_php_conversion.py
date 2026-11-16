"""Produce a compact, auditable delivery report from completed conversion tasks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from supermimic.acp.php_conversion import atomic_json
from tools.audit_php_conversion import audit
from tools.convert_php import load_json


def report(root, inventory=None):
    root = Path(root).resolve()
    verified = audit(root, inventory)
    plan = load_json(root / 'plan.json')
    categories = defaultdict(Counter)
    reasons = Counter()
    clips = []
    for task in plan['tasks']:
        record = task['record']
        state = load_json(root / 'tasks' / (task['id'] + '.json'))
        quality = load_json(root / state['quality_path'])
        category = categories[record['category']]
        category['sources'] += 1
        category['frames'] += record['frames']
        category[state['status']] += 1
        category['variants'] += record['variants']
        category['accepted_variants'] += len(state['accepted_variants'])
        for variant in quality['variants']:
            reasons.update(variant['rejection_reasons'])
        clips.append(dict(key=record['key'], source=record['source'], source_frames=record['source_frames'],
                          status=state['status'], accepted_variants=state['accepted_variants'],
                          quality_path=state['quality_path'],
                          solve_seconds=state['solve_seconds'], assessment_seconds=state['assessment_seconds'],
                          variants=[{k: v[k] for k in ('variant', 'geometry_pass', 'strict_raw_geometry_pass',
                                    'quality_pass', 'fidelity_pass', 'fidelity', 'posture_pass', 'posture',
                                    'rejection_reasons', 'terrain_penetration_max_m',
                                    'self_penetration_policy_max_m', 'joint_velocity_max_ratio', 'contacts') if k in v}
                                    for v in quality['variants']]))
    return dict(audit=verified, root=str(root), categories=dict(categories),
                rejected_variant_reason_counts=dict(reasons.most_common()), clips=clips)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--inventory', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = report(args.root, args.inventory)
    atomic_json(args.out, result)
    print(result['audit'])
    print(result['categories'])


if __name__ == '__main__':
    main()
