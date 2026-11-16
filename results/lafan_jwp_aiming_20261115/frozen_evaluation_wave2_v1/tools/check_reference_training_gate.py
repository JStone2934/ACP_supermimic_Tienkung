"""Strict reference-tracking gate. Passing does not certify standing or aiming."""
import argparse
import json
import math
from pathlib import Path
from tools.prepare_lafan_jwp_training import sha


def check(report, manifest, checkpoint):
    if report.get('checkpoint_sha256') != sha(checkpoint):
        raise ValueError('Evaluation checkpoint identity mismatch')
    expected = {row['key']: int(row['frames']) - 1 for row in manifest['entries']}
    measured = report['per_clip']
    if not expected or set(expected) != set(measured):
        raise ValueError('Evaluation does not cover exactly the held-out dataset')
    good = []
    failures = {}
    for key, steps in expected.items():
        row = measured[key]
        reasons = []
        if row.get('expected_steps') != steps:
            reasons.append('reference_length_mismatch')
        if row.get('alive_steps') != steps or row.get('failed', True):
            reasons.append('incomplete_or_fallen')
        if row.get('finite') is not True:
            reasons.append('nonfinite_or_unverified_state')
        for metric, limit in [('root_rmse_m', .1), ('joint_rmse_rad', .2)]:
            value = row.get(metric, float('nan'))
            if not math.isfinite(value) or value > limit:
                reasons.append(metric)
        if reasons:
            failures[key] = reasons
        else:
            good.append(key)
    fraction = len(good) / len(expected)
    return dict(scope='Held-out reference tracking only; standing and aiming require separate physical gates',
                passed=fraction >= .9, successful_clips=len(good), clips=len(expected),
                successful_fraction=fraction, required_fraction=.9, failures=failures,
                checkpoint_sha256=report['checkpoint_sha256'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    result = check(json.loads(args.report.read_text()), json.loads(args.manifest.read_text()), args.checkpoint)
    result.update(report_sha256=sha(args.report), manifest_sha256=sha(args.manifest))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
