"""Measure signed sagittal lean using native FK and a shared source heading.

This is a diagnostic, not a replacement for conversion acceptance. Calibrate
each body's axes to its robot's neutral pose before comparing different robots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from supermimic.acp.motion import load_php_npz
from supermimic.acp.retarget_fidelity import native_poses, neutral_poses


def anatomical_rotations(robot, rotations):
    _, neutral = neutral_poses(robot)
    return rotations @ neutral.transpose(0, 2, 1)


def heading_from_root(rotations):
    heading = rotations[:, 0, :, 0].copy()
    heading[:, 2] = 0
    norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError('source root forward axis is vertical; sagittal heading undefined')
    return heading / norm


def lean_deg(rotations, heading):
    up = rotations[..., :, 2]
    forward = np.einsum('tbi,ti->tb', up, heading)
    return np.rad2deg(np.arctan2(forward, up[..., 2]))


def statistics(source, target):
    delta = (target-source+180) % 360-180
    return dict(source_mean_deg=float(source.mean()), target_mean_deg=float(target.mean()),
                added_forward_mean_deg=float(delta.mean()),
                error_rmse_deg=float(np.sqrt(np.mean(delta**2))),
                error_abs_p95_deg=float(np.percentile(np.abs(delta), 95)),
                added_forward_max_deg=float(delta.max()),
                error_abs_max_deg=float(np.abs(delta).max()))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    rows, series = [], []
    for clip in manifest['clips']:
        for path_key, hash_key in [('source_motion', 'source_motion_sha256'),
                                   ('converted_motion', 'converted_motion_sha256')]:
            if digest(clip[path_key]) != clip[hash_key]:
                raise ValueError('motion changed since comparison manifest: '+clip[path_key])
        a, b = clip['source_frame_range']
        sp, sq, sj, sfps = load_php_npz(clip['source_motion'], 'g1')
        tp, tq, tj, tfps = load_php_npz(clip['converted_motion'], 'tienkung3_29dof')
        if len(tp) != b-a or sfps != tfps:
            raise ValueError('posture comparison requires identical frames and fps')
        _, sr = native_poses('g1', sp[a:b], sq[a:b], sj[a:b])
        _, tr = native_poses('tienkung3_29dof', tp, tq, tj)
        sr = anatomical_rotations('g1', sr)
        tr = anatomical_rotations('tienkung3_29dof', tr)
        heading = heading_from_root(sr)
        source, target = lean_deg(sr, heading), lean_deg(tr, heading)
        row = dict(name=clip['name'], source_key=clip['source_key'], frames=b-a, fps=sfps,
                   source_motion=clip['source_motion'], target_motion=clip['converted_motion'],
                   source_sha256=digest(clip['source_motion']), target_sha256=digest(clip['converted_motion']))
        for name, idx in [('pelvis', 0), ('torso', 7)]:
            row[name] = statistics(source[:, idx], target[:, idx])
            n = min(round(sfps), len(source)//2)
            row[name]['first_second'] = statistics(source[:n, idx], target[:n, idx])
            row[name]['last_second'] = statistics(source[-n:, idx], target[-n:, idx])
        rows.append(row)
        series.append((clip['name'], np.arange(b-a)/sfps, source[:, 7], target[:, 7]))
    return dict(manifest=str(Path(manifest_path).resolve()), manifest_sha256=digest(manifest_path),
                method='Native MuJoCo FK; neutral-calibrated body up axis projected into source root heading/up plane; positive is forward; unchanged source frame timing',
                acceptance_status='diagnostic_only', clips=rows), series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    results = [diagnose(p) for p in args.manifests]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([r for r, _ in results], indent=2)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(max(len(s) for _, s in results), 1, figsize=(11, 10), squeeze=False)
        for j, (_, series) in enumerate(results):
            for ax, (name, t, source, target) in zip(axes[:, 0], series):
                if j == 0:
                    ax.plot(t, source, label='G1 source', color='black', linewidth=1.4)
                label = args.manifests[j].parts[-3]
                ax.plot(t, target, label=label, linewidth=1.1)
                ax.set_title(name)
                ax.set_ylabel('Torso forward lean (deg)')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        axes[-1, 0].set_xlabel('Time (s), positive lean = forward')
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
    for result, _ in results:
        print(result['manifest'])
        for c in result['clips']:
            print(c['name'], 'torso', c['torso'])


if __name__ == '__main__':
    main()
