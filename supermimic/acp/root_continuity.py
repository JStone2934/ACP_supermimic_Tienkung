"""Independent linear-root continuity screen for retargeted references.

These are engineering fidelity limits, not actuator limits or a dynamics proof.
Score target-minus-source derivatives so the original locomotion is retained
and constant morphology/height offsets are allowed. All original 50 Hz frames
are scored, including support transitions and window boundaries.
"""
from __future__ import annotations
import numpy as np

POLICY = dict(version='root-continuity-v1', fps=50,
              added_velocity_rmse_max_m_s=.5, added_velocity_p95_max_m_s=1.,
              added_velocity_max_m_s=2.5, added_acceleration_max_m_s2=100.,
              scope='all frames, world linear root target-minus-source derivatives; constant offsets allowed; not dynamic feasibility')


def score_root_positions(source, target, fps=50.):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('root continuity requires matching complete (T, 3) positions with T >= 3')
    if fps != POLICY['fps'] or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('root continuity requires finite original 50 Hz trajectories')
    correction = target-source
    speed = np.linalg.norm(np.diff(correction, axis=0)*fps, axis=1)
    acceleration = np.linalg.norm(np.diff(correction, n=2, axis=0)*fps**2, axis=1)
    metrics = dict(added_velocity_rmse_m_s=float(np.sqrt(np.mean(speed**2))),
                   added_velocity_p95_m_s=float(np.percentile(speed, 95)),
                   added_velocity_max_m_s=float(speed.max()),
                   added_acceleration_max_m_s2=float(acceleration.max()))
    limits = dict(added_velocity_rmse_m_s='added_velocity_rmse_max_m_s',
                  added_velocity_p95_m_s='added_velocity_p95_max_m_s',
                  added_velocity_max_m_s='added_velocity_max_m_s',
                  added_acceleration_max_m_s2='added_acceleration_max_m_s2')
    reasons = [name for name, limit in limits.items() if metrics[name] > POLICY[limit]]
    return dict(policy=POLICY, frames=len(source), fps=float(fps), metrics=metrics,
                worst_velocity_frame=int(speed.argmax())+1,
                worst_acceleration_frame=int(acceleration.argmax())+2,
                passed=not reasons, rejection_reasons=reasons)


def score_root_files(record, target_path):
    from .motion import load_php_npz
    from .robots import sha256
    source, _, _, source_fps = load_php_npz(record['source'], 'g1')
    target, _, _, target_fps = load_php_npz(target_path, 'tienkung3_29dof')
    first, last = record['source_frames']
    if source_fps != target_fps or len(target) != last-first:
        raise ValueError('root continuity cannot change source sampling or frame interval')
    result = score_root_positions(source[first:last], target, target_fps)
    result.update(source_sha256=sha256(record['source']), motion_sha256=sha256(target_path),
                  source_frames=[first, last])
    if result['source_sha256'] != record['source_hash']:
        raise ValueError('root continuity source identity changed')
    return result


def required(version):
    prefix='php-segment-v'
    suffix=version.removeprefix(prefix)
    return version.startswith(prefix) and suffix.isdigit() and int(suffix)>=21


def verify_report(report):
    if report.get('policy') != POLICY:
        raise ValueError('missing or changed root continuity policy')
    limits = dict(added_velocity_rmse_m_s='added_velocity_rmse_max_m_s',
                  added_velocity_p95_m_s='added_velocity_p95_max_m_s',
                  added_velocity_max_m_s='added_velocity_max_m_s',
                  added_acceleration_max_m_s2='added_acceleration_max_m_s2')
    metrics=report['metrics']
    if set(metrics)!=set(limits) or any(not np.isfinite(v) or v<0 for v in metrics.values()):
        raise ValueError('invalid root continuity measurements')
    reasons=[name for name,limit in limits.items() if metrics[name]>POLICY[limit]]
    if report.get('passed') != (not reasons) or report.get('rejection_reasons') != reasons:
        raise ValueError('root continuity verdict omits a failed metric')
