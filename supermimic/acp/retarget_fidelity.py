"""Segment-length retargeting and native-FK motion fidelity measurements.

The reference is rebuilt from the original G1 motion, never from a solved target
or its stored intent. Arm measurements are in torso coordinates, so walking or
turning the root cannot make frozen arms appear to track.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
import mujoco

from .robots import get_robot
from .motion import load_php_npz

VERSION = 'segment-directions-v1'
# Canonical semantic order shared by RobotSpec.tracked_bodies.
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12)
ARM_POINTS = (9, 10, 12, 13)
ARM_NAMES = ('left_elbow', 'left_hand', 'right_elbow', 'right_hand')
POLICY = dict(version='arm-fidelity-v1', reference=VERSION,
              rmse_max_m=.05, p95_max_m=.10, dynamic_rms_min_m=.025,
              amplitude_ratio_min=.70, amplitude_ratio_max=1.30,
              correlation_min=.85, max_lag_frames=2, lag_search_frames=10,
              min_phase_frames=8, fps=50,
              scope='free arm phases in torso coordinates; support phases use independent contact gates')
POSTURE_POLICY = dict(version='torso-attitude-v1', reference='neutral-calibrated-source-orientation',
                      angular_rmse_max_deg=3., angular_p95_max_deg=5., angular_max_deg=10.,
                      scope='torso full 3D attitude over every frame; pelvis measured separately, not gated',
                      fps=50)


def posture_required(version):
    number = version.removeprefix('php-segment-v')
    return version.startswith('php-segment-v') and number.isdigit() and int(number) >= 9


def verify_fidelity_report(report, arm_policy, posture_policy=None):
    """Require every declared gate in the exported quality conjunction."""
    if report.get('fidelity_policy') != arm_policy or report.get('posture_policy') != posture_policy:
        raise ValueError('quality report omits required fidelity/posture policy')
    for row in report['variants']:
        expected = row['geometry_pass']
        if arm_policy is not None:
            if row['fidelity']['policy'] != arm_policy or row.get('fidelity_pass') != row['fidelity']['passed']:
                raise ValueError('inconsistent arm fidelity verdict')
            expected = expected and row['fidelity']['passed']
        if posture_policy is not None:
            if row['posture']['policy'] != posture_policy or row.get('posture_pass') != row['posture']['passed']:
                raise ValueError('inconsistent posture verdict')
            expected = expected and row['posture']['passed']
        if report.get('root_continuity_policy') is not None:
            from .root_continuity import POLICY as ROOT_POLICY, verify_report
            if report['root_continuity_policy'] != ROOT_POLICY:
                raise ValueError('changed root continuity policy')
            measured=report['root_continuity']
            verify_report(measured)
            if row.get('root_continuity_pass') != measured['passed']:
                raise ValueError('inconsistent root continuity verdict')
            expected = expected and measured['passed']
        if row.get('quality_pass', row['geometry_pass']) != expected:
            raise ValueError('quality pass omits required fidelity/posture verdict')


def score_posture(reference, actual, policy=None):
    """Native body orientations, ordered pelvis/torso; retain source inclination.

    The torso gate covers pitch, roll AND yaw. It is not invariant to spurious
    whole-torso rotation, unlike the existing torso-local arm measurements.
    Pelvis orientation remains diagnostic because leg/contact adaptation may
    require a different pelvis attitude on the taller destination robot.
    """
    policy = POSTURE_POLICY if policy is None else policy
    if reference.shape != actual.shape or reference.shape[1:] != (2, 3, 3) or not len(reference):
        raise ValueError('posture requires matching nonempty pelvis/torso orientations')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('nonfinite posture input')
    relative = reference.transpose(0, 1, 3, 2) @ actual
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    bodies, reasons = {}, []
    for i, name in enumerate(('pelvis', 'torso')):
        row = dict(angular_rmse_deg=float(np.sqrt(np.mean(angle[:, i]**2))),
                   angular_p95_deg=float(np.percentile(angle[:, i], 95)),
                   angular_max_deg=float(angle[:, i].max()), gated=name == 'torso')
        if row['gated']:
            for metric in ('rmse', 'p95', 'max'):
                limit_key = 'angular_max_deg' if metric == 'max' else f'angular_{metric}_max_deg'
                if row[f'angular_{metric}_deg'] > policy[limit_key]:
                    reasons.append(name+':'+metric)
        bodies[name] = row
    return dict(policy=policy, passed=not reasons, rejection_reasons=reasons, bodies=bodies)


def native_posture(record, target_path):
    p, quat, q, fps = load_php_npz(record['source'], 'g1')
    a, b = record['source_frames']
    _, source = native_poses('g1', p[a:b], quat[a:b], q[a:b])
    p, quat, q, target_fps = load_php_npz(target_path, 'tienkung3_29dof')
    if len(p) != b-a or fps != target_fps or fps != POSTURE_POLICY['fps']:
        raise ValueError('posture requires unchanged full source timing')
    _, actual = native_poses('tienkung3_29dof', p, quat, q)
    _, sr = neutral_poses('g1')
    _, tr = neutral_poses('tienkung3_29dof')
    indices = [0, 7]
    reference = source[:, indices] @ sr[indices].transpose(0, 2, 1) @ tr[indices]
    return score_posture(reference, actual[:, indices])


@lru_cache(maxsize=2)
def native_model(robot_id):
    robot = get_robot(robot_id)
    model = robot.get_spec().compile()
    return model, robot.joint_addresses(model)[0], [model.body(n).id for n in robot.tracked_bodies]


def native_poses(robot_id, p, quat, q):
    model, qa, ids = native_model(robot_id)
    state = mujoco.MjData(model)
    positions, rotations = [], []
    for pos, rot, joints in zip(p, quat, q):
        state.qpos[:3], state.qpos[3:7], state.qpos[qa] = pos, rot, joints
        mujoco.mj_kinematics(model, state)
        positions.append(state.xpos[ids].copy())
        rotations.append(state.xmat[ids].reshape(-1, 3, 3).copy())
    return np.array(positions), np.array(rotations)


@lru_cache(maxsize=2)
def neutral_poses(robot_id):
    robot = get_robot(robot_id)
    p, R = native_poses(robot_id, np.zeros((1, 3)), np.array([[1., 0, 0, 0]]), np.array([robot.default_q]))
    return p[0], R[0]


def segment_targets(source_positions, source_rotations, target_root):
    """Preserve source bone directions using each target segment's own length.

    Hip/shoulder mounts use target anatomy, in the root/torso frames. Each thigh,
    shin, upper arm and forearm is reconstructed separately. No G1 joint angle
    is copied to the differently articulated destination robot.
    """
    sp, sr = neutral_poses('g1')
    tp, tr = neutral_poses('tienkung3_29dof')
    P, R = np.asarray(source_positions), np.asarray(source_rotations)
    target = np.zeros_like(P)
    target[:, 0] = target_root
    root_R = R[:, 0] @ sr[0].T @ tr[0]
    torso_R = R[:, 7] @ sr[7].T @ tr[7]
    for child in range(1, 14):
        parent = PARENTS[child]
        length = np.linalg.norm(tp[child]-tp[parent])
        if child in (1, 4):
            vector = (root_R @ (tr[0].T @ (tp[child]-tp[parent]))[..., None]).squeeze(-1)
        elif child in (8, 11):
            vector = (torso_R @ (tr[7].T @ (tp[child]-tp[parent]))[..., None]).squeeze(-1)
        else:
            vector = P[:, child]-P[:, parent]
            norm = np.linalg.norm(vector, axis=-1, keepdims=True)
            if np.any(norm < 1e-8):
                raise ValueError('degenerate source segment')
            vector = vector / norm * length
        target[:, child] = target[:, parent] + vector
    arm_local = (torso_R[:, None].transpose(0, 1, 3, 2) @
                 (target[:, ARM_POINTS]-target[:, 7, None])[..., None]).squeeze(-1)
    return target, arm_local


def free_arm_mask(contact):
    """A supporting hand constrains the whole arm, including its elbow."""
    left = ~(contact[:, 4] | contact[:, 6])
    right = ~(contact[:, 5] | contact[:, 7])
    return np.stack((left, left, right, right), -1)


def _phases(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _correlation(a, b):
    a, b = a-a.mean(0), b-b.mean(0)
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denom) if denom > 1e-10 else 0.


def score_arm_trajectories(reference, actual, free, policy=None):
    policy = POLICY if policy is None else policy
    if reference.shape != actual.shape or reference.shape != (*free.shape, 3):
        raise ValueError('arm fidelity frame/point shapes differ')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('nonfinite arm fidelity input')
    reports, reasons = {}, []
    for j, name in enumerate(ARM_NAMES):
        phases = []
        for a, b in _phases(free[:, j]):
            if b-a < policy['min_phase_frames']:
                continue
            ref, got = reference[a:b, j], actual[a:b, j]
            error = np.linalg.norm(ref-got, axis=-1)
            ref_amp = float(np.sqrt(np.mean(np.sum((ref-ref.mean(0))**2, axis=-1))))
            got_amp = float(np.sqrt(np.mean(np.sum((got-got.mean(0))**2, axis=-1))))
            row = dict(start=int(a), end=int(b), rmse_m=float(np.sqrt(np.mean(error**2))),
                       p95_m=float(np.percentile(error, 95)), reference_rms_amplitude_m=ref_amp,
                       actual_rms_amplitude_m=got_amp, dynamic=ref_amp >= policy['dynamic_rms_min_m'])
            if row['rmse_m'] > policy['rmse_max_m'] or row['p95_m'] > policy['p95_max_m']:
                reasons.append(name+':trajectory')
            if row['dynamic']:
                ratio = got_amp/ref_amp
                correlation = _correlation(ref, got)
                scores = []
                for lag in range(-min(policy['lag_search_frames'], (b-a)//4), min(policy['lag_search_frames'], (b-a)//4)+1):
                    x, y = (ref[:lag], got[-lag:]) if lag < 0 else ((ref[lag:], got[:-lag]) if lag else (ref, got))
                    scores.append((_correlation(x, y), lag))
                # Prefer zero/small lag when numerically tied.
                best = max(scores, key=lambda x: (round(x[0], 8), -abs(x[1])))
                row.update(amplitude_ratio=ratio, correlation=correlation, best_lag_frames=best[1])
                if not policy['amplitude_ratio_min'] <= ratio <= policy['amplitude_ratio_max']:
                    reasons.append(name+':amplitude')
                if correlation < policy['correlation_min'] or abs(best[1]) > policy['max_lag_frames']:
                    reasons.append(name+':timing')
            phases.append(row)
        reports[name] = dict(free_frames=int(free[:, j].sum()), phases=phases)
    return dict(policy=policy, passed=not reasons, rejection_reasons=sorted(set(reasons)), points=reports)


def native_arm_trajectories(record, target_path):
    """Independent native FK on both motion files, preserving source frame times."""
    p, quat, q, fps = load_php_npz(record['source'], 'g1')
    a, b = record['source_frames']
    P, R = native_poses('g1', p[a:b], quat[a:b], q[a:b])
    _, reference = segment_targets(P, R, P[:, 0])
    p, quat, q, target_fps = load_php_npz(target_path, 'tienkung3_29dof')
    if len(p) != b-a or target_fps != fps or fps != POLICY['fps']:
        raise ValueError('fidelity requires unchanged full source timing')
    P, R = native_poses('tienkung3_29dof', p, quat, q)
    actual = (R[:, 7, None].transpose(0, 1, 3, 2) @ (P[:, ARM_POINTS]-P[:, 7, None])[..., None]).squeeze(-1)
    return reference, actual


def contact_transition_weights(free, ramp_frames=8):
    """Objective weights only; independent free-phase fidelity scoring is unchanged.

    Fade near actual support transitions, never at a source clip or solver window
    boundary. Keep at least half of a short free phase at full tracking weight.
    """
    free = np.asarray(free, dtype=bool)
    if free.ndim != 2 or ramp_frames < 0:
        raise ValueError('expected time/point free mask and nonnegative ramp length')
    weights = free.astype(np.float32)
    for b in range(free.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[False, free[:, b], False].astype(np.int8)))
        for a, z in zip(edges[::2], edges[1::2]):
            n = min(ramp_frames, (z-a)//4)
            if not n:
                continue
            ramp = np.sin(.5*np.pi*np.arange(1, n+1)/(n+1))**2
            if a > 0:
                weights[a:a+n, b] = ramp
            if z < len(free):
                weights[z-n:z, b] = ramp[::-1]
    return weights
