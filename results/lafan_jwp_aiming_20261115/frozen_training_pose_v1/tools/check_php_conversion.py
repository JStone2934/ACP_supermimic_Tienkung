"""Independent full-body, per-terrain PHP conversion acceptance and target labels."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from supermimic.acp.make_labels import PER_VARIANT, geom_distance, variant_labels
from supermimic.acp.source_contacts import SOURCE_CONTACT_POLICY, verify_source_receipts, source_validation_required
from supermimic.acp.motion import load_php_npz
from supermimic.acp.root_continuity import POLICY as ROOT_POLICY, required as root_required, score_root_files
from supermimic.acp.php_conversion import COLLISION_POLICY, STRUCTURAL_PAIRS, atomic_json, atomic_npz, digest
from supermimic.acp.robots import CONTACT_NAMES, get_robot, sha256
from supermimic.acp.retarget_fidelity import (POLICY as FIDELITY_POLICY, POSTURE_POLICY, free_arm_mask,
                                            native_arm_trajectories, score_arm_trajectories,
                                            native_posture, posture_required)
from supermimic.acp.contact_intent import POLICY as CONTACT_INTENT_POLICY, PATCH_POLICY
from supermimic.acp.terrain import load_terrain, terrain_path
from tools.check_php_pilot import longest, scene


def assess_variant(robot, path, intent, variant):
    p, quat, q, fps = load_php_npz(path, robot)
    boxes = load_terrain(str(path))[variant]
    m = scene(robot, boxes, self_collision=True)
    d = mujoco.MjData(m)
    qadr, _ = robot.joint_addresses(m)
    frames = [m.body(robot.contact_bodies[n][0]).id for n in CONTACT_NAMES]
    groups = [[g for g in range(m.ngeom) if m.geom_conaffinity[g] and
               m.body(m.geom_bodyid[g]).name in robot.contact_bodies[n][1]] for n in CONTACT_NAMES]
    terrain = [m.geom('ground').id] + [m.geom(f'box{k}').id for k in range(len(boxes))]
    active, offsets = intent['contact'][variant], intent['offsets'][variant]
    normals, anchors, surfaces = (intent[k][variant] for k in ('normals', 'anchors', 'surfaces'))
    points = np.zeros((len(p), 9, 3))
    distances = np.full((len(p), 9), np.nan)
    pen_terrain, pen_raw, pen_policy = [], [], []
    self_pairs, terrain_pairs = defaultdict(float), defaultdict(float)
    witness = np.zeros(6)
    for t in range(len(p)):
        d.qpos[:3], d.qpos[3:7], d.qpos[qadr] = p[t], quat[t], q[t]
        mujoco.mj_forward(m, d)
        pt, pr, pp = 0., 0., 0.
        for c in d.contact:
            b1, b2 = (m.body(m.geom_bodyid[g]).name for g in c.geom)
            penetration = max(0., -float(c.dist))
            key = ' / '.join(sorted((b1, b2)))
            if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                pt = max(pt, penetration)
                terrain_pairs[key] = max(terrain_pairs[key], penetration)
            else:
                pr = max(pr, penetration)
                self_pairs[key] = max(self_pairs[key], penetration)
                if frozenset((b1, b2)) not in STRUCTURAL_PAIRS:
                    pp = max(pp, penetration)
        pen_terrain.append(pt)
        pen_raw.append(pr)
        pen_policy.append(pp)
        for b, bid in enumerate(frames):
            points[t, b] = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ offsets[t, b]
            if active[t, b]:
                surface = int(surfaces[t, b])
                if not 0 <= surface < len(terrain):
                    raise ValueError('intent refers to a missing terrain surface')
                distances[t, b] = min(geom_distance(m, d, g, terrain[surface], .5, witness,
                                                    strict_reference=True) for g in groups[b])
    lo = np.array([m.joint(n).range[0] for n in robot.joint_names])
    hi = np.array([m.joint(n).range[1] for n in robot.joint_names])
    limit = max(0., float(np.maximum(lo-q, q-hi).max()))
    speed_ratio = float((np.abs(np.gradient(q, axis=0)*fps) / robot.velocity).max())
    pt, pr, pp = np.array(pen_terrain), np.array(pen_raw), np.array(pen_policy)
    reasons = []
    if limit > 1e-4:
        reasons.append('joint_limit')
    if speed_ratio > 1:
        reasons.append('joint_velocity')
    if pt.max() > .005 or longest(pt > .002) >= 3:
        reasons.append('terrain_penetration')
    if pp.max() > .005 or longest(pp > .002) >= 3:
        reasons.append('self_collision')
    contacts = {}
    for b, name in enumerate(CONTACT_NAMES):
        mask = active[:, b]
        # Different patches/surfaces are separate contact phases; never score a switch as slip.
        consecutive = mask[1:] & mask[:-1] & (surfaces[1:, b] == surfaces[:-1, b])
        consecutive &= np.linalg.norm(offsets[1:, b] - offsets[:-1, b], axis=-1) < 1e-6
        displacement = np.diff(points[:, b], axis=0)*fps
        tangent = displacement - (displacement*normals[1:, b]).sum(-1, keepdims=True)*normals[1:, b]
        errors, slip = np.abs(distances[mask, b]), np.linalg.norm(tangent[consecutive], axis=-1)
        touch95 = float(np.percentile(errors, 95)) if len(errors) else None
        slip95 = float(np.percentile(slip, 95)) if len(slip) else None
        anchor95 = float(np.percentile(np.linalg.norm(points[mask, b]-anchors[mask, b], axis=-1), 95)) if mask.any() else None
        contacts[name] = dict(frames=int(mask.sum()), distance_p95_m=touch95, slip_p95_m_s=slip95, anchor_error_p95_m=anchor95)
        if mask.any() and (touch95 is None or not np.isfinite(touch95) or touch95 > .01):
            reasons.append(f'{name}:support_distance')
        if consecutive.any() and (slip95 is None or not np.isfinite(slip95) or slip95 > .05):
            reasons.append(f'{name}:support_slip')
    if not active.any():
        reasons.append('no_contact_intent')
    raw_ok = bool(pr.max() <= .005 and longest(pr > .002) < 3)
    return dict(variant=variant, frames=len(p), duration_s=(len(p)-1)/fps, geometry_pass=not reasons,
                strict_raw_geometry_pass=not reasons and raw_ok, rejection_reasons=reasons,
                collision_policy_hash=digest(COLLISION_POLICY), joint_limit_max_rad=limit,
                joint_velocity_max_ratio=speed_ratio, terrain_penetration_max_m=float(pt.max()),
                terrain_penetration_over_2mm_longest_frames=longest(pt > .002),
                self_penetration_raw_max_m=float(pr.max()), self_penetration_policy_max_m=float(pp.max()),
                self_penetration_policy_over_2mm_longest_frames=longest(pp > .002),
                self_pair_max_m=dict(self_pairs), terrain_pair_max_m=dict(terrain_pairs), contacts=contacts,
                pd_status='not_run', learned_tracking_status='not_run')


def evaluate_task(task, root):
    """One CPU process per source. Report/labels are committed before the terminal marker."""
    started = time.monotonic()
    root = Path(root)
    record = task['record']
    path = root/'candidates'/(record['key']+'_motion.npz')
    intent_path = path.with_name(path.name.replace('_motion.npz', '_intent.npz'))
    robot = get_robot('tienkung3_29dof')
    robot_metadata = robot.metadata()
    with np.load(intent_path) as z:
        intent = {k: z[k] for k in ('contact', 'offsets', 'normals', 'anchors', 'surfaces')}
        intent_policy = json.loads(str(z['contact_intent_policy'].item())) if 'contact_intent_policy' in z else None
        patch_policy = json.loads(str(z['contact_patch_policy'].item())) if 'contact_patch_policy' in z else None
        source_validation_policy = json.loads(str(z['source_contact_validation_policy'].item())) if 'source_contact_validation_policy' in z else None
        source_receipts = json.loads(str(z['source_contact_validation'].item())) if 'source_contact_validation' in z else None
    with np.load(path) as z:
        version = str(z['data_version'].item()) if 'data_version' in z else ''
        fidelity_required = version.startswith('php-segment-')
    if (version in ('php-segment-v5', 'php-segment-v6', 'php-segment-v7', 'php-segment-v8') or posture_required(version)) and intent_policy != CONTACT_INTENT_POLICY:
        raise ValueError('missing or changed contact intent classification policy')
    if (version in ('php-segment-v6', 'php-segment-v7', 'php-segment-v8') or posture_required(version)) and patch_policy != PATCH_POLICY:
        raise ValueError('missing or changed contact patch policy')
    if source_validation_required(version):
        if source_validation_policy != SOURCE_CONTACT_POLICY:
            raise ValueError('missing or changed source contact validation policy')
        verify_source_receipts(record, source_receipts)
    trajectories = native_arm_trajectories(record, path) if fidelity_required else None
    posture = native_posture(record, path) if posture_required(version) else None
    root_continuity = score_root_files(record, path) if root_required(version) else None
    reports, labels = {}, {}
    terrains = load_terrain(str(path))
    for v in record['unique_variants']:
        reports[v] = assess_variant(robot, path, intent, v)
        if fidelity_required:
            fidelity = score_arm_trajectories(*trajectories, free_arm_mask(intent['contact'][v]))
            reports[v].update(fidelity=fidelity, fidelity_pass=fidelity['passed'],
                              quality_pass=reports[v]['geometry_pass'] and fidelity['passed'],
                              geometry_rejection_reasons=list(reports[v]['rejection_reasons']))
            reports[v]['rejection_reasons'] += ['fidelity:'+r for r in fidelity['rejection_reasons']]
        if posture is not None:
            reports[v].update(posture=posture, posture_pass=posture['passed'],
                              quality_pass=reports[v]['quality_pass'] and posture['passed'])
            reports[v]['rejection_reasons'] += ['posture:'+r for r in posture['rejection_reasons']]
        if root_continuity is not None:
            reports[v].update(root_continuity_pass=root_continuity['passed'],
                              quality_pass=reports[v]['quality_pass'] and root_continuity['passed'])
            reports[v]['rejection_reasons'] += ['root_continuity:'+r for r in root_continuity['rejection_reasons']]
        labels[v] = variant_labels((str(path), terrains[v], .005, .1, robot))
    variants = [dict(reports[alias], variant=v, computed_variant=alias)
                for v, alias in enumerate(record['variant_aliases'])]
    accepted = [v for v, report in enumerate(variants) if report.get('quality_pass', report['geometry_pass'])]
    report_path = root/'quality'/(task['id']+'.json')
    report = dict(task_id=task['id'], source=record, motion_hash=sha256(path), terrain_hash=sha256(terrain_path(str(path))),
                  intent_hash=sha256(intent_path), robot_metadata=robot_metadata,
                  collision_policy=COLLISION_POLICY, variants=variants, accepted_variants=accepted,
                  fidelity_policy=FIDELITY_POLICY if fidelity_required else None,
                  posture_policy=POSTURE_POLICY if posture is not None else None,
                  contact_intent_policy=intent_policy, contact_patch_policy=patch_policy,
                  source_contact_validation_policy=source_validation_policy,
                  source_contact_validation_verified=source_validation_required(version),
                  root_continuity_policy=ROOT_POLICY if root_continuity is not None else None,
                  root_continuity=root_continuity)
    atomic_json(report_path, report)
    rows = [labels[alias] for alias in record['variant_aliases']]
    arrays = {k: np.stack([r[k] for r in rows]) for k in PER_VARIANT}
    arrays.update(body_pos_w=rows[0]['body_pos_w'], body_quat_w=rows[0]['body_quat_w'],
                  other_link_names=rows[0]['other_link_names'], fps=rows[0]['fps'], boxes=terrains.astype(np.float32),
                  threshold=np.float32(.005), body_names=np.array(CONTACT_NAMES),
                  link_names=np.array([robot.contact_bodies[n][0] for n in CONTACT_NAMES]),
                  robot_metadata=json.dumps(robot_metadata), motion_hash=sha256(path),
                  terrain_hash=sha256(terrain_path(str(path))), quality_hash=sha256(report_path),
                  quality_pass=np.array([r.get('quality_pass', r['geometry_pass']) for r in variants]), source_motion=record['key']+'.npz')
    label_path = root/'candidate_contacts'/(record['key']+'.npz')
    atomic_npz(label_path, **arrays)
    exported = {}
    if accepted:
        dest = root/'accepted'/(record['key']+'_motion.npz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Keep the original candidate immutable; only terrain/labels/intent are subsetted.
        dest.write_bytes(path.read_bytes())
        np.save(terrain_path(str(dest)), np.load(terrain_path(str(path)))[:, accepted])
        with np.load(intent_path) as original:
            selected = {k: original[k][accepted] if k in ('contact', 'offsets', 'normals', 'anchors', 'surfaces', 'foot_target') else original[k]
                        for k in original.files}
        selected['source_variants'] = np.array(accepted)
        accepted_intent = dest.with_name(dest.name.replace('_motion.npz', '_intent.npz'))
        atomic_npz(accepted_intent, **selected)
        for k in PER_VARIANT:
            arrays[k] = arrays[k][accepted]
        arrays.update(boxes=terrains[accepted].astype(np.float32), quality_pass=np.ones(len(accepted), bool),
                      terrain_hash=sha256(terrain_path(str(dest))), source_variants=np.array(accepted))
        accepted_label = root/'accepted_contacts'/(record['key']+'.npz')
        atomic_npz(accepted_label, **arrays)
        exported = {str(p.relative_to(root)): sha256(p) for p in
                    (dest, Path(terrain_path(str(dest))), accepted_intent, accepted_label)}
    return dict(status='accepted' if accepted else 'rejected', accepted_variants=accepted,
                quality_path=str(report_path.relative_to(root)), quality_hash=sha256(report_path),
                labels_path=str(label_path.relative_to(root)), labels_hash=sha256(label_path),
                exported=exported, assessment_seconds=time.monotonic()-started)


def collision_evidence():
    """Document the actual serial ankle hierarchy and raw overlap at representative joint states."""
    robot = get_robot('tienkung3_29dof')
    m = scene(robot, np.zeros((0, 8)), self_collision=True)
    d = mujoco.MjData(m)
    qa, _ = robot.joint_addresses(m)
    rows = []
    for pitch, roll in ((0., 0.), (-.15, 0.), (.25, .1), (-.35, -.1)):
        d.qpos[:3] = [0, 0, 2]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[qa] = robot.default_q
        for side in ('l', 'r'):
            d.qpos[qa[robot.joint_names.index(f'ankle_pitch_{side}_joint')]] = pitch
            d.qpos[qa[robot.joint_names.index(f'ankle_roll_{side}_joint')]] = roll
        mujoco.mj_forward(m, d)
        distances = {}
        for pair in STRUCTURAL_PAIRS:
            a, b = sorted(pair)
            ga = [g for g in range(m.ngeom) if m.geom_conaffinity[g] and m.geom_bodyid[g] == m.body(a).id]
            gb = [g for g in range(m.ngeom) if m.geom_conaffinity[g] and m.geom_bodyid[g] == m.body(b).id]
            distances[' / '.join(sorted(pair))] = min(geom_distance(m, d, x, y, 1., np.zeros(6), strict_reference=True) for x in ga for y in gb)
        rows.append(dict(ankle_pitch=pitch, ankle_roll=roll, pair_distance_m=distances))
    chains = []
    for side in ('l', 'r'):
        foot = m.body(f'ankle_roll_{side}_link').id
        parent = m.body_parentid[foot]
        grandparent = m.body_parentid[parent]
        chains.append([m.body(b).name for b in (grandparent, parent, foot)])
        if chains[-1] != [f'knee_pitch_{side}_link', f'ankle_pitch_{side}_link', f'ankle_roll_{side}_link']:
            raise ValueError('structural exclusion does not match the model hierarchy')
    return dict(robot_metadata=robot.metadata(), policy=COLLISION_POLICY, chains=chains, representative_poses=rows,
                note='This changes the documented acceptance pair policy, not robot geometry, FK, mass or raw metrics. '
                     'Old reports remain historical; every new output is checked and relabeled with this policy hash.')
