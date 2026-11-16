"""Revalidate stored zero source distances before inferring support phases.

Original motions, terrains and label files remain immutable. Native queries
cover every collision geom of the body and every terrain surface, using the
strict witness validation shared with the independent geometry checker.
"""
import numpy as np
import mujoco

from .make_labels import build_model, geom_distance, box_normal
from .robots import get_robot, CONTACT_NAMES
from .motion import load_php_npz
from .terrain import load_terrain

SOURCE_CONTACT_POLICY = dict(version='validated-source-zeros-v1', zero_tolerance_m=1e-10,
    query_horizon_m=.1, method='strict native geometry with bounded witness retries',
    scope='all stored near-zero source body distances; original files unchanged')


def validate_zero_contacts(robot, positions, quaternions, joints, boxes,
                           distances, points, normals, surfaces, first=0):
    distances, points, normals, surfaces = (x.copy() for x in (distances, points, normals, surfaces))
    indices = np.argwhere(np.abs(distances) <= SOURCE_CONTACT_POLICY['zero_tolerance_m'])
    receipt = dict(policy=SOURCE_CONTACT_POLICY, checked_count=len(indices), queries=[])
    if not len(indices):
        return distances, points, normals, surfaces, receipt
    robot = get_robot(robot)
    model = build_model(boxes, robot)
    state = mujoco.MjData(model)
    qadr, _ = robot.joint_addresses(model)
    groups = [[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g])
               and model.body(model.geom_bodyid[g]).name in robot.contact_bodies[name][1]]
              for name in CONTACT_NAMES]
    terrain = [(model.geom('ground').id, 0)] + [(model.geom(f'box{k}').id, k+1)
                for k, box in enumerate(boxes) if box[5] > 0]
    previous_frame = None
    for t, b in indices:
        if t != previous_frame:
            state.qpos[:3], state.qpos[3:7], state.qpos[qadr] = positions[t], quaternions[t], joints[t]
            mujoco.mj_kinematics(model, state)
            previous_frame = t
        old = dict(distance=float(distances[t, b]), surface=int(surfaces[t, b]))
        best, witness, surface, selected_geom = SOURCE_CONTACT_POLICY['query_horizon_m'], None, -1, None
        for g in groups[b]:
            for tg, surf in terrain:
                candidate = np.zeros(6)
                value = geom_distance(model, state, g, tg, SOURCE_CONTACT_POLICY['query_horizon_m'],
                                      candidate, strict_reference=True)
                if value < best:
                    best, witness, surface, selected_geom = value, candidate.copy(), surf, g
        distances[t, b], surfaces[t, b] = best, surface
        if witness is None:
            points[t, b] = normals[t, b] = np.nan
        else:
            points[t, b] = witness[3:]
            if surface == 0:
                normals[t, b] = [0., 0., 1.]
            elif abs(best) > 1e-6:
                delta = witness[:3]-witness[3:]
                normals[t, b] = np.sign(best)*delta/max(np.linalg.norm(delta), 1e-12)
            else:
                normals[t, b] = box_normal(witness[:3], boxes[surface-1])
        receipt['queries'].append(dict(frame=int(t+first), body=CONTACT_NAMES[b], original=old,
            validated=dict(distance=float(distances[t, b]), surface=int(surface), geom=selected_geom,
                           witness=None if witness is None else witness.tolist())))
    return distances, points, normals, surfaces, receipt


def verify_source_receipts(record, receipts):
    """CPU acceptance recomputes every recorded source-zero query from inputs."""
    positions, quaternions, joints, _ = load_php_npz(record['source'], 'g1')
    first, last = record['source_frames']
    positions, quaternions, joints = (x[first:last] for x in (positions, quaternions, joints))
    terrains = load_terrain(record['source'])
    if set(receipts) != {str(v) for v in record['unique_variants']}:
        raise ValueError('incomplete source-zero validation variants')
    with np.load(record['labels']) as source:
        for v in record['unique_variants']:
            lv = min(v, record['label_variants']-1)
            arrays = [source[k][lv, first:last] for k in
                      ('contact_dist', 'contact_pos_w', 'contact_normal_w', 'contact_surface')]
            expected = validate_zero_contacts('g1', positions, quaternions, joints, terrains[v],
                                              *arrays, first=first)[-1]
            if receipts[str(v)] != expected:
                raise ValueError('source-zero validation receipt differs from original geometry')


def source_validation_required(version):
    prefix = 'php-segment-v'
    suffix = version.removeprefix(prefix)
    return version.startswith(prefix) and suffix.isdigit() and int(suffix) >= 18
