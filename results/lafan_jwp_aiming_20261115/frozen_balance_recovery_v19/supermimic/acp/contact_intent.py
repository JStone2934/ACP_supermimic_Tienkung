"""Separate brief, fast limb grazes from stationary support in source motion.

Raw proximity labels remain immutable. A knee/elbow that sweeps across terrain
is still subject to target collision avoidance, but is not pinned to the ground
as if the original robot held a stationary support. Feet and hands retain their
existing support classification; long and slow limb contacts are preserved.
"""
import numpy as np

POLICY = dict(version='brief-limb-graze-v1', bodies=['left_knee', 'right_knee', 'left_elbow', 'right_elbow'],
              max_duration_s=.3, min_material_tangent_speed_m_s=.35,
              effect='omit stationary anchor for brief fast graze; retain full-body terrain collision gate')

PATCH_POLICY = dict(version='anatomical-contact-patch-v1', distal_shank_fraction_min=.85,
                    ankle_distance_fraction_max=.20,
                    mapping='normalized source collision surface coordinates to corresponding target surface patch',
                    distal_mapping='distal shank/ankle contact aliases an already active same-surface foot support')

# G1 rubber fingers extend along local +X; TienKung TCP fingers extend along -Z.
# Columns are forward, lateral and palm-normal axes in the target hand frame.
HAND_BASIS = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])


def material_tangent_speed(positions, rotations, distances, terrain_points, normals, fps):
    valid = np.isfinite(terrain_points).all(-1) & np.isfinite(normals).all(-1)
    point = np.where(valid[..., None], terrain_points+distances[..., None]*normals, positions)
    local = (rotations.transpose(0, 1, 3, 2) @ (point-positions)[..., None]).squeeze(-1)
    transported = positions[1:] + (rotations[1:] @ local[:-1, ..., None]).squeeze(-1)
    velocity = (transported-point[:-1])*fps
    normal = np.nan_to_num(normals[:-1])
    tangent = velocity-np.sum(velocity*normal, axis=-1, keepdims=True)*normal
    speed = np.linalg.norm(tangent, axis=-1)
    speed[~(valid[1:] & valid[:-1])] = np.nan
    return speed


def is_brief_graze(body_name, duration_s, median_speed):
    return bool(body_name in POLICY['bodies'] and duration_s < POLICY['max_duration_s'] and
                np.isfinite(median_speed) and median_speed > POLICY['min_material_tangent_speed_m_s'])


def distal_shank_alias(knee, ankle, source_point, same_surface_foot_support):
    """Do not interpret contact at the ankle end of the shank as kneeling.

    Every frame must already have foot support on that SAME surface. Genuine
    proximal knee support, isolated shank support and other surfaces are kept.
    """
    axis = ankle-knee
    length2 = np.sum(axis*axis, axis=-1)
    fraction = np.sum((source_point-knee)*axis, axis=-1)/np.maximum(length2, 1e-12)
    ankle_distance = np.linalg.norm(source_point-ankle, axis=-1)/np.sqrt(np.maximum(length2, 1e-12))
    mapped = bool(np.all(same_surface_foot_support) and
                  np.all(fraction >= PATCH_POLICY['distal_shank_fraction_min']) and
                  np.all(ankle_distance <= PATCH_POLICY['ankle_distance_fraction_max']))
    return mapped, dict(axial_fraction_min=float(fraction.min()), axial_fraction_median=float(np.median(fraction)),
                        ankle_distance_fraction_max=float(ankle_distance.max()))


def corresponding_patch(source_local, source_cloud, target_cloud, target_normal, target_basis=None):
    """Use the source contact's anatomical region, rather than the whole limb's tip."""
    basis = np.eye(3) if target_basis is None else target_basis
    target_anatomical = target_cloud @ basis
    low, high = source_cloud.min(0), source_cloud.max(0)
    fraction = np.clip((source_local-low)/np.maximum(high-low, 1e-6), 0, 1)
    mapped = basis @ (target_anatomical.min(0)+fraction*(target_anatomical.max(0)-target_anatomical.min(0)))
    distance = np.linalg.norm(target_cloud-mapped, axis=-1)
    # Restrict to the corresponding physical neighborhood before selecting the
    # side facing the terrain. This prevents a proximal knee target becoming an
    # ankle-end target merely because it is lower in the reference orientation.
    nearby = target_cloud[distance <= distance.min()+.012]
    projection = nearby @ target_normal
    patch = nearby[projection <= projection.min()+.002]
    return patch.mean(0)
