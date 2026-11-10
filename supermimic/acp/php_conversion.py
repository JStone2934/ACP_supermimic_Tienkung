"""Offline, provenance-preserving PHP retargeting. No simulator environments are allocated.

Sampled collision losses guide IK; native MuJoCo distances remain the acceptance oracle.
The historical seven-clip pilot is deliberately kept separate.
"""
from __future__ import annotations
from .collision_objective import pair_collision_cost

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.ndimage import binary_closing
from scipy.spatial import ConvexHull
import torch

from .geometry import quat_mat, surface_points
from .kinematics import RobotKinematics
from .motion import load_php_npz
from .robots import CONTACT_NAMES, get_robot, sha256
from .retarget_fidelity import ARM_POINTS, free_arm_mask, segment_targets, contact_transition_weights
from .contact_intent import HAND_BASIS, corresponding_patch, distal_shank_alias, is_brief_graze, material_tangent_speed
from .terrain import box_sdf, load_terrain, terrain_path
from ..rotations import axis_angle_to_mat, mat_to_quat, quat_to_mat, so3_exp
from tools.retarget_php_pilot import link_cloud, runs, sole, support_height

VERSION = 'php-segment-v26'
STRUCTURAL_PAIRS = {frozenset((f'knee_pitch_{s}_link', f'ankle_roll_{s}_link')) for s in ('l', 'r')}
COLLISION_POLICY = {
    'version': 'serial-ankle-adjacency-v1',
    'excluded_pairs': [sorted(p) for p in sorted(STRUCTURAL_PAIRS, key=lambda p: sorted(p))],
    'reason': 'Shank and foot are adjacent physical assemblies across the two serial ankle hinges; '
              'the intervening ankle_pitch link is the universal-joint bearing. Original convex '
              'hulls overlap at this articulation. Exclude only these two self pairs, retain all '
              'terrain collisions and report raw penetration separately. No collision mesh is changed.',
}


def root_correction_cost(dp, fps=50.):
    """Penalize added root velocity/acceleration, preserving source motion.

    dp is target minus reference root position (metres), so constant morphology
    offsets cost zero. Temporal derivatives use seconds rather than frame units.
    This objective is not an independent dynamics acceptance gate.
    """
    velocity = (dp[1:] - dp[:-1]) * fps
    acceleration = (dp[2:] - 2*dp[1:-1] + dp[:-2]) * fps**2
    return .25*velocity.square().mean() + .001*acceleration.square().mean()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(tmp, path)


@dataclass(frozen=True)
class SolverConfig:
    iterations: int = 1000
    window_frames: int = 512
    overlap: int = 48
    learning_rate: float = .012
    seed: int = 0
    graph: bool = True
    min_iterations: int = 400
    convergence_rtol: float = 1e-5
    collision_rounds: int = 12
    collision_iterations: int = 400
    clearance_max_m: float = .008

    def __post_init__(self):
        if self.iterations < 1 or self.window_frames < 8 or not 2 <= self.overlap < self.window_frames // 2:
            raise ValueError('positive iterations and 2 <= overlap < window_frames/2 required')
        if self.collision_rounds < 0 or self.collision_iterations < 1:
            raise ValueError('nonnegative collision rounds and positive collision iterations required')


def unique_variants(terrain):
    """Return original variant indices; zero-box slots are explicitly aliased."""
    representatives, aliases = [], []
    for v in range(terrain.shape[1]):
        match = next((j for j in representatives if np.array_equal(terrain[:, v], terrain[:, j])), None)
        if match is None:
            representatives.append(v)
            match = v
        aliases.append(match)
    return representatives, aliases


def discover(source, contacts, only=None):
    source, contacts = Path(source).resolve(), Path(contacts).resolve()
    files = sorted(source.glob('*/*_motion.npz'))
    if not files:
        raise ValueError(f'no source motions in {source}')
    records = []
    for path in files:
        key = str(path.relative_to(source)).removesuffix('_motion.npz')
        if only and not any(s in key for s in only):
            continue
        label = contacts / (key + '.npz')
        if not label.is_file():
            raise FileNotFoundError(label)
        p, _, _, fps = load_php_npz(path, 'g1')
        if fps != 50 or len(p) < 3:
            raise ValueError(f'{path}: expected >=3 frames at 50 Hz')
        terrain = np.load(terrain_path(str(path)))
        boxes = load_terrain(str(path))  # validates quaternion, dimensions and finite values
        representatives, aliases = unique_variants(terrain)
        with np.load(label) as z:
            if list(z['body_names']) != list(CONTACT_NAMES) or z['contact_dist'].shape[1:] != (len(p), 9):
                raise ValueError(f'{label}: incompatible body names/frame count')
            label_variants = z['contact_dist'].shape[0]
            if label_variants != len(boxes) and not (label_variants == 1 and representatives == [0]):
                raise ValueError(f'{label}: incomplete terrain variant labels')
            if not np.allclose(z['boxes'], boxes[:label_variants], atol=1e-6):
                raise ValueError(f'{label}: terrain differs from source')
        records.append(dict(key=key, source=str(path), labels=str(label), frames=len(p), fps=fps,
                            source_hash=sha256(path), terrain_hash=sha256(terrain_path(str(path))),
                            labels_hash=sha256(label), variants=len(boxes), unique_variants=representatives,
                            variant_aliases=aliases, label_variants=label_variants,
                            # A whole source file, all windows and variants always share one split.
                            family=key, category=path.parent.name))
    if not records:
        raise ValueError('no sources matched --only')
    return records


def farthest_points(points, count):
    if len(points) <= count:
        return np.asarray(points)
    selected = [int(np.argmin(points[:, 2]))]
    distance = np.linalg.norm(points - points[selected[0]], axis=1)
    for _ in range(count - 1):
        j = int(distance.argmax())
        selected.append(j)
        distance = np.minimum(distance, np.linalg.norm(points - points[j], axis=1))
    return points[selected]


def poses(kin, p, R, q):
    """Position-only traversal using the validated RobotKinematics tree and hinge anchors."""
    positions, rotations = [], []
    for i, parent in enumerate(kin.parent):
        if parent < 0:
            positions.append(p)
            rotations.append(R)
            continue
        Rp = rotations[parent]
        base_R = Rp @ kin.off_R[i]
        base_p = positions[parent] + (Rp @ kin.off_p[i, :, None]).squeeze(-1)
        j = kin.joint_idx[i]
        if j >= 0:
            rotation = base_R @ axis_angle_to_mat(kin.axes[i], q[..., j] - kin.q_ref[j])
            base_p = base_p + ((base_R - rotation) @ kin.anchors[i, :, None]).squeeze(-1)
        else:
            rotation = base_R
        positions.append(base_p)
        rotations.append(rotation)
    return torch.stack(positions, -2), torch.stack(rotations, -3)


def project_surface(point, normal, surface, boxes):
    """Project to the specified plane/box face, including yawed vertical sides."""
    point = np.asarray(point).copy()
    if surface == 0:
        point[2] = 0
        return point, np.array([0., 0., 1.])
    if not 1 <= surface <= len(boxes):
        raise ValueError(f'invalid contact surface {surface}')
    cx, cy, top, c, s, hx, hy, hz = boxes[surface - 1]
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    center, extent = np.array([cx, cy, top - hz]), np.array([hx, hy, hz])
    local = R.T @ (point - center)
    nlocal = R.T @ normal
    axis = int(np.argmax(np.abs(nlocal)))
    sign = 1. if nlocal[axis] >= 0 else -1.
    local = np.clip(local, -extent, extent)
    local[axis] = sign * extent[axis]
    return center + R @ local, sign * R[:, axis]


class FullBodyRetargeter:
    def __init__(self, device='cuda:0', config=None):
        self.device, self.config = device, config or SolverConfig()
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        self.src, self.dst = get_robot('g1'), get_robot('tienkung3_29dof')
        self.sk, self.tk = RobotKinematics(device, self.src), RobotKinematics(device, self.dst)
        self.si = [self.sk.body_names.index(self.src.contact_bodies[n][0]) for n in CONTACT_NAMES]
        self.ti = [self.tk.body_names.index(self.dst.contact_bodies[n][0]) for n in CONTACT_NAMES]
        self.st = self.sk.tracked_idx
        self.tt = self.tk.tracked_idx
        self.source_clouds = [link_cloud(self.src, n) for n in CONTACT_NAMES]
        self.target_clouds = [link_cloud(self.dst, n) for n in CONTACT_NAMES]
        self.source_soles = np.array([sole(self.src, n) for n in CONTACT_NAMES[:2]])
        self.target_soles = np.array([sole(self.dst, n) for n in CONTACT_NAMES[:2]])
        z, eye = self.tensor(np.zeros((1, 3))), self.tensor(np.eye(3)[None])
        self.sn = poses(self.sk, z, eye, self.tensor(self.src.default_q)[None])
        self.tn = poses(self.tk, z, eye, self.tensor(self.dst.default_q)[None])
        # Calibration comes from each model's neutral sole height, not a G1 magic constant.
        self.source_height = -float((self.sn[0][:, self.si[:2]] +
                                     (self.sn[1][:, self.si[:2]] @ self.tensor(self.source_soles)[..., None]).squeeze(-1))[..., 2].min())
        self.target_height = -float((self.tn[0][:, self.ti[:2]] +
                                     (self.tn[1][:, self.ti[:2]] @ self.tensor(self.target_soles)[..., None]).squeeze(-1))[..., 2].min())
        self.scale = self.target_height / self.source_height
        # Contact locations must accommodate each robot's hip/shoulder width,
        # just as free-limb targets already use its own segment lengths.
        source_mounts = self.sn[0][0, self.st].cpu().numpy()
        target_mounts = self.tn[0][0, self.tt].cpu().numpy()
        mounts = [1, 4, 1, 4, 8, 11, 8, 11, 0]
        parents = [0, 0, 0, 0, 7, 7, 7, 7, 0]
        self.contact_lateral_delta = ((target_mounts[mounts]-target_mounts[parents])[:, 1] -
                                      (source_mounts[mounts]-source_mounts[parents])[:, 1])
        self._collision_samples()

    def tensor(self, value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

    def collision_witnesses(self, motion, boxes, first=0):
        """Native convex-contact planes, refreshed after each local optimization round.

        Contact points are stored in each body's local coordinates so both sides
        receive gradients. World/terrain witnesses remain fixed. This is solver
        guidance only; the independent checker still decides acceptance.
        """
        from tools.check_php_pilot import scene
        model = scene(self.dst, boxes, self_collision=True)
        state = mujoco.MjData(model)
        qa, _ = self.dst.joint_addresses(model)
        rows = []
        for t in range(first, len(motion)):
            state.qpos[:7], state.qpos[qa] = motion[t, :7], motion[t, 7:]
            mujoco.mj_forward(model, state)
            for c in state.contact:
                ids = [int(model.geom_bodyid[g]) for g in c.geom]
                names = [model.body(b).name for b in ids]
                if frozenset(names) in STRUCTURAL_PAIRS or c.dist >= -.0015:
                    continue
                normal = c.frame[:3].copy()
                points = [c.pos - normal*c.dist/2, c.pos + normal*c.dist/2]
                local, owners = [], []
                for bid, name, point in zip(ids, names, points):
                    owners.append(self.tk.body_names.index(name) if bid else -1)
                    local.append(state.xmat[bid].reshape(3, 3).T @ (point-state.xpos[bid]) if bid else point.copy())
                rows.append((t, owners, local, normal))
        if not rows:
            return None
        times, owners, local, normals = zip(*rows)
        return (torch.tensor(times, device=self.device), torch.tensor(owners, device=self.device),
                self.tensor(local), self.tensor(normals))

    def terrain_clearance(self, motion, boxes):
        """Bounded, smooth root-height correction for foot-only support trajectories.

        Preserve the terrain, joints and horizontal foot locations. This removes
        small heel/toe hull penetrations missed by sampled losses. The subsequent
        independent check still rejects excessive separation, slip or side impact.
        """
        from tools.check_php_pilot import scene
        model = scene(self.dst, boxes, self_collision=True)
        state = mujoco.MjData(model)
        qa, _ = self.dst.joint_addresses(model)
        lift = np.zeros(len(motion))
        for t, frame in enumerate(motion):
            state.qpos[:7], state.qpos[qa] = frame[:7], frame[7:]
            mujoco.mj_forward(model, state)
            for c in state.contact:
                if 0 in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]) and abs(c.frame[2]) > .95:
                    lift[t] = max(lift[t], min(self.config.clearance_max_m,
                                             max(0., -.0005-c.dist)/abs(c.frame[2])))
        smoothed = lift.copy()
        for offset in range(1, min(5, len(motion))):
            weight = (1+np.cos(np.pi*offset/5))/2
            smoothed[offset:] = np.maximum(smoothed[offset:], weight*lift[:-offset])
            smoothed[:-offset] = np.maximum(smoothed[:-offset], weight*lift[offset:])
        result = motion.copy()
        result[:, 2] += smoothed
        return result, dict(max_lift_m=float(smoothed.max()), adjusted_frames=int((smoothed > 0).sum()))

    def _collision_samples(self):
        m = self.tk.model
        points, owners, centers, radii, bodies = [], [], [], [], []
        for g in range(m.ngeom):
            if not m.geom_conaffinity[g]:
                continue
            name = m.body(m.geom_bodyid[g]).name
            if name not in self.tk.body_names:
                continue
            pts = surface_points(m, g, spacing=.025)
            pts = m.geom_pos[g] + pts @ quat_mat(m.geom_quat[g]).T
            sampled = farthest_points(pts, 96 if 'ankle_roll' in name else 32)
            owner = self.tk.body_names.index(name)
            points.extend(sampled)
            owners.extend([owner] * len(sampled))
            center = pts.mean(0)
            hull = ConvexHull(pts)
            radius = float(np.min(-(hull.equations[:, :3] @ center + hull.equations[:, 3])))
            centers.append(center)
            radii.append(max(radius, 0.))
            bodies.append(owner)
        self.points = self.tensor(points)
        self.owners = torch.tensor(owners, device=self.device)
        self.centers, self.radii = self.tensor(centers), self.tensor(radii)
        self.center_bodies = torch.tensor(bodies, device=self.device)
        pairs = []
        for a, ba in enumerate(bodies):
            for b in range(a):
                bb = bodies[b]
                if ba == bb or self.tk.parent[ba] == bb or self.tk.parent[bb] == ba:
                    continue
                if frozenset((self.tk.body_names[ba], self.tk.body_names[bb])) in STRUCTURAL_PAIRS:
                    continue
                # Fixed descendants of the same welded assembly are not self-collision pairs.
                ma, mb = self.tk.body_ids[ba], self.tk.body_ids[bb]
                if m.body_weldid[ma] == m.body_weldid[mb]:
                    continue
                pairs.append((a, b))
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def prepare(self, record, variant=0, start=0, end=None):
        p, quat, q, fps = load_php_npz(record['source'], self.src)
        end = len(p) if end is None else end
        if not 0 <= start < end <= len(p) or end - start < 3:
            raise ValueError('invalid source frame interval')
        p, quat, q = p[start:end], quat[start:end], q[start:end]
        p0, R0 = self.tensor(p), quat_to_mat(self.tensor(quat))
        P, Rs = poses(self.sk, p0, R0, self.tensor(q))
        boxes = load_terrain(record['source'])[variant]
        with np.load(record['labels']) as z:
            lv = min(variant, record['label_variants'] - 1)
            dist = z['contact_dist'][lv, start:end].copy()
            source_points = z['contact_pos_w'][lv, start:end].copy()
            normals = z['contact_normal_w'][lv, start:end].copy()
            surfaces = z['contact_surface'][lv, start:end].copy()
            if not np.allclose(P[:, self.si].cpu(), z['body_pos_w'][start:end], atol=3e-5):
                raise ValueError('source labels do not match current G1 FK')
        from .source_contacts import validate_zero_contacts
        dist, source_points, normals, surfaces, source_contact_validation = validate_zero_contacts(
            self.src, p, quat, q, boxes, dist, source_points, normals, surfaces, first=start)
        source_body_positions = P[:, self.si].cpu().numpy()
        contact_positions = source_body_positions.copy()
        contact_rotations = Rs[:, self.si].cpu().numpy()
        contact_positions[:, :2] += (contact_rotations[:, :2] @ self.source_soles[..., None]).squeeze(-1)
        orientations = (Rs[:, self.si] @ self.sn[1][:, self.si].transpose(-1, -2) @ self.tn[1][:, self.ti]).cpu().numpy()
        orientations[:, 4:6] = contact_rotations[:, 4:6] @ HAND_BASIS.T
        # Use source root heading for lower limbs and torso heading for upper
        # limbs. Shifts are horizontal; projection retains the original surface.
        heading_frames = [self.st[0]]*4 + [self.st[7]]*4 + [self.st[0]]
        forward = Rs[:, heading_frames, :2, 0].cpu().numpy()
        forward /= np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)
        lateral = np.concatenate((-forward[..., 1:2], forward[..., :1], np.zeros_like(forward[..., :1])), -1)
        support_shifts = lateral*self.contact_lateral_delta[None, :, None]
        source_slip = material_tangent_speed(P[:, self.si].cpu().numpy(), contact_rotations,
                                              dist, source_points, normals, fps)
        mask = np.zeros(dist.shape, bool)
        anchors, normal_targets, offsets = np.zeros((*dist.shape, 3)), np.zeros((*dist.shape, 3)), np.zeros((*dist.shape, 3))
        segments, incidental_segments, contact_aliases = [], [], []
        for b, name in enumerate(CONTACT_NAMES):
            speed = np.linalg.norm(np.gradient(contact_positions[:, b], axis=0), axis=-1) * fps
            active = ((dist[:, b] < .009) | ((dist[:, b] < .035) & (speed < .35))) if b < 2 else dist[:, b] <= .005
            active &= np.isfinite(source_points[:, b]).all(-1) & np.isfinite(normals[:, b]).all(-1)
            closed = binary_closing(active, iterations=2)
            closed[:2], closed[-2:] = active[:2], active[-2:]
            # Never bridge missing witnesses or a change of contact surface.
            closed &= np.isfinite(source_points[:, b]).all(-1) & np.isfinite(normals[:, b]).all(-1)
            for surface in sorted(set(surfaces[closed, b].tolist())):
                if surface < 0:
                    continue
                for a, z in runs(closed & (surfaces[:, b] == surface)):
                    if z - a < 4:
                        continue
                    body_points = source_points[a:z, b] + dist[a:z, b, None]*normals[a:z, b]
                    if b in (2, 3):
                        foot = b-2
                        same_support = mask[a:z, foot] & (surfaces[a:z, foot] == surface)
                        alias, anatomical = distal_shank_alias(source_body_positions[a:z, b],
                            source_body_positions[a:z, foot], body_points, same_support)
                        if alias:
                            contact_aliases.append(dict(body=name, mapped_to=CONTACT_NAMES[foot], start=int(a+start),
                                end=int(z+start), surface=int(surface), reason='distal_shank_at_supported_ankle', **anatomical))
                            continue
                    source_speed = float(np.median(source_slip[a:z-1, b]))
                    if is_brief_graze(name, (z-a)/fps, source_speed):
                        incidental_segments.append(dict(body=name, start=int(a+start), end=int(z+start),
                            surface=int(surface), source_material_slip_median_m_s=source_speed,
                            classification='brief_fast_graze', target_requirement='collision_free'))
                        continue
                    normal = np.median(normals[a:z, b], axis=0)
                    if np.linalg.norm(normal) < .1:
                        continue
                    normal /= np.linalg.norm(normal)
                    point = np.median(source_points[a:z, b], axis=0)
                    if b < 2 and normal[2] > .95:
                        point[:2] = np.median(contact_positions[a:z, b, :2], axis=0)
                    morphology_shift = np.median(support_shifts[a:z, b], axis=0)
                    point += morphology_shift
                    anchor, normal = project_surface(point, normal, surface, boxes)
                    if b < 2 and normal[2] > .95:
                        yaw = np.unwrap(np.arctan2(contact_rotations[a:z, b, 1, 0], contact_rotations[a:z, b, 0, 0])).mean()
                        c, s = np.cos(yaw), np.sin(yaw)
                        orientations[a:z, b] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                        offset = self.target_soles[b]
                    else:
                        # Preserve the source contact's anatomical surface region.
                        direction = orientations[a:z, b].mean(0).T @ normal
                        source_local = (contact_rotations[a:z, b].transpose(0, 2, 1) @
                                        (body_points-source_body_positions[a:z, b])[..., None]).squeeze(-1)
                        offset = corresponding_patch(np.median(source_local, axis=0), self.source_clouds[b],
                                                     self.target_clouds[b], direction, HAND_BASIS if b in (4, 5) else None)
                    mask[a:z, b] = True
                    anchors[a:z, b], normal_targets[a:z, b], offsets[a:z, b] = anchor, normal, offset
                    segments.append(dict(body=name, start=int(a + start), end=int(z + start), surface=int(surface),
                                         anchor=anchor.tolist(), normal=normal.tolist(), local_point=offset.tolist(),
                                         morphology_shift_world=morphology_shift.tolist(),
                                         anchor_mapping='neutral-mount-lateral-difference-v1'))
        proot = p0.clone()
        proot[:, 2] += self.target_height - self.source_height
        body_target, arm_target_local = segment_targets(P[:, self.st].cpu().numpy(),
                                                        Rs[:, self.st].cpu().numpy(), proot.cpu().numpy())
        foot_target = contact_positions[:, :2].copy() + support_shifts[:, :2]
        for b in range(2):
            foot_target[:, b, 2] = np.maximum(foot_target[:, b, 2], support_height(foot_target[:, b, :2], boxes))
            foot_target[mask[:, b], b] = anchors[mask[:, b], b]
        torso_orientation = (Rs[:, self.st[7]] @ self.sn[1][:, self.st[7]].transpose(-1, -2)
                             @ self.tn[1][:, self.tt[7]])
        return dict(root=proot.cpu().numpy(), R=R0.cpu().numpy(), body_target=body_target,
                    torso_orientation=torso_orientation.cpu().numpy(),
                    arm_target_local=arm_target_local, arm_free=free_arm_mask(mask),
                    arm_weight=contact_transition_weights(free_arm_mask(mask)),
                    orientations=orientations, contact=mask, anchors=anchors, normals=normal_targets,
                    offsets=offsets, surfaces=surfaces, foot_target=foot_target, boxes=boxes, segments=segments,
                    incidental_segments=incidental_segments,
                    contact_aliases=contact_aliases, source_contact_validation=source_contact_validation,
                    source_frames=[start, end], arm_mode='tracked')

    def solve(self, prepared, checkpoint=None, identity=None):
        cfg, T = self.config, len(prepared['root'])
        torch.manual_seed(cfg.seed)
        result = np.zeros((T, 36), np.float32)
        completed, stats = 0, []
        if checkpoint and Path(checkpoint).exists():
            with np.load(checkpoint) as z:
                if str(z['identity'].item()) != identity:
                    raise ValueError('window checkpoint identity mismatch')
                result, completed = z['motion'].copy(), int(z['completed'])
                stats = json.loads(str(z['stats'].item()))
                if result.shape != (T, 36) or not 0 < completed <= T or not np.isfinite(result[:completed]).all():
                    raise ValueError('invalid window checkpoint')
        while completed < T:
            start = max(0, completed - cfg.overlap)
            end = min(T, start + cfg.window_frames)
            window = {k: v[start:end] if isinstance(v, np.ndarray) and k != 'boxes' else v
                      for k, v in prepared.items()}
            fixed = result[start:completed].copy()
            motion, metrics = self.solve_window(window, fixed)
            result[start:end], completed = motion, end
            metrics.update(start=start, end=end, fixed_frames=len(fixed))
            stats.append(metrics)
            if checkpoint:
                atomic_npz(checkpoint, identity=identity, motion=result, completed=completed,
                           stats=json.dumps(stats))
        if not prepared['contact'][:, 2:].any():
            result, clearance = self.terrain_clearance(result, prepared['boxes'])
            stats[-1]['terrain_clearance'] = clearance
        from .native_pose_polish import polish
        result, native_polish = polish(self.dst, result.astype(np.float64), prepared['boxes'],
                                       prepared['contact'], prepared['offsets'])
        stats[-1]['native_polish'] = native_polish
        return result, stats

    def initialize_pose(self, data):
        """Fit source segment directions before adding terrain/contact constraints.

        This only supplies a starting point; source timing, support intent and
        all independent acceptance checks remain unchanged. Root pose is fixed
        in this stage, and destination joints are fit through destination FK.
        """
        root, R, target = (self.tensor(data[k]) for k in ('root', 'R', 'body_target'))
        torso = self.tensor(data['torso_orientation'])
        # Mount positions differ between models; compare articulated bone
        # vectors rather than requiring all body origins to coincide.
        children = [2, 3, 5, 6, 9, 10, 12, 13]
        parents = [1, 2, 4, 5, 8, 9, 11, 12]
        reference = target[:, children]-target[:, parents]
        lengths = reference.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        reference = reference/lengths
        child_ids = torch.tensor([self.tt[i] for i in children], device=self.device)
        parent_ids = torch.tensor([self.tt[i] for i in parents], device=self.device)
        q = self.tensor(self.dst.default_q)[None].repeat(len(root), 1).requires_grad_()
        default = q.detach().clone()
        cuda = self.device.startswith('cuda')
        lr = torch.tensor(.025, device=self.device) if cuda else .025
        opt = torch.optim.Adam((q,), lr=lr, capturable=cuda)
        loss_value = torch.zeros((), device=self.device)
        def step():
            opt.zero_grad(set_to_none=True)
            P, Rs = poses(self.tk, root, R, q)
            bones = (P[:, child_ids]-P[:, parent_ids])/lengths
            loss = (bones-reference).square().mean()
            loss = loss + (Rs[:, self.tt[7]]-torso).square().mean()
            loss = loss + .002*(q[1:]-q[:-1]).square().mean()
            loss = loss + .0001*(q-default).square().mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                q.clamp_(self.tk.q_lo+.001, self.tk.q_hi-.001)
                loss_value.copy_(loss.detach())
        graph = None
        if cuda and self.config.graph:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
        for i in range(500):
            rate = .0002+.0248*(1+np.cos(np.pi*i/500))/2
            if cuda:
                lr.fill_(rate)
            else:
                opt.param_groups[0]['lr'] = rate
            step() if graph is None else graph.replay()
        if not torch.isfinite(q).all():
            raise FloatingPointError('nonfinite pose initialization')
        return q.detach(), float(loss_value)

    def solve_window(self, data, fixed):
        cfg, device = self.config, self.device
        root, R0, body = (self.tensor(data[k]) for k in ('root', 'R', 'body_target'))
        contact, anchors, offsets, orientations, foot_target = (self.tensor(data[k]) for k in
                                      ('contact', 'anchors', 'offsets', 'orientations', 'foot_target'))
        boxes = self.tensor(data['boxes'])
        default, velocity = self.tensor(self.dst.default_q), self.tensor(self.dst.velocity)
        arm_target, arm_free = self.tensor(data['arm_target_local']), self.tensor(data['arm_free'])
        arm_indices = torch.tensor([self.tt[i] for i in ARM_POINTS], device=device)
        torso_index = self.tt[7]
        torso_orientation = self.tensor(data['torso_orientation'])
        arm_weight = self.tensor(data['arm_weight'])
        arm_consecutive = arm_weight[1:] * arm_weight[:-1]
        dp, dr = torch.zeros_like(root, requires_grad=True), torch.zeros_like(root, requires_grad=True)
        q, initialization_loss = self.initialize_pose(data)
        q.requires_grad_()
        nfixed = len(fixed)
        fixed_motion = self.tensor(fixed)
        if nfixed:
            from ..rotations import so3_log
            fixed_dr = so3_log(R0[:nfixed].transpose(-1, -2) @ quat_to_mat(fixed_motion[:, 3:7]))
            with torch.no_grad():
                dp[:nfixed] = fixed_motion[:, :3] - root[:nfixed]
                dr[:nfixed] = fixed_dr
                q[:nfixed] = fixed_motion[:, 7:]
        lr = torch.tensor(cfg.learning_rate, device=device) if device.startswith('cuda') else cfg.learning_rate
        opt = torch.optim.Adam((dp, dr, q), lr=lr, capturable=device.startswith('cuda'))
        foot_offsets = self.tensor(self.target_soles)
        contact_indices = torch.tensor(self.ti, device=device)
        tracked_indices = torch.tensor(self.tt, device=device)
        foot_indices = contact_indices[:2]
        count = contact.sum().clamp(min=1)
        start_time = time.monotonic()
        losses = torch.zeros(6, device=device)
        witnesses = None
        normals = self.tensor(data['normals'])
        consecutive = self.tensor(data['contact'][1:] & data['contact'][:-1] &
                                  (data['surfaces'][1:] == data['surfaces'][:-1]) &
                                  (np.linalg.norm(data['offsets'][1:] - data['offsets'][:-1], axis=-1) < 1e-6))

        def step():
            opt.zero_grad(set_to_none=True)
            R = R0 @ so3_exp(dr)
            P, Rs = poses(self.tk, root + dp, R, q)
            points = P[:, self.owners] + (Rs[:, self.owners] @ self.points[..., None]).squeeze(-1)
            distances = points[..., 2]
            for box in boxes:
                distances = torch.minimum(distances, box_sdf(points, box[None].expand(len(root), -1)))
            penetration = torch.relu(-distances).square().amax(-1).mean()
            positions = P[:, contact_indices] + (Rs[:, contact_indices] @ offsets[..., None]).squeeze(-1)
            # Balance body groups: short knee/hand support must not be diluted
            # by the many frames of foot support in a long climbing sequence.
            body_counts = contact.sum(0)
            contact_error = (((positions - anchors).square().sum(-1) * contact).sum(0)
                             / body_counts.clamp(min=1)).sum() / (body_counts > 0).sum().clamp(min=1)
            feet = P[:, foot_indices] + (Rs[:, foot_indices] @ foot_offsets[..., None]).squeeze(-1)
            foot_error = (feet - foot_target).square().sum(-1).mean()
            body_error = (P[:, tracked_indices] - body).square().mean()
            arm_local = (Rs[:, torso_index, None].transpose(-1, -2) @
                         (P[:, arm_indices]-P[:, torso_index, None])[..., None]).squeeze(-1)
            arm_error = ((arm_local-arm_target).square().sum(-1)*arm_weight).sum()/arm_weight.sum().clamp(min=1)
            arm_delta = ((arm_local[1:]-arm_local[:-1])-(arm_target[1:]-arm_target[:-1]))*50
            arm_velocity_error = (arm_delta.square().sum(-1)*arm_consecutive).sum()/arm_consecutive.sum().clamp(min=1)
            rotation = ((Rs[:, contact_indices] - orientations).square().mean((-1, -2)) * (contact + .05)).mean()
            foot_rotation = (Rs[:, foot_indices] - orientations[:, :2]).square().mean()
            centers = P[:, self.center_bodies] + (Rs[:, self.center_bodies] @ self.centers[..., None]).squeeze(-1)
            a, b = self.pairs[:, 0], self.pairs[:, 1]
            self_overlap = torch.relu(self.radii[a] + self.radii[b] -
                                      (centers[:, a] - centers[:, b]).square().sum(-1).clamp(min=1e-12).sqrt()).square().amax(-1).mean()
            smooth = (q[1:] - q[:-1]).square().mean() + 20 * (q[2:] - 2*q[1:-1] + q[:-2]).square().mean()
            root_temporal = root_correction_cost(dp)
            speed = torch.relu((q[1:] - q[:-1]).abs()*50 - .8*velocity).square().mean()
            loss = 900*contact_error + 100*foot_error + 2*body_error + rotation + 2*foot_rotation + 6000*penetration
            loss = loss + 3000*self_overlap + .5*smooth + .5*speed + .01*dp.square().mean() + .01*dr.square().mean() + .0001*(q-default).square().mean()
            loss = loss + 200*arm_error + .1*arm_velocity_error + root_temporal
            # Explicitly preserve source torso and pelvis attitude. Torso-local
            # arm tracking is invariant to a spurious whole-torso lean.
            torso_frame_error = (Rs[nfixed:, torso_index]-torso_orientation[nfixed:]).square().mean((-1, -2))
            torso_error = torso_frame_error.mean() + .1*torso_frame_error.amax()
            pelvis_error = (R[nfixed:]-R0[nfixed:]).square().mean()
            loss = loss + 80*torso_error + 5*pelvis_error
            # Sliding is a contact constraint even when no collision witnesses
            # exist. Previously this term vanished as soon as geometry cleared.
            delta = (positions[1:]-positions[:-1])*50
            tangent = delta - (delta*normals[1:]).sum(-1, keepdim=True)*normals[1:]
            phase_counts = consecutive.sum(0)
            slip = (torch.relu(tangent.norm(dim=-1)-.025).square()*consecutive).sum(0)
            loss = loss + 20*(slip/phase_counts.clamp(min=1)).sum()/(phase_counts > 0).sum().clamp(min=1)
            if witnesses is not None:
                times, owners, local_points, witness_normals = witnesses
                wp = P[times[:, None], owners.clamp(min=0)] + (Rs[times[:, None], owners.clamp(min=0)] @ local_points[..., None]).squeeze(-1)
                wp = torch.where((owners < 0)[..., None], local_points, wp)
                gap = ((wp[:, 1]-wp[:, 0])*witness_normals).sum(-1)
                # Each frame/body pair contributes, without amplifying repeated
                # historical planes for the same colliding pair.
                loss = loss + 100000*pair_collision_cost(
                    gap, witness_groups, witness_group_count, len(root))
                loss = loss + 5*speed
            loss.backward()
            opt.step()
            with torch.no_grad():
                q.clamp_(self.tk.q_lo + .001, self.tk.q_hi - .001)
                if nfixed:
                    dp[:nfixed].copy_(fixed_motion[:, :3] - root[:nfixed])
                    dr[:nfixed].copy_(fixed_dr)
                    q[:nfixed].copy_(fixed_motion[:, 7:])
                losses.copy_(torch.stack((loss, contact_error, penetration, self_overlap, speed, foot_error)).detach())

        graph, iteration = None, 0
        if cfg.graph and device.startswith('cuda') and cfg.iterations >= 5:
            # Capture once per window. lr is a device tensor updated before every replay.
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            iteration = 3  # capture records kernels; only the three warm-up updates executed
        stable, previous = 0, None
        for it in range(iteration, cfg.iterations):
            rate = .0001 + (cfg.learning_rate - .0001) * (1 + np.cos(np.pi * it / cfg.iterations)) / 2
            if isinstance(lr, torch.Tensor):
                lr.fill_(rate)
            else:
                opt.param_groups[0]['lr'] = rate
            if graph is None:
                step()
            else:
                graph.replay()
            iteration = it + 1
            if iteration % 100 == 0 or iteration == cfg.iterations:
                values = losses.detach().cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError('nonfinite optimization loss')
                stable = stable + 1 if previous is not None and abs(values[0] - previous) < cfg.convergence_rtol*max(abs(previous), 1e-6) else 0
                previous = float(values[0])
                if iteration >= cfg.min_iterations and stable >= 3:
                    break
        # Refresh exact collision witnesses rather than relying on inscribed spheres
        # for narrow features such as crossing shanks, heels and box edges.
        collision_stats = []
        if cfg.collision_rounds:
            for round_index in range(cfg.collision_rounds):
                with torch.no_grad():
                    current = torch.cat((root+dp, mat_to_quat(R0 @ so3_exp(dr)), q), -1).cpu().numpy()
                refreshed = self.collision_witnesses(current, data['boxes'], nfixed)
                if refreshed is None:
                    break
                # Retain previously corrected planes: dropping them lets the
                # tracking objective push heels/box edges back into collision.
                witnesses = refreshed if witnesses is None else tuple(
                    torch.cat((old, new), dim=0) for old, new in zip(witnesses, refreshed))
                # Preserve gradients for every colliding body pair; repeated
                # historical planes for the same frame/pair still count once.
                pairs = torch.sort(witnesses[1], dim=-1).values
                groups, witness_groups = torch.unique(
                    torch.cat((witnesses[0][:, None], pairs), dim=1),
                    dim=0, return_inverse=True)
                witness_group_count = len(groups)
                # Reset Adam moments because the local constraints changed.
                opt = torch.optim.Adam((dp, dr, q), lr=lr, capturable=device.startswith('cuda'))
                if isinstance(lr, torch.Tensor):
                    lr.fill_(.001)
                else:
                    opt.param_groups[0]['lr'] = .001
                graph = None
                if cfg.graph and device.startswith('cuda'):
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            step()
                    torch.cuda.current_stream(device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        step()
                for polish_it in range(cfg.collision_iterations):
                    rate = .00002 + .00098*(1+np.cos(np.pi*polish_it/cfg.collision_iterations))/2
                    if isinstance(lr, torch.Tensor):
                        lr.fill_(rate)
                    else:
                        opt.param_groups[0]['lr'] = rate
                    step() if graph is None else graph.replay()
                collision_stats.append(dict(round=round_index, witnesses=len(witnesses[0]), loss=float(losses[0])))
        with torch.no_grad():
            motion = torch.cat((root + dp, mat_to_quat(R0 @ so3_exp(dr)), q), -1).cpu().numpy()
        if nfixed:
            motion[:nfixed] = fixed  # preserve exact quaternion signs and committed overlap bytes
        if not np.isfinite(motion).all():
            raise FloatingPointError('nonfinite output trajectory')
        values = losses.cpu().tolist()
        return motion, dict(root_temporal_policy="source-relative-derivatives-v1", arm_transition_frames=8, initialization_loss=initialization_loss, iterations=iteration, seconds=time.monotonic()-start_time, loss=values[0],
                            contact_mse=values[1], penetration_mse=values[2], self_overlap_mse=values[3],
                            speed_loss=values[4], foot_mse=values[5], collision_refinement=collision_stats, cuda_graph=graph is not None,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0)
