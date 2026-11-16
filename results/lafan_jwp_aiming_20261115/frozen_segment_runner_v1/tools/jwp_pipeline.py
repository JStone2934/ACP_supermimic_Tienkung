"""JWP BVH -> local ACP TienKung 29-DoF references and native contact labels.

Run through ``python -m tools.convert_php`` in acp_supermimic, so the existing
host memory queue owns every conversion controller, including CPU-only runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
VERSION = "jwp-gmr-acp-v4-axis-calibrated"
POLICY = dict(contact_threshold_m=.005, max_contact_distance_m=.1,
              joint_limit_tolerance_rad=1e-4, joint_velocity_max_ratio=1.,
              penetration_max_m=.005, penetration_persistent_m=.002,
              penetration_max_consecutive_frames=2, support_distance_p95_m=.01,
              support_slip_p95_m_s=.05, human_support_height_m=.04,
              human_support_speed_m_s=.25, fps=50., ground_clearance_m=.001,
              scope="Kinematic references on a flat ground plane; not a dynamics certificate")
BASELINE_PAIRS = {frozenset((f"knee_pitch_{s}_link", f"ankle_roll_{s}_link")) for s in ("l", "r")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def configure(acp_root, gmr_root):
    for p in (str(gmr_root), str(acp_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    torch.set_num_threads(1)


def read_header(path):
    bones, channels = [], []
    frames = None
    with Path(path).open() as f:
        for line in f:
            s = line.strip()
            if s.startswith(("ROOT ", "JOINT ")):
                bones.append(s.split()[1])
            elif s.startswith("CHANNELS "):
                channels.append(s.split()[2:])
            elif s.startswith("Frames:"):
                frames = int(s.split(":", 1)[1])
            elif s.startswith("Frame Time:"):
                dt = float(s.split(":", 1)[1])
                first = np.fromstring(next(f), sep=" ")
                break
        else:
            raise ValueError(f"missing Frame Time: {path}")
    if len(set(bones)) != len(bones) or len(bones) != len(channels):
        raise ValueError("duplicate bones or missing channel declarations")
    if not frames or frames < 4 or not 0 < dt < 1 or first.size != sum(map(len, channels)):
        raise ValueError("invalid BVH header / first frame")
    cursor, rotations = 0, []
    for names in channels:
        rotations.extend(first[cursor+i] for i, name in enumerate(names) if name.endswith("rotation"))
        cursor += len(names)
    calibration = bool(np.max(np.abs(rotations)) < 1e-5)
    return dict(frames=frames, frame_time=dt, fps=1/dt, bones=bones,
                channels=channels, leading_reference_frames=int(calibration))


def load_human(path):
    """Resample local BVH poses, then FK: preserve bone lengths and real timing."""
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
    from general_motion_retargeting.utils.lafan_vendor import utils
    h = read_header(path)
    expected_channels = ["Xposition", "Yposition", "Zposition", "Yrotation", "Xrotation", "Zrotation"]
    if any(channels != expected_channels for channels in h["channels"]):
        raise ValueError("JWP adapter requires per-joint XYZ translation then intrinsic YXZ rotation")
    anim = read_bvh(str(path))
    first = h["leading_reference_frames"]
    pos, quat = anim.pos[first:], anim.quats[first:]
    if len(pos) != h["frames"]-first:
        raise ValueError("BVH frame count mismatch")
    source_t = np.arange(len(pos))*h["frame_time"]
    target_t = np.arange(int(np.floor(source_t[-1]*POLICY["fps"]+1e-9))+1)/POLICY["fps"]
    sample_t = np.minimum(target_t, source_t[-1])  # decimal BVH dt may round an exact endpoint by 1e-15 s
    positions = np.empty((len(target_t), pos.shape[1], 3))
    quats = np.empty((len(target_t), quat.shape[1], 4))
    for b in range(pos.shape[1]):
        for k in range(3):
            positions[:, b, k] = np.interp(sample_t, source_t, pos[:, b, k])
        quats[:, b] = Slerp(source_t, Rotation.from_quat(quat[:, b, [1, 2, 3, 0]]))(sample_t).as_quat()[:, [3, 0, 1, 2]]
    global_q, global_p = utils.quat_fk(quats, positions, anim.parents)
    basis = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    basis_q = Rotation.from_matrix(basis).as_quat(scalar_first=True)
    global_p = global_p @ basis.T / 100.
    global_q = utils.quat_mul(basis_q, global_q)
    names = list(anim.bones)
    required = ["Hips", "Spine2", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg",
                "LeftFoot", "RightFoot", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]
    if set(required)-set(names):
        raise ValueError(f"missing bones: {set(required)-set(names)}")
    frames = []
    for t in range(len(target_t)):
        row = {name: [global_p[t, i].copy(), global_q[t, i].copy()] for i, name in enumerate(names)}
        for side in ("Left", "Right"):
            orientation = next((n for n in (side+"Toe", side+"ToeBase", side+"Foot") if n in row))
            row[side+"FootMod"] = [row[side+"Foot"][0].copy(), row[orientation][1].copy()]
        frames.append(row)
    feet = global_p[:, [names.index("LeftFoot"), names.index("RightFoot")]]
    smooth_feet = savgol_filter(feet, min(7, (len(feet)-1)//2*2+1), 2, axis=0)
    speed = np.linalg.norm(np.gradient(smooth_feet[..., :2], axis=0)*POLICY["fps"], axis=-1)
    floor = np.percentile(smooth_feet[..., 2], 5, axis=0)
    contact = (smooth_feet[..., 2] < floor + POLICY["human_support_height_m"]) & (speed < POLICY["human_support_speed_m_s"])
    for b in range(2):
        padded = np.pad(contact[:, b], 4, mode="edge")
        contact[:, b] = binary_opening(binary_closing(padded, iterations=2), iterations=1)[4:-4]
    provenance = dict(source_frames=h["frames"], source_fps=h["fps"], source_frame_time=h["frame_time"],
                      source_frame_interval=[first, h["frames"]], removed_reference_frames=first,
                      source_motion_duration_s=float(source_t[-1]), output_duration_s=float(target_t[-1]),
                      output_frames=len(target_t), output_fps=POLICY["fps"],
                      coordinate_conversion="centimetres Y-up -> metres Z-up: (x,-z,y)",
                      toe_fallback="LeftFoot/RightFoot orientation when toe joints are absent")
    return frames, contact, provenance


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def longest(mask):
    return max((z-a for a, z in runs(mask)), default=0)


def calibration_offsets(robot):
    """Match the Axis/Noitom zero-rotation T pose to the native robot T pose.

    BVH rest: left=+X, up=+Y, forward=+Z. After B=Rx(90), the
    robot must face world -Y, with left along +X. Each offset is computed
    as B^-1 Rz(-90) R_body(T), NOT copied from LAFAN's different bone axes.
    """
    import mujoco
    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for side, sign in (("l", 1), ("r", -1)):
        # The MJCF already has +/-5 degrees shoulder-roll in the body transform.
        data.qpos[model.joint(f"shoulder_roll_{side}_joint").qposadr] = sign*np.deg2rad(85.)
    mujoco.mj_kinematics(model, data)
    basis = Rotation.from_euler("x", 90, degrees=True)
    heading = Rotation.from_euler("z", -90, degrees=True)
    return {model.body(i).name: (basis.inv()*heading*Rotation.from_quat(data.xquat[i], scalar_first=True)).as_quat(scalar_first=True).tolist()
            for i in range(1, model.nbody)}


def make_ik_config(gmr_root, destination, robot):
    spec = importlib.util.spec_from_file_location("upstream_tiangong_retarget", ROOT/"scripts/tiangong3/retarget_lafan1_gmr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE_IK_CONFIG = Path(gmr_root)/"general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json"
    module.build_ik_config(destination)
    config = json.loads(destination.read_text())
    offsets = calibration_offsets(robot)
    for table in ("ik_match_table1", "ik_match_table2"):
        for body, entry in config[table].items():
            entry[4] = offsets[body]
        for side in ("l", "r"):
            config[table][f"ankle_roll_{side}_link"][1] = 2000.
            config[table][f"ankle_roll_{side}_link"][2] = 2000.
    atomic_json(destination, config)


def geom_lowest(model, data, ids):
    import mujoco
    lowest = np.inf
    for g in ids:
        R = data.geom_xmat[g].reshape(3, 3)
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            low = np.min(model.mesh_vert[a:a+n] @ R[2])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            low = -model.geom_size[g, 0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            low = -model.geom_size[g, 0]-abs(R[2, 2])*model.geom_size[g, 1]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            low = -model.geom_size[g, 0]*np.linalg.norm(R[2, :2])-abs(R[2, 2])*model.geom_size[g, 1]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            low = -np.abs(R[2]) @ model.geom_size[g]
        else:
            raise ValueError(f"unsupported robot collision geom {kind}")
        lowest = min(lowest, data.geom_xpos[g, 2]+low)
    return float(lowest)


def retarget(frames, robot, config_path, human_height, source_contact):
    import mujoco
    import mink
    from general_motion_retargeting import GeneralMotionRetargeting, params
    params.ROBOT_XML_DICT["acp_tienkung3_29dof"] = Path(robot.mjcf_path)
    params.IK_CONFIG_DICT["bvh_lafan1"]["acp_tienkung3_29dof"] = Path(config_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = GeneralMotionRetargeting(src_human="bvh_lafan1", tgt_robot="acp_tienkung3_29dof",
                                         actual_human_height=human_height, verbose=False)
    model = solver.model
    model.opt.timestep = 1/POLICY["fps"]
    qadr, vadr = robot.joint_addresses(model)
    initial = model.qpos0.copy()
    initial[:3] = [0, 0, robot.stand_height]
    initial[qadr] = robot.default_q
    solver.configuration.update(initial)
    active_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    pairs = []
    for i, a in enumerate(active_geoms):
        for b in active_geoms[:i]:
            names = frozenset((model.body(model.geom_bodyid[a]).name, model.body(model.geom_bodyid[b]).name))
            if names not in BASELINE_PAIRS:
                pairs.append(([a], [b]))
    collision_limit = mink.CollisionAvoidanceLimit(model, pairs, minimum_distance_from_collisions=.001,
                                                  collision_detection_distance=.03)
    solver.ik_limits.append(collision_limit)
    # Compute support-plane offsets from this exact ACP collision model.
    foot_groups, foot_clearance = [], []
    for side in ("left", "right"):
        link, links = robot.contact_bodies[side+"_foot"]
        group = [g for g in active_geoms if model.body(model.geom_bodyid[g]).name in links]
        foot_groups.append(group)
        foot_clearance.append(solver.configuration.data.xpos[model.body(link).id, 2]
                              - geom_lowest(model, solver.configuration.data, group))
    original_update = solver.update_targets
    target_positions, target_rotations = [], []
    for frame in frames:
        original_update(copy.deepcopy(frame))
        target_positions.append([solver.scaled_human_data[n][0].copy() for n in ("LeftFootMod", "RightFootMod")])
        target_rotations.append([solver.scaled_human_data[n][1].copy() for n in ("LeftFootMod", "RightFootMod")])
    target_positions, target_rotations = np.array(target_positions), np.array(target_rotations)
    for b in range(2):
        support_floor = np.percentile(target_positions[source_contact[:, b], b, 2], 50) if source_contact[:, b].any() else np.percentile(target_positions[:, b, 2], 5)
        target_positions[:, b, 2] += foot_clearance[b]+POLICY["ground_clearance_m"]-support_floor
        for a, z in runs(source_contact[:, b]):
            target_positions[a:z, b, :2] = np.median(target_positions[a:z, b, :2], axis=0)
            target_positions[a:z, b, 2] = foot_clearance[b]+POLICY["ground_clearance_m"]
            rotation = Rotation.from_quat(target_rotations[a:z, b][:, [1, 2, 3, 0]]).mean()
            yaw = rotation.as_euler("xyz")[2]
            target_rotations[a:z, b] = Rotation.from_euler("z", yaw).as_quat(scalar_first=True)
    current_frame = [0]
    def update_with_support(frame, offset_to_ground=False):
        original_update(frame, offset_to_ground)
        t = current_frame[0]
        for b, name in enumerate(("LeftFootMod", "RightFootMod")):
            target = mink.SE3.from_rotation_and_translation(mink.SO3(target_rotations[t, b]), target_positions[t, b])
            for table in (solver.human_body_to_task1, solver.human_body_to_task2):
                table[name].set_target(target)
    solver.update_targets = update_with_support
    # Start with the source heading, avoiding a different leg-twist IK branch.
    original_update(copy.deepcopy(frames[0]))
    initial[:3] = solver.scaled_human_data["Hips"][0]
    initial[3:7] = solver.scaled_human_data["Hips"][1]
    solver.configuration.update(initial)
    # Converge the first real pose before recording; never export solver warm-up.
    for _ in range(20):
        solver.retarget(copy.deepcopy(frames[0]))
    from mink.limits.limit import Constraint, Limit
    class FrameStepLimit(Limit):
        """Budget the total frame step, including both GMR IK stages/iterations."""
        def __init__(self):
            self.previous = solver.configuration.q.copy()
            self.bound = np.empty(model.nv)
            self.bound[:6] = np.array([1., 1., 1., 3., 3., 3.])/POLICY["fps"]
            self.bound[vadr] = np.asarray(robot.velocity)*.9/POLICY["fps"]
            self.matrix = np.r_[np.eye(model.nv), -np.eye(model.nv)]
        def compute_qp_inequalities(self, configuration, dt):
            delta = np.empty(model.nv)
            mujoco.mj_differentiatePos(model, delta, 1., self.previous, configuration.q)
            return Constraint(G=self.matrix, h=np.r_[self.bound-delta, self.bound+delta])
    frame_limit = FrameStepLimit()
    solver.ik_limits.append(frame_limit)
    output = []
    for index, frame in enumerate(frames):
        current_frame[0] = index
        frame_limit.previous = solver.configuration.q.copy()
        qp = solver.retarget(copy.deepcopy(frame))
        output.append(np.r_[qp[:7], qp[qadr]])
    raw = np.asarray(output)
    motion = raw.copy()
    lo = np.array([model.joint(n).range[0] for n in robot.joint_names])
    hi = np.array([model.joint(n).range[1] for n in robot.joint_names])
    motion[:, 7:] = np.clip(motion[:, 7:], lo, hi)
    data = mujoco.MjData(model)
    foot_links = set(robot.contact_bodies["left_foot"][1]) | set(robot.contact_bodies["right_foot"][1])
    foot_geoms = [g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g])
                  and model.body(model.geom_bodyid[g]).name in foot_links]
    shifts = []
    for row in motion:
        data.qpos[:7], data.qpos[qadr] = row[:7], row[7:]
        mujoco.mj_kinematics(model, data)
        shift = max(0., POLICY["ground_clearance_m"]-geom_lowest(model, data, foot_geoms))
        row[2] += shift
        shifts.append(shift)
    return motion, raw, dict(ground_shift_min_m=float(min(shifts)), ground_shift_max_m=float(max(shifts)),
                             smoothing="Local BVH interpolation followed by contact-constrained GMR",
                             collision_pairs_requested=len(pairs),
                             foot_clearance_m=foot_clearance, warmup_iterations=20)


def quality_and_labels(path, robot, source_contact, *, diagnostics=False):
    import mujoco
    from supermimic.acp.make_labels import variant_labels, PER_VARIANT
    from supermimic.acp.motion import load_php_npz
    from supermimic.acp.terrain import terrain_path
    empty = np.zeros((0, 8))
    one = variant_labels((str(path), empty, POLICY["contact_threshold_m"], POLICY["max_contact_distance_m"], robot))
    labels = {k: one[k][None] for k in PER_VARIANT}
    labels.update(body_pos_w=one["body_pos_w"], body_quat_w=one["body_quat_w"],
                  other_link_names=one["other_link_names"], fps=np.float32(POLICY["fps"]),
                  boxes=empty[None].astype(np.float32), threshold=np.float32(POLICY["contact_threshold_m"]),
                  body_names=np.array(list(robot.contact_bodies)),
                  link_names=np.array([x[0] for x in robot.contact_bodies.values()]),
                  robot_metadata=json.dumps(robot.metadata()), motion_hash=sha(path), terrain_hash=sha(terrain_path(str(path))))
    p, quat, q, fps = load_php_npz(path, robot)
    spec = robot.get_spec()
    for geom in spec.geoms:
        if geom.conaffinity:
            geom.contype = 1
    spec.worldbody.add_geom(name="jwp_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1])
    model = spec.compile()
    data = mujoco.MjData(model)
    qa, _ = robot.joint_addresses(model)
    terrain_pen, self_pen, raw_self = [], [], []
    pairs = {}
    for t in range(len(p)):
        data.qpos[:3], data.qpos[3:7], data.qpos[qa] = p[t], quat[t], q[t]
        mujoco.mj_forward(model, data)
        pt = ps = pr = 0.
        for c in data.contact:
            a, b = (int(model.geom_bodyid[g]) for g in c.geom)
            pen = max(0., -float(c.dist))
            if not a or not b:
                pt = max(pt, pen)
            else:
                names = frozenset((model.body(a).name, model.body(b).name))
                pr = max(pr, pen)
                key = " / ".join(sorted(names))
                pairs[key] = max(pairs.get(key, 0.), pen)
                if names not in BASELINE_PAIRS:
                    ps = max(ps, pen)
        terrain_pen.append(pt); self_pen.append(ps); raw_self.append(pr)
    low = np.array([model.joint(n).range[0] for n in robot.joint_names])
    high = np.array([model.joint(n).range[1] for n in robot.joint_names])
    limit = max(0., float(np.maximum(low-q, q-high).max()))
    speed = np.abs(np.gradient(q, axis=0)*fps)
    speed_ratio = float((speed/robot.velocity).max())
    reasons = []
    if limit > POLICY["joint_limit_tolerance_rad"]: reasons.append("joint_limit")
    if speed_ratio > POLICY["joint_velocity_max_ratio"]: reasons.append("joint_velocity")
    for name, values in (("terrain_penetration", terrain_pen), ("self_collision", self_pen)):
        values = np.asarray(values)
        if values.max() > POLICY["penetration_max_m"] or longest(values > POLICY["penetration_persistent_m"]) > POLICY["penetration_max_consecutive_frames"]:
            reasons.append(name)
    support = {}
    for b, name in enumerate(("left_foot", "right_foot")):
        mask = source_contact[:, b]
        distances = np.abs(one["contact_dist"][mask, b])
        # Link-frame XY speed is also reported as a conservative slip diagnostic.
        adjacent = mask[1:] & mask[:-1]
        speed_xy = np.linalg.norm(np.diff(one["body_pos_w"][:, b, :2], axis=0)*fps, axis=-1)[adjacent]
        dp95 = float(np.percentile(distances, 95)) if distances.size else None
        sp95 = float(np.percentile(speed_xy, 95)) if speed_xy.size else None
        support[name] = dict(source_support_frames=int(mask.sum()), distance_p95_m=dp95, link_slip_p95_m_s=sp95)
        if dp95 is not None and dp95 > POLICY["support_distance_p95_m"]: reasons.append(name+":support_distance")
        if sp95 is not None and sp95 > POLICY["support_slip_p95_m_s"]: reasons.append(name+":support_slip")
    if not source_contact.any(): reasons.append("no_inferred_source_support")
    report = dict(frames=len(p), fps=fps, policy=POLICY, strict_quality_pass=not reasons,
                  rejection_reasons=reasons, joint_limit_max_rad=limit, joint_velocity_max_ratio=speed_ratio,
                  terrain_penetration_max_m=float(max(terrain_pen)), self_penetration_policy_max_m=float(max(self_pen)),
                  self_penetration_raw_max_m=float(max(raw_self)), self_pair_max_m=pairs, support=support,
                  contact_rates={n: float(one["contact_info"][:, i, 0].mean()) for i, n in enumerate(robot.contact_bodies)},
                  root_step_max_m=float(np.linalg.norm(np.diff(p, axis=0), axis=-1).max()),
                  pd_status="not_run", learned_tracking_status="not_run")
    if diagnostics:
        return labels, report, dict(terrain_penetration_m=np.asarray(terrain_pen),
                                   self_penetration_m=np.asarray(self_pen),
                                   joint_velocity_ratio=speed/robot.velocity,
                                   joint_limit_violation_rad=np.maximum(0., np.maximum(low-q, q-high)))
    return labels, report


def worker(job):
    started = time.monotonic()
    source, out, acp_root, gmr_root, human_height = job
    source, out = Path(source), Path(out)
    key = "jwp/" + source.stem
    configure(acp_root, gmr_root)
    from supermimic.acp.robots import get_robot
    robot = get_robot("tienkung3_29dof")
    try:
        frames, contact, provenance = load_human(source)
        motion, raw, stats = retarget(frames, robot, out/"ik_config.json", human_height, contact)
        if not np.isfinite(motion).all(): raise ValueError("nonfinite retarget output")
        path = out/"motions"/(key+"_motion.npz")
        atomic_npz(path, joint_pos=motion.astype(np.float32), joint_names=np.array(robot.joint_names),
                   fps=np.array([POLICY["fps"]]), robot_metadata=json.dumps(robot.metadata()),
                   source=str(source), source_sha256=sha(source), source_frames=provenance["source_frame_interval"],
                   data_version=VERSION, provenance=json.dumps(provenance))
        terrain = path.with_name(source.stem+"_terrain.npy")
        np.save(terrain, np.zeros((0, 1, 10), dtype=np.float32))
        atomic_npz(out/"raw"/(key+".npz"), qpos=raw, source_contact=contact,
                   source_root=np.array([f["Hips"][0] for f in frames]), fps=POLICY["fps"])
        labels, quality = quality_and_labels(path, robot, contact)
        label_path = out/"contacts"/(key+".npz")
        atomic_npz(label_path, source_motion=key+".npz", **labels)
        quality_path = out/"quality"/(key+".json")
        atomic_json(quality_path, quality)
        record = dict(key=key, source=str(source), source_sha256=sha(source), provenance=provenance, stats=stats,
                      frames=len(motion), state="converted", strict_quality_pass=quality["strict_quality_pass"],
                      rejection_reasons=quality["rejection_reasons"], seconds=time.monotonic()-started)
        for name, p in (("motion", path), ("terrain", terrain), ("labels", label_path), ("quality", quality_path)):
            record[name], record[name+"_sha256"] = str(p), sha(p)
    except Exception:
        record = dict(key=key, source=str(source), state="failed", error=traceback.format_exc(), seconds=time.monotonic()-started)
    atomic_json(out/"states"/(source.stem+".json"), record)
    return record


def verify_loaders(out, acp_root, gmr_root):
    configure(acp_root, gmr_root)
    import torch
    from supermimic.acp.robots import get_robot
    from supermimic.acp.kinematics import RobotKinematics
    from supermimic.acp.motion import MotionLib
    from supermimic.acp.mdp import MDP
    from supermimic.acp.task import ContactTask, ContactCfg
    robot = get_robot("tienkung3_29dof")
    kin = RobotKinematics("cpu", robot)
    motion = MotionLib.from_dir(str(out/"motions"), kin, "cpu")
    mdp = MDP(kin, motion, torch.tensor(robot.default_q))
    contact = ContactTask(ContactCfg(contacts_dir=str(out/"contacts"), contact_geometry="spheres224"), mdp, str(out/"motions"), "cpu")
    for start in range(0, motion.num_frames, 2048):
        f = torch.arange(start, min(start+2048, motion.num_frames))
        cmd = contact.command(motion.ref_state(f), f, torch.zeros_like(f), torch.full_like(f, 50), noise=False)
        if not bool(torch.isfinite(cmd).all()): raise ValueError("nonfinite ContactTask command")
    return dict(passed=True, clips=len(motion.names), frames=motion.num_frames, names=motion.names,
                sampleable_pairs=int(motion.variant_ok.sum()), data_hash=motion.data_hash,
                contact_command_dim=contact.cmd_dim, contact_geometry="spheres224", device="cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT.parent/"ACP_supermimic_Tienkung/data/JWP/bvh")
    ap.add_argument("--acp-root", type=Path, default=ROOT.parent/"standing-aiming")
    ap.add_argument("--gmr-root", type=Path, default=ROOT/".cache/GMR")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--workers-per-gpu", type=int, default=0)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--human-height", type=float, default=1.75)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    if args.workers_per_gpu != 0 or args.cpu_workers < 1:
        ap.error("this GMR pipeline uses --workers-per-gpu 0 and positive --cpu-workers")
    for name in ("source", "acp_root", "gmr_root", "out"):
        setattr(args, name, getattr(args, name).resolve())
    files = sorted(args.source.glob("*.bvh"))
    if args.only: files = [p for p in files if p.stem in args.only]
    if not files or (args.only and len(files) != len(set(args.only))):
        raise ValueError("missing or duplicate requested BVH clips")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/"controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure(args.acp_root, args.gmr_root)
        from supermimic.acp.robots import get_robot
        robot = get_robot("tienkung3_29dof")
        candidate = args.out/"ik_config.candidate.json"
        make_ik_config(args.gmr_root, candidate, robot)
        code_files = sorted((ROOT/"tools").glob("*.py"))+[ROOT/"scripts/tiangong3/retarget_lafan1_gmr.py"]
        identity = dict(version=VERSION, source={str(p): sha(p) for p in files},
                        code={str(p.relative_to(ROOT)): sha(p) for p in code_files},
                        acp_root=str(args.acp_root), robot_metadata=robot.metadata(),
                        ik_config_sha256=sha(candidate),
                        acp_code={str(p.relative_to(args.acp_root)): sha(p) for p in sorted((args.acp_root/"supermimic/acp").glob("*.py"))},
                        gmr_code={str(p.relative_to(args.gmr_root)): sha(p) for p in sorted((args.gmr_root/"general_motion_retargeting").rglob("*.py"))},
                        packages={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "mujoco", "mink", "qpsolvers", "daqp", "torch")},
                        gmr_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.gmr_root, text=True).strip(),
                        human_height=args.human_height, policy=POLICY)
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        plan_path = args.out/"plan.json"
        if plan_path.exists():
            old = json.loads(plan_path.read_text())
            if old["identity_hash"] != identity_hash or not args.resume:
                raise ValueError("output identity changed or --resume missing; use a new output directory")
            if sha(args.out/"ik_config.json") != sha(candidate):
                raise ValueError("saved IK configuration was changed")
            candidate.unlink()
        else:
            atomic_json(plan_path, dict(identity=identity, identity_hash=identity_hash, command=sys.argv,
                                       workers=dict(cpu=args.cpu_workers, gpu=0), source_count=len(files)))
            candidate.replace(args.out/"ik_config.json")
        if args.scan:
            print(json.dumps(dict(clips=len(files), frames=sum(read_header(p)["frames"] for p in files),
                                  reserve_gib=2+6*args.cpu_workers, plan=str(plan_path))), flush=True)
            return
        records, pending = [], []
        for path in files:
            state = args.out/"states"/(path.stem+".json")
            if state.exists():
                row = json.loads(state.read_text())
                if row["state"] == "converted":
                    for field in ("motion", "terrain", "labels", "quality"):
                        if sha(row[field]) != row[field+"_sha256"]: raise ValueError("completed output was changed")
                    records.append(row)
                    continue
            pending.append(path)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = "1"
        print(f"{len(files)} sources; {len(records)} reused; {len(pending)} pending; {args.cpu_workers} CPU workers", flush=True)
        with ProcessPoolExecutor(args.cpu_workers, mp_context=mp.get_context("spawn")) as pool:
            jobs = {pool.submit(worker, (str(p), str(args.out), str(args.acp_root), str(args.gmr_root), args.human_height)): p for p in pending}
            for future in as_completed(jobs):
                row = future.result()
                records.append(row)
                print(json.dumps({k: row.get(k) for k in ("key", "state", "frames", "strict_quality_pass", "rejection_reasons", "seconds", "error")}), flush=True)
        records.sort(key=lambda row: row["key"])
        failed = [r for r in records if r["state"] != "converted"]
        converted = [r for r in records if r["state"] == "converted"]
        validation = verify_loaders(args.out, args.acp_root, args.gmr_root) if converted else dict(passed=False)
        if converted and (validation["clips"] != len(converted) or validation["frames"] != sum(r["frames"] for r in converted)):
            raise ValueError("training loader changed frame or clip coverage")
        atomic_json(args.out/"loader_validation.json", validation)
        manifest = dict(version=VERSION, identity_hash=identity_hash, robot_metadata=robot.metadata(),
                        source_count=len(files), converted_count=len(converted), failed_count=len(failed),
                        total_frames=sum(r["frames"] for r in converted),
                        strict_quality_pass_count=sum(r["strict_quality_pass"] for r in converted),
                        all_terminal=True, status="completed_with_failures" if failed else "completed",
                        training_loader_validation=validation, policy=POLICY, entries=records)
        atomic_json(args.out/"manifest.json", manifest)
        print(json.dumps({k: manifest[k] for k in ("status", "source_count", "converted_count", "failed_count", "total_frames", "strict_quality_pass_count")}), flush=True)


if __name__ == "__main__":
    main()
