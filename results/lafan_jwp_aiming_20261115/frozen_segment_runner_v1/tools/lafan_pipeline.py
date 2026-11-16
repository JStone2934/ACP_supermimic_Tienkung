"""31-joint upstream Tiangong NPZ -> native ACP 29-joint references.

Always launch via python -m tools.convert_php --lafan-source ... .  GPU workers
recompute all-body FK/velocities; independent native CPU workers label contacts
and check collisions. Source positions are metres, Z-up, quaternions wxyz.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.ndimage import binary_opening
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from tools.jwp_pipeline import (ROOT, POLICY, atomic_json, atomic_npz, configure,
                                sha, quality_and_labels, verify_loaders)

VERSION = "lafan-native29-v1"
KIN = ROBOT = DEVICE = SOURCE_MODEL = None
HEAD_JOINTS = {"head_yaw_joint", "head_pitch_joint"}
HEAD_BODIES = {"head_yaw_link", "head_pitch_link"}


def map_source(z, robot):
    names = list(map(str, z["joint_names"]))
    bodies = list(map(str, z["body_names"]))
    if len(set(names)) != len(names) or set(names)-set(robot.joint_names) != HEAD_JOINTS:
        raise ValueError("expected exactly the canonical 29 joints plus yaw/pitch head joints")
    if not set(robot.joint_names).issubset(names) or len(names) != 31:
        raise ValueError("source joint names do not match target robot")
    if len(set(bodies)) != len(bodies) or "pelvis" not in bodies:
        raise ValueError("ambiguous/missing root body")
    fps = float(np.asarray(z["fps"]).item())
    if abs(fps-POLICY["fps"]) > 1e-6:
        raise ValueError("source must already be 50 Hz; no implicit retiming")
    q = np.asarray(z["joint_pos"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 31 or len(q) < 3:
        raise ValueError("expected T x 31 source angles")
    p = np.asarray(z["body_pos_w"][:, bodies.index("pelvis")], dtype=np.float64)
    quat = np.asarray(z["body_quat_w"][:, bodies.index("pelvis")], dtype=np.float64).copy()
    if not all(np.isfinite(x).all() for x in (p, quat, q)):
        raise ValueError("nonfinite source")
    if np.max(np.abs(np.linalg.norm(quat, axis=-1)-1)) > 1e-4:
        raise ValueError("source root quaternion not unit wxyz")
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    signs = np.where(np.sum(quat[1:]*quat[:-1], axis=-1) < 0, -1., 1.)
    quat[1:] *= np.cumprod(signs)[:, None]
    idx = [names.index(n) for n in robot.joint_names]
    return np.concatenate((p, quat, q[:, idx]), axis=-1), idx


def initialize_gpu(devices, acp_root):
    global KIN, ROBOT, DEVICE, SOURCE_MODEL
    configure(acp_root, ROOT/".cache/GMR")
    sys.path.insert(0, str(ROOT/"src"))
    import torch
    from supermimic.acp.robots import get_robot
    from supermimic.acp.kinematics import RobotKinematics
    from tiangong_supermimic.robot import get_spec
    DEVICE = devices.get()
    torch.cuda.set_device(DEVICE)
    ROBOT = get_robot("tienkung3_29dof")
    KIN = RobotKinematics(DEVICE, ROBOT)
    SOURCE_MODEL = get_spec().compile()
    # Compare actual joint axes, anchors, references and body transforms by name.
    for name in ROBOT.joint_names:
        sj, dj = SOURCE_MODEL.joint(name).id, KIN.model.joint(name).id
        for field in ("jnt_axis", "jnt_pos", "jnt_range"):
            np.testing.assert_allclose(getattr(SOURCE_MODEL, field)[sj], getattr(KIN.model, field)[dj], atol=1e-10, rtol=0)
        sb, db = int(SOURCE_MODEL.jnt_bodyid[sj]), int(KIN.model.jnt_bodyid[dj])
        for field in ("body_pos", "body_quat"):
            np.testing.assert_allclose(getattr(SOURCE_MODEL, field)[sb], getattr(KIN.model, field)[db], atol=1e-10, rtol=0)
        np.testing.assert_allclose(SOURCE_MODEL.qpos0[SOURCE_MODEL.jnt_qposadr[sj]], KIN.model.qpos0[KIN.model.jnt_qposadr[dj]], atol=1e-10)
    print(f"LAFAN FK worker ready: {DEVICE}, {torch.cuda.get_device_name(DEVICE)}, pid={os.getpid()}", flush=True)


def gpu_fk(motion, *, kin=None, device=None):
    import torch
    from supermimic.rotations import quat_to_mat, mat_to_quat, so3_log
    kin = KIN if kin is None else kin
    device = DEVICE if device is None else device
    with torch.no_grad():
        x = torch.as_tensor(motion, dtype=torch.float32, device=device)
        p, R, q = x[:, :3], quat_to_mat(x[:, 3:7]), x[:, 7:]
        v = torch.gradient(p, spacing=.02, dim=0)[0]
        qd = torch.gradient(q, spacing=.02, dim=0)[0]
        w = torch.zeros_like(p)
        w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2))/.04
        w[0] = so3_log(R[1] @ R[0].T)/.02
        w[-1] = so3_log(R[-1] @ R[-2].T)/.02
        chunks = []
        for a in range(0, len(x), 2048):
            sl = slice(a, a+2048)
            P, B, V, W = kin.forward(p[sl], R[sl], v[sl], w[sl], q[sl], qd[sl], all_bodies=True)
            chunks.append(tuple(y.cpu().numpy() for y in (P, mat_to_quat(B), V, W)))
        result = {k: np.concatenate([chunk[i] for chunk in chunks]) for i, k in enumerate(
            ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"))}
        result.update(joint_vel=qd.cpu().numpy(), root_lin_vel_w=v.cpu().numpy(), root_ang_vel_w=w.cpu().numpy(),
                      body_names=np.array(kin.body_names))
    if not all(np.isfinite(v).all() for k, v in result.items() if k != "body_names"):
        raise ValueError("nonfinite recomputed FK/velocity")
    return result


def foot_heights_native(poses):
    """Exact convex support function of the current robot's collision meshes."""
    m = KIN.model
    heights = []
    for side in ("left", "right"):
        _, links = ROBOT.contact_bodies[side+"_foot"]
        group = []
        for g in range(m.ngeom):
            name = m.body(int(m.geom_bodyid[g])).name
            if not m.geom_conaffinity[g] or name not in links:
                continue
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            points = m.mesh_vert[a:a+n].astype(np.float64)
            points = points[ConvexHull(points).vertices]
            points = points @ Rotation.from_quat(m.geom_quat[g], scalar_first=True).as_matrix().T + m.geom_pos[g]
            b = KIN.body_names.index(name)
            axes = Rotation.from_quat(poses["body_quat_w"][:, b], scalar_first=True).as_matrix()[:, 2]
            group.append(poses["body_pos_w"][:, b, 2] + np.min(axes @ points.T, axis=1))
        if not group:
            raise ValueError("native foot colliders not found")
        heights.append(np.min(group, axis=0))
    return np.stack(heights, axis=1)


def prepare(source, out):
    import mujoco
    import torch
    from tiangong_supermimic.data.motion_convert import foot_heights
    started = time.monotonic()
    source, out = Path(source), Path(out)
    key = "lafan/"+source.stem
    try:
        with np.load(source, allow_pickle=False) as z:
            motion, mapping = map_source(z, ROBOT)
            original = motion.copy()
            poses = gpu_fk(motion)
            source_names = list(z["body_names"])
            shared = [n for n in KIN.body_names if n in source_names and n not in HEAD_BODIES]
            if set(shared) != set(source_names)-HEAD_BODIES:
                raise ValueError("unmapped non-head source bodies")
            target_indices = [KIN.body_names.index(n) for n in shared]
            source_indices = [source_names.index(n) for n in shared]
            source_error = float(np.linalg.norm(poses["body_pos_w"][:, target_indices]-z["body_pos_w"][:, source_indices], axis=-1).max())
            sr = Rotation.from_quat(z["body_quat_w"][:, source_indices].reshape(-1,4), scalar_first=True)
            tr = Rotation.from_quat(poses["body_quat_w"][:, target_indices].reshape(-1,4), scalar_first=True)
            rotation_error = float((sr.inv()*tr).magnitude().max())
            if source_error > 1e-5 or rotation_error > 1e-5:
                raise ValueError(f"31->29 kinematic parity failure: {source_error=}, {rotation_error=}")
            native_order = [source_names.index(SOURCE_MODEL.body(i).name) for i in range(1, SOURCE_MODEL.nbody)]
            source_positions = z["body_pos_w"][:, native_order]
            source_quaternions = z["body_quat_w"][:, native_order]
            heights, source_foot_ids = foot_heights(SOURCE_MODEL, source_positions, source_quaternions)
            # Frozen stance evidence uses original source rail geometry and motion.
            # It is never recomputed from a repair to hide sliding/hovering.
            foot_pos = source_positions[:, source_foot_ids]
            speeds = np.linalg.norm(np.gradient(foot_pos, .02, axis=0), axis=-1)
            contact = (heights.T < POLICY["human_support_height_m"]) & (speeds < POLICY["human_support_speed_m_s"])
            for b in range(2):
                contact[:, b] = binary_opening(contact[:, b], structure=np.ones(3))
            native_heights = foot_heights_native(poses)
            anchor = native_heights[:50][contact[:50]]
            level = float(np.median(anchor)) if len(anchor) else float(np.percentile(native_heights.min(axis=1), 1))
            shift = POLICY["ground_clearance_m"]-level
            motion[:, 2] += shift
            # A constant shift preserves all velocities and airborne intervals.
            # No per-frame floor snapping or copying of source reset_lift.
            poses = gpu_fk(motion)
            data = mujoco.MjData(KIN.model)
            qa, _ = ROBOT.joint_addresses(KIN.model)
            native_error = 0.
            for t in np.linspace(0, len(motion)-1, 33, dtype=int):
                data.qpos[:7], data.qpos[qa] = motion[t, :7], motion[t, 7:]
                mujoco.mj_kinematics(KIN.model, data)
                native_error = max(native_error, float(np.abs(data.xpos[KIN.body_ids]-poses["body_pos_w"][t]).max()))
            if native_error > 1e-5:
                raise ValueError("CUDA/native MuJoCo FK mismatch")
            provenance = dict(source_frame_interval=[0, len(motion)], source_joint_names=list(map(str,z["joint_names"])),
                              target_to_source_indices=mapping, dropped_joint_names=sorted(HEAD_JOINTS),
                              source_body_names=list(map(str,source_names)), source_csv=str(z["source"].item()),
                              positions="xyz metres, Z-up, unchanged XY", root_quaternion="wxyz",
                              ground="constant vertical offset from native foot meshes at initial stance", ground_shift_m=shift,
                              source_reset_lift_applied=False, source_reset_lift_max_m=float(z["reset_lift"].max()),
                              stance_evidence="original source rail height <4cm and original foot link speed <0.25m/s; 3-frame opening",
                              source_head_angles="removed; target head is fixed by current MJCF")
        path = out/"motions"/(key+"_motion.npz")
        atomic_npz(path, joint_pos=motion.astype(np.float32), joint_names=np.array(ROBOT.joint_names),
                   fps=np.array([50.]), robot_metadata=json.dumps(ROBOT.metadata()), source=str(source),
                   source_sha256=sha(source), data_version=VERSION, provenance=json.dumps(provenance), **poses)
        terrain = path.with_name(source.stem+"_terrain.npy")
        np.save(terrain, np.zeros((0,1,10), dtype=np.float32))
        atomic_npz(out/"raw"/(key+".npz"), qpos=original, source_contact=contact,
                   source_root=original[:,:3], fps=50., source_foot_height=heights.T, source_foot_speed=speeds)
        row = dict(key=key, source=str(source), source_sha256=sha(source), motion=str(path), terrain=str(terrain),
                   frames=len(motion), state="prepared", provenance=provenance,
                   axis_audit=dict(passed=True, source_target_position_max_m=source_error,
                                   source_target_rotation_max_rad=rotation_error, cuda_native_position_max_m=native_error,
                                   joint_axes_anchors_limits_and_body_frames_match=True,
                                   shared_body_names=shared, added_fixed_frames=sorted(set(KIN.body_names)-set(source_names))),
                   stats=dict(device=DEVICE, gpu_name=torch.cuda.get_device_name(DEVICE), seconds=time.monotonic()-started))
    except Exception:
        row = dict(key=key, source=str(source), state="failed", error=traceback.format_exc())
    atomic_json(out/"prepare_states"/(source.stem+".json"), row)
    return row


def check(row, out, acp_root):
    configure(acp_root, ROOT/".cache/GMR")
    from supermimic.acp.robots import get_robot
    out = Path(out); key = row["key"]
    try:
        with np.load(out/"raw"/(key+".npz")) as raw:
            contact = raw["source_contact"]
        labels, quality, diagnostics = quality_and_labels(Path(row["motion"]), get_robot("tienkung3_29dof"), contact, diagnostics=True)
        # Inactive contact positions/normals are intentionally NaN. Active ones
        # must have finite geometry and a signed distance within the threshold.
        active = labels["contact_info"][...,0].astype(bool)
        for field in ("contact_pos_w", "contact_normal_w"):
            if field in labels and not np.isfinite(labels[field][active]).all():
                raise ValueError(f"nonfinite active {field}")
        quality.update(axis_audit=row["axis_audit"], source_support_inference=row["provenance"]["stance_evidence"])
        label_path, quality_path = out/"contacts"/(key+".npz"), out/"quality"/(key+".json")
        atomic_npz(label_path, source_motion=key+".npz", **labels)
        atomic_npz(out/"frame_checks"/(key+".npz"), **diagnostics)
        atomic_json(quality_path, quality)
        row.update(labels=str(label_path), quality=str(quality_path), state="converted",
                   strict_quality_pass=quality["strict_quality_pass"], rejection_reasons=quality["rejection_reasons"])
        for field in ("motion", "terrain", "labels", "quality"):
            row[field+"_sha256"] = sha(row[field])
    except Exception:
        row.update(state="failed", error=traceback.format_exc())
    atomic_json(out/"states"/(Path(key).name+".json"), row)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lafan-source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--acp-root", type=Path, default=ROOT.parent/"standing-aiming")
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--cpu-workers", type=int, default=2)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    args.out, args.lafan_source, args.acp_root = args.out.resolve(), args.lafan_source.resolve(), args.acp_root.resolve()
    devices = ["cuda:"+d for d in args.devices.split(",") for _ in range(args.workers_per_gpu)]
    if not devices or args.cpu_workers < 1:
        ap.error("positive GPU and CPU worker counts required")
    files = sorted(args.lafan_source.glob("*.npz"))
    if args.only:
        files = [p for p in files if p.stem in args.only]
    if not files or (args.only and len(files) != len(set(args.only))):
        raise ValueError("missing or duplicate input clips")
    if args.scan:
        print(json.dumps(dict(clips=len(files), gpu_workers=len(devices), cpu_workers=args.cpu_workers,
                              reserve_gib=2+9*len(devices)+6*args.cpu_workers, out=str(args.out))))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/"controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure(args.acp_root, ROOT/".cache/GMR")
        from supermimic.acp.robots import get_robot
        robot = get_robot("tienkung3_29dof")
        identity = dict(version=VERSION, source={str(p):sha(p) for p in files}, policy=POLICY,
                        robot_metadata=robot.metadata(), acp_root=str(args.acp_root),
                        code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"tools").glob("*.py"))},
                        acp_code={str(p.relative_to(args.acp_root)):sha(p) for p in sorted((args.acp_root/"supermimic/acp").glob("*.py"))},
                        source_model=sha(ROOT/"assets/tiangong3/tiangong3.xml"),
                        packages={n:importlib.metadata.version(n) for n in ("torch","numpy","scipy","mujoco")})
        identity_hash = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        plan = args.out/"plan.json"
        if plan.exists():
            if not args.resume or json.loads(plan.read_text())["identity_hash"] != identity_hash:
                raise ValueError("existing output identity differs or --resume missing")
        else:
            atomic_json(plan, dict(identity=identity, identity_hash=identity_hash, command=sys.argv,
                                   workers=dict(gpu=len(devices),cpu=args.cpu_workers), source_count=len(files)))
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
            os.environ[k] = "1"
        records, pending = [], []
        for path in files:
            state = args.out/"states"/(path.stem+".json")
            if state.exists():
                row = json.loads(state.read_text())
                if row["state"] == "converted":
                    for f in ("motion","terrain","labels","quality"):
                        if sha(row[f]) != row[f+"_sha256"]:
                            raise ValueError("completed output changed")
                    records.append(row)
                    continue
            pending.append(path)
        print(f"{len(files)} sources, {len(records)} reused, {len(devices)} GPU + {args.cpu_workers} CPU workers", flush=True)
        context = mp.get_context("spawn")
        queue = context.Queue()
        for device in devices: queue.put(device)
        with ProcessPoolExecutor(len(devices), mp_context=context, initializer=initialize_gpu, initargs=(queue,str(args.acp_root))) as gpu, \
             ProcessPoolExecutor(args.cpu_workers, mp_context=context) as cpu:
            futures = {gpu.submit(prepare,str(p),str(args.out)):"prepare" for p in pending}
            while futures:
                done,_ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    stage = futures.pop(future)
                    row = future.result()
                    if stage == "prepare" and row["state"] == "prepared":
                        futures[cpu.submit(check,row,str(args.out),str(args.acp_root))] = "check"
                    else:
                        records.append(row)
                        print(json.dumps({k:row.get(k) for k in ("key","state","frames","strict_quality_pass","rejection_reasons","error")}),flush=True)
        records.sort(key=lambda r:r["key"])
        converted = [r for r in records if r["state"] == "converted"]
        validation = verify_loaders(args.out,args.acp_root,ROOT/".cache/GMR") if converted else dict(passed=False)
        if converted and (validation["clips"] != len(converted) or validation["frames"] != sum(r["frames"] for r in converted)):
            raise ValueError("loader coverage mismatch")
        atomic_json(args.out/"loader_validation.json", validation)
        atomic_json(args.out/"axis_audit.json",dict(passed=all(r["axis_audit"]["passed"] for r in converted),
                    clips=len(converted), entries={r["key"]:r["axis_audit"] for r in converted}))
        manifest = dict(version=VERSION, identity_hash=identity_hash, robot_metadata=robot.metadata(), source_count=len(files),
                        converted_count=len(converted), failed_count=len(records)-len(converted), total_frames=sum(r["frames"] for r in converted),
                        strict_quality_pass_count=sum(r["strict_quality_pass"] for r in converted), policy=POLICY,
                        all_terminal=True, status="completed" if len(converted)==len(files) else "completed_with_failures",
                        training_loader_validation=validation,entries=records)
        atomic_json(args.out/"manifest.json",manifest)
        print(json.dumps({k:v for k,v in manifest.items() if k not in ("entries","robot_metadata","policy","training_loader_validation")}),flush=True)


if __name__ == "__main__":
    main()
