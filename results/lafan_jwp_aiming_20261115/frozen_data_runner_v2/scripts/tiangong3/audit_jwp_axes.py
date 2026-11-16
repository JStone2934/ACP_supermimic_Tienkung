"""Independently audit JWP channel matrices, world axes and native joint indexing."""
from pathlib import Path
import argparse
import json
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.jwp_pipeline import configure, calibration_offsets, read_header


def channel_matrices(angles):
    """Explicit Ry @ Rx @ Rz, independent of GMR's quaternion implementation."""
    y, x, z = np.deg2rad(angles).T
    out = []
    for a,b,c in zip(y,x,z):
        ry = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        rx = np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
        rz = np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
        out.append(ry@rx@rz)
    return np.asarray(out)


def audit(source, acp_root, output):
    configure(acp_root, ROOT/".cache/GMR")
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
    from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk
    from supermimic.acp.robots import get_robot
    rows = []
    for path in sorted(source.glob("*.bvh")):
        header = read_header(path)
        assert all(x == ["Xposition","Yposition","Zposition","Yrotation","Xrotation","Zrotation"] for x in header["channels"])
        anim = read_bvh(str(path))
        text = path.read_text().splitlines()
        start = next(i for i,s in enumerate(text) if s.startswith("Frame Time:"))+1
        indices = [0,1,header["frames"]//2,header["frames"]-1]
        worst_p = worst_r = 0.
        for t in indices:
            raw = np.fromstring(text[start+t], sep=" ").reshape(-1,6)
            local = channel_matrices(raw[:,3:])
            rotations, positions = [], []
            for j, parent in enumerate(anim.parents):
                rotations.append(local[j] if parent < 0 else rotations[parent]@local[j])
                positions.append(raw[j,:3] if parent < 0 else positions[parent]+rotations[parent]@raw[j,:3])
            gq, gp = quat_fk(anim.quats[t], anim.pos[t], anim.parents)
            worst_p = max(worst_p,float(np.abs(gp-positions).max()/100))
            worst_r = max(worst_r,float(np.abs(Rotation.from_quat(gq,scalar_first=True).as_matrix()-rotations).max()))
        assert worst_p < 1e-10 and worst_r < 1e-10
        rows.append(dict(clip=path.stem,checked_source_frames=indices,position_max_error_m=worst_p,
                         rotation_matrix_max_error=worst_r,bones=len(anim.bones),channels_per_bone=6))
    robot = get_robot("tienkung3_29dof")
    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    zero, delta = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_kinematics(model,zero)
    qa, va = robot.joint_addresses(model)
    joints = []
    for name,q,v in zip(robot.joint_names,qa,va):
        j = model.joint(name); body = int(j.bodyid[0])
        delta.qpos[:] = model.qpos0
        delta.qpos[q] += 1e-5
        mujoco.mj_kinematics(model,delta)
        observed = (Rotation.from_quat(delta.xquat[body],scalar_first=True)*Rotation.from_quat(zero.xquat[body],scalar_first=True).inv()).as_rotvec()/1e-5
        expected = zero.xmat[body].reshape(3,3)@j.axis
        err = float(np.max(np.abs(observed-expected)))
        assert err < 1e-8
        joints.append(dict(name=name,qpos_index=int(q),velocity_index=int(v),axis=j.axis.tolist(),range_rad=j.range.tolist(),positive_rotation_axis_max_error=err))
    offsets = calibration_offsets(robot)
    basis = Rotation.from_euler("x",90,degrees=True)
    calibrated = {n:(basis*Rotation.from_quat(q,scalar_first=True)).as_matrix().tolist() for n,q in offsets.items()}
    report = dict(passed=True,source_clips=len(rows),source_frame_samples=4*len(rows),
                  translation_channels="XYZ, centimetres, local to parent (root in world)",
                  rotation_channels="intrinsic YXZ degrees; matrix Ry @ Rx @ Rz",
                  world_basis=[[1,0,0],[0,0,-1],[0,1,0]],world_coordinates="(x,-z,y)/100 metres Z-up",
                  quaternion_order="wxyz in GMR, MuJoCo and ACP NPZ; explicit conversion for SciPy",
                  calibration="BVH rest left +X/up +Y/forward +Z; robot T pose shoulder roll +/-85 degrees, including MJCF +/-5 degree offsets; C_body = B^-1 Rz(-90) R_body(T)",
                  previous_pilots="pilot_v1 through pilot_v3 INVALID: LAFAN bone-axis offsets do not match JWP",
                  robot_metadata=robot.metadata(),joint_order_and_axes=joints,
                  calibrated_rest_body_matrices=calibrated,clips=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(passed=True,clips=len(rows),frames=4*len(rows),joints=len(joints),
                         position_max_error_m=max(x["position_max_error_m"] for x in rows),
                         rotation_matrix_max_error=max(x["rotation_matrix_max_error"] for x in rows))))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=ROOT.parent/"ACP_supermimic_Tienkung/data/JWP/bvh")
    ap.add_argument("--acp-root",type=Path,default=ROOT.parent/"standing-aiming")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    audit(args.source,args.acp_root,args.output)
