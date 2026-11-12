"""Render recorded evaluations (supermimic.acp.evaluate --record) with MuJoCo (EGL): the simulated robot solid, the
reference a translucent ghost, the clip's boxes, and the commanded contact targets as colored spheres (feet red / blue,
knees orange, hands green, elbows purple, pelvis yellow). Each clip plays until its rollout ends, plus ``--tail`` s.

    python -m supermimic.acp.render traj.npz out.mp4 --clips motions-low-climb-76/forward_walk0.6s_step0.7s
    python -m supermimic.acp.render traj.npz out.mp4 --per_folder 1      # the first clip of every motion folder
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .robots import get_robot, resolve_robot

TARGET_RGBA = [(0.9, 0.2, 0.2, 0.9), (0.2, 0.4, 0.95, 0.9), (1.0, 0.55, 0.1, 0.9), (1.0, 0.55, 0.1, 0.9), (0.2, 0.8, 0.3, 0.9),
               (0.2, 0.8, 0.3, 0.9), (0.6, 0.3, 0.8, 0.9), (0.6, 0.3, 0.8, 0.9), (0.95, 0.85, 0.2, 0.9)]  # fmt: skip


def scene_xml(boxes: np.ndarray, robot="g1") -> str:
    """Robot and ghost assembled by XML structure; assets and joints keep explicit names."""
    robot = get_robot(robot)
    root = ET.parse(robot.mjcf_path).getroot()
    compiler = root.find("compiler")
    meshdir = Path(robot.mjcf_path).parent / (compiler.get("meshdir", "") if compiler is not None else "")
    asset = root.find("asset")
    if asset is None: asset = ET.SubElement(root, "asset")
    for mesh in asset.findall("mesh"):
        mesh.set("file", str((meshdir / mesh.get("file")).resolve()))
    wb = root.find("worldbody")
    body = wb.find(f"body[@name='{robot.root_body}']")
    if body is None: raise ValueError("root body missing")
    ghost = copy.deepcopy(body)
    for elem in ghost.iter():
        if "name" in elem.attrib: elem.set("name", "ref_" + elem.get("name"))
        if elem.tag == "geom":
            elem.attrib.pop("material", None)
            elem.set("rgba", "0.35 0.65 1.0 0.25")
            elem.set("contype", "0"); elem.set("conaffinity", "0")
    wb.append(ghost)
    ET.SubElement(wb,"light",pos="0 0 6",dir="0 0 -1",directional="true",diffuse="0.8 0.8 0.8")
    ET.SubElement(wb,"geom",name="floor",type="plane",size="0 0 0.05",material="grid",contype="0",conaffinity="0")
    for k,(cx,cy,top,c,s,hx,hy,hz) in enumerate(boxes):
        if hx>0:
            yaw=np.arctan2(s,c)
            ET.SubElement(wb,"geom",name=f"box{k}",type="box",size=f"{hx} {hy} {hz}",pos=f"{cx} {cy} {top-hz}",
                          quat=f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}",rgba="0.62 0.55 0.45 1",contype="0",conaffinity="0")
    ET.SubElement(asset,"texture",name="sky",type="skybox",builtin="gradient",rgb1="0.85 0.9 0.97",rgb2="0.55 0.65 0.8",width="256",height="256")
    ET.SubElement(asset,"texture",name="grid",type="2d",builtin="checker",rgb1="0.92 0.93 0.94",rgb2="0.80 0.82 0.85",width="512",height="512")
    ET.SubElement(asset,"material",name="grid",texture="grid",texrepeat="8 8",reflectance="0.1")
    return ET.tostring(root,encoding="unicode")


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["g1", "tienkung3_29dof"], default=None)
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None, help="clip names (default: --per_folder)")
    ap.add_argument("--per_folder", type=int, default=1, help="without --clips: this many clips of every motion folder")
    ap.add_argument("--seconds", type=float, default=None, help="cap per clip")
    ap.add_argument("--tail", type=float, default=1.0, help="s shown after the rollout ends")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--title", default=None, help="short video title")
    ap.add_argument("--solid-label", default="policy", help="legend for the solid robot, e.g. reference PD")
    ap.add_argument("--hide-reference", action="store_true", help="hide the blue reference robot")
    args = ap.parse_args()

    # Load once: accessing an NpzFile repeatedly decompresses whole arrays per frame.
    with np.load(args.traj) as archive:
        d = {key: archive[key] for key in archive.files}
    robot = resolve_robot(args.robot, d)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    names = list(d["names"])
    if args.clips:
        clips = [names.index(c) for c in args.clips]
    else:
        folders = sorted({n.split("/")[0] for n in names})
        clips = [i for g in folders for i in [j for j, n in enumerate(names) if n.startswith(g + "/")][: args.per_folder]]
    alive, f = d["alive"], d["f"]  # (T, C)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
                               "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", args.out],
                              stdin=subprocess.PIPE)  # fmt: skip
    for c in clips:
        m = mujoco.MjModel.from_xml_string(scene_xml(d["boxes"][c], robot))
        m.vis.global_.offwidth, m.vis.global_.offheight = args.width, args.height
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = 0.45, 0.6
        if args.hide_reference:
            for geom_id in range(m.ngeom):
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[geom_id])) or ""
                if body_name.startswith("ref_"):
                    m.geom_rgba[geom_id, 3] = 0
        data = mujoco.MjData(m)
        nq = 7 + d["q"].shape[-1]
        qadr, _ = robot.joint_addresses(m)
        ref_qadr = np.array([m.jnt_qposadr[m.joint("ref_" + n).id] for n in robot.joint_names])
        r = mujoco.Renderer(m, args.height, args.width)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 4.0, -20.0, 135.0
        T_alive = int(alive[:, c].sum())
        T = min(T_alive + int(args.tail * args.fps), len(alive))
        if args.seconds:
            T = min(T, int(args.seconds * args.fps))
        failed = T_alive < int(d["clip_len"][c]) - 1
        for t in range(T):
            fr = int(f[t, c])
            data.qpos[:3], data.qpos[3:7], data.qpos[qadr] = d["root_p"][t, c], d["root_quat"][t, c], d["q"][t, c]
            data.qpos[nq : nq + 3], data.qpos[nq + 3 : nq + 7] = d["ref_root_p"][fr], mat_to_quat(d["ref_root_R"][fr])
            data.qpos[ref_qadr] = d["ref_q"][fr]
            mujoco.mj_kinematics(m, data)
            cam.lookat[:] = d["root_p"][t, c]
            r.update_scene(data, cam)
            scn = r.scene
            for b in np.flatnonzero(d["tgt_valid"][t, c]):
                if scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.035, 0, 0]), d["tgt_w"][t, c, b].astype(np.float64),
                                        np.eye(3).reshape(-1), np.array(TARGET_RGBA[b], dtype=np.float32))  # fmt: skip
                    scn.ngeom += 1
            img = Image.fromarray(r.render())
            draw = ImageDraw.Draw(img)
            draw.text((10, 8), f"{args.title or names[c]}  t={t / args.fps:.2f}s" + ("  FAILED" if failed and t >= T_alive else ""), font=font, fill=(0, 0, 0))
            draw.text((10, args.height - 30), f"Solid: {args.solid_label}" + ("" if args.hide_reference else " | Blue ghost: reference") + " | Dots: planned contacts", font=font, fill=(0, 0, 0))
            ffmpeg.stdin.write(img.tobytes())
        r.close()
        print(f"{names[c]}: {T} frames, {'failed at %.2f s' % (T_alive / args.fps) if failed else 'completed'}", flush=True)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the video")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
