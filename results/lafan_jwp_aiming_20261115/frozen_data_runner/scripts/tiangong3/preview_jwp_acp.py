"""Render converted ACP references with their native TienKung model (no rollout)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")


def render(dataset, acp_root, clips, output, video=True, collection="jwp"):
    sys.path.insert(0, str(acp_root))
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from supermimic.acp.robots import get_robot
    from supermimic.acp.motion import load_php_npz
    robot = get_robot("tienkung3_29dof")
    spec = robot.get_spec()
    spec.worldbody.add_geom(name="preview_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[0, 0, 1], rgba=[.55, .60, .65, 1], group=2)
    spec.worldbody.add_light(name="preview_light", pos=[0,0,4], dir=[0,0,-1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[.3,.3,.3], ambient=[.05,.05,.05],specular=[0,0,0])
    for i in range(-20,21):
        for axis in range(2):
            pos=[0.,0.,.0002];pos[1-axis]=i*.5
            size=[.002,.002,.0001];size[axis]=10.
            spec.worldbody.add_geom(name=f"grid_{axis}_{i}",type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,size=size,rgba=[.28,.32,.36,1],group=2,contype=0,conaffinity=0)
    model = spec.compile()
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    qa, _ = robot.joint_addresses(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.1, 130, -15
    output.mkdir(parents=True, exist_ok=True)
    records = []
    renderer = mujoco.Renderer(model, 720, 960)
    options=mujoco.MjvOption()
    options.geomgroup[0]=0  # Hide collision hulls overlapping the visual meshes.
    try:
        for clip in clips:
            path = dataset/"motions"/collection/(clip+"_motion.npz")
            p, quat, q, fps = load_php_npz(path, robot)
            labels=np.load(dataset/"contacts"/collection/(clip+".npz"))["contact_info"][0]
            writer = imageio.get_writer(output/(clip+".mp4"), fps=25, codec="libx264", quality=7) if video else None
            indices = sorted(set(range(0, len(p), 2)) | {len(p)//2, len(p)-1}) if video else [0, len(p)//2, len(p)-1]
            for t in indices:
                data.qpos[:3], data.qpos[3:7], data.qpos[qa] = p[t], quat[t], q[t]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = p[t]
                camera.lookat[2] = .85
                renderer.update_scene(data, camera,scene_option=options)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 960, 48), fill=(15, 20, 26))
                draw.text((16, 10), f"{clip}  |  ACP TienKung 29 DoF  |  {t/fps:.2f} s  |  kinematic reference", fill="white")
                draw.text((16, 28), f"Native foot contacts: left={int(labels[t,0,0])} right={int(labels[t,1,0])}", fill="white")
                if t in (0, len(p)//2, len(p)-1):
                    im.save(output/f"{clip}_{t:04d}.png")
                if writer is not None and t%2==0:
                    writer.append_data(np.asarray(im))
            if writer is not None: writer.close()
            records.append(dict(clip=clip, frames=len(p), fps=fps, motion=str(path)))
    finally:
        renderer.close()
    (output/"manifest.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--acp-root", type=Path, default=Path(__file__).resolve().parents[3]/"standing-aiming")
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--collection", default="jwp")
    args = parser.parse_args()
    render(args.dataset, args.acp_root, args.clips, args.output, not args.images_only, args.collection)
