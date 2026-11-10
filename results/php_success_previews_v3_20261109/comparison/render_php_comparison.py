"""Synchronized original G1 / converted TienKung kinematic comparisons.

Reads a verified preview manifest. Both panels use the same terrain, frame,
world scale and camera pose; no motion resampling or geometric normalization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.acp.motion import load_php_npz
from supermimic.acp.render import scene_xml
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.terrain import load_terrain


def render_clip(row, out, width, height):
    quality_path = Path(row['quality_path'])
    if sha256(quality_path) != row['quality_sha256']:
        raise ValueError('preview quality report changed')
    quality = json.loads(quality_path.read_text())
    record = quality['source']
    original = Path(record['source'])
    converted = Path(row['parent'])/'accepted'/(row['source_key']+'_motion.npz')
    if sha256(original) != record['source_hash'] or sha256(converted) != row['motion_sha256']:
        raise ValueError('preview/source motion changed')
    source_variant, export_slot = row['source_variant'], row['export_slot']
    if quality['accepted_variants'][export_slot] != source_variant:
        raise ValueError('preview variant mapping differs from quality report')
    boxes = load_terrain(str(original))[source_variant]
    if not np.array_equal(boxes, load_terrain(str(converted))[export_slot]):
        raise ValueError('source and target terrain differ')
    robots = [get_robot('g1'), get_robot('tienkung3_29dof')]
    motions = [load_php_npz(original, robots[0]), load_php_npz(converted, robots[1])]
    first, last = record['source_frames']
    motions[0] = tuple(a[first:last] for a in motions[0][:3]) + (motions[0][3],)
    frames, fps = len(motions[0][0]), motions[0][3]
    if frames != len(motions[1][0]) or fps != motions[1][3] or frames != row['frames']:
        raise ValueError('source and target frame timing differs')
    models, states, renderers, addresses = [], [], [], []
    option = mujoco.MjvOption()
    option.geomgroup[5] = 0
    for robot in robots:
        model = mujoco.MjModel.from_xml_string(scene_xml(boxes, robot))
        model.vis.global_.offwidth, model.vis.global_.offheight = width, height
        model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = .45, .6
        # The shared scene builder includes a reference ghost; exclude it entirely
        # from rendering, so each panel shows only its own robot.
        for g in range(model.ngeom):
            if model.body(model.geom_bodyid[g]).name.startswith('ref_'):
                model.geom_group[g] = 5
        models.append(model)
        states.append(mujoco.MjData(model))
        renderers.append(mujoco.Renderer(model, height, width))
        addresses.append(robot.joint_addresses(model)[0])
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 4.2, -20, 135
    font = ImageFont.truetype('DejaVuSans.ttf', 21)
    small = ImageFont.truetype('DejaVuSans.ttf', 17)
    video = out/(row['name']+'_comparison.mp4')
    ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{2*width}x{height}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    titles = ['Original PHP / Unitree G1', 'Converted / TienKung 3.0']
    category = row['source_key'].split('/')[0]
    skill = {'motions-low-step': 'Low-speed step', 'motions-high-step': 'High-speed step',
             'motions-locomotion': 'Locomotion'}[category]
    try:
        for t in range(frames):
            camera.lookat[:] = (motions[0][0][t] + motions[1][0][t])/2
            panels = []
            for i, (model, state, renderer, qa, motion) in enumerate(zip(models, states, renderers, addresses, motions)):
                p, quat, joints, _ = motion
                state.qpos[:3], state.qpos[3:7], state.qpos[qa] = p[t], quat[t], joints[t]
                mujoco.mj_kinematics(model, state)
                renderer.update_scene(state, camera, scene_option=option)
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, width, 68), fill=(20, 28, 40))
                draw.text((14, 8), titles[i], font=font, fill='white')
                draw.text((14, 37), f'{skill} | terrain variant {source_variant} | t={t/fps:.2f}s | frame {t+1}/{frames}', font=small, fill='white')
                draw.rectangle((0, height-30, width, height), fill=(20, 28, 40))
                draw.text((14, height-25), 'Kinematic replay | same camera, terrain and timing | 1x speed', font=small, fill='white')
                panels.append(np.asarray(panel))
            joined = np.concatenate(panels, axis=1)
            joined[:, width-1:width+1] = 30
            ffmpeg.stdin.write(joined.tobytes())
    finally:
        ffmpeg.stdin.close()
        for renderer in renderers:
            renderer.close()
        returncode = ffmpeg.wait()
    if returncode:
        raise RuntimeError('comparison encoder failed')
    print(f'{video}: {frames} frames', flush=True)
    return dict(name=row['name'], source_key=row['source_key'], source_motion=str(original.resolve()),
        converted_motion=str(converted.resolve()), source_motion_sha256=sha256(original),
        converted_motion_sha256=sha256(converted), quality_path=str(quality_path),
        quality_sha256=sha256(quality_path), source_variant=source_variant, export_slot=export_slot,
        source_frame_range=[first, last], frames=frames, fps=fps, duration_s=frames/fps,
        camera=dict(distance=4.2, elevation=-20, azimuth=135, lookat='mean of both root positions per frame'),
        identical_terrain=True, identical_timing=True, robot_metadata=[r.metadata() for r in robots],
        video=str(video.resolve()), video_sha256=sha256(video))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('preview_dir', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--width', type=int, default=960, help='width of each panel')
    ap.add_argument('--height', type=int, default=640)
    args = ap.parse_args()
    out = args.out or args.preview_dir/'comparison'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.preview_dir/'manifest.json').read_text())
    rows = [render_clip(row, out, args.width, args.height) for row in manifest['clips']]
    concat = out/'concat.txt'
    concat.write_text(''.join(f"file '{Path(row['video']).name}'\n" for row in rows))
    overview = out/'all_comparisons.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(concat), '-c', 'copy', '-movflags', '+faststart', str(overview)], check=True)
    (out/'manifest.json').write_text(json.dumps(dict(purpose='Synchronized kinematic comparison; no policy or physics simulation',
        source_preview_manifest=str((args.preview_dir/'manifest.json').resolve()),
        renderer_sha256=sha256(Path(__file__)), clips=rows, overview=str(overview.resolve()),
        overview_sha256=sha256(overview)), indent=2)+'\n')


if __name__ == '__main__':
    main()
