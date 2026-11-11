"""Render the exact G1 BodySpheres used by the contact reward, at true radius."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.acp.robots import get_robot
from supermimic.acp.task import BodySpheres

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
COLORS = [(54, 157, 255), (54, 157, 255), (62, 211, 146), (62, 211, 146),
          (255, 156, 56), (255, 156, 56), (255, 86, 146), (255, 86, 146), (177, 128, 255)]
BG, TEXT, MUTED = (16, 23, 34), (242, 246, 250), (180, 195, 213)


def font(size):
    return ImageFont.truetype(FONT, size)


class SphereRenderer:
    def __init__(self, width, height):
        self.robot = get_robot('g1')
        self.spheres = BodySpheres(self.robot)
        spec = self.robot.get_spec()
        spec.worldbody.add_geom(name='display_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=(0, 0, 1), rgba=(.10, .13, .18, 1))
        self.model = spec.compile()
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = width, height
        self.model.vis.headlight.ambient[:] = .5
        self.model.vis.headlight.diffuse[:] = .7
        self.model.vis.headlight.specular[:] = .15
        self.data = mujoco.MjData(self.model)
        qa, _ = self.robot.joint_addresses(self.model)
        self.data.qpos[:7] = [0, 0, self.robot.stand_height, 1, 0, 0, 0]
        self.data.qpos[qa] = self.robot.default_q
        for name, value in [('left_shoulder_roll_joint', .65), ('right_shoulder_roll_joint', -.65),
                            ('left_elbow_joint', .7), ('right_elbow_joint', .7)]:
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
        mujoco.mj_forward(self.model, self.data)
        self.frame_ids = [self.model.body(frame).id for frame, _ in self.robot.contact_bodies.values()]
        self.world = self.world_centers()
        self.data.qpos[2] -= float((self.world[:8, 2]-self.spheres.rad[:8].numpy()).min())
        mujoco.mj_forward(self.model, self.data)
        self.world = self.world_centers()
        self.render = mujoco.Renderer(self.model, height, width, max_geom=2000)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 1

    def world_centers(self):
        out = []
        for b, (s, e) in enumerate(self.spheres.bounds):
            body = self.frame_ids[b]
            out.extend(self.data.xpos[body]+self.spheres.pts[s:e].numpy()@self.data.xmat[body].reshape(3, 3).T)
        return np.asarray(out)

    def image(self, mode='overlay', azimuth=35, part=None):
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = azimuth, -12, getattr(self, 'camera_distance', 2.45)
        cam.lookat[:] = getattr(self, 'camera_target', (0, 0, .72))
        if part is not None:
            s, e = self.spheres.bounds[part]
            points = self.spheres.pts[s:e].numpy()
            cam.lookat[:] = (points.min(0)+points.max(0))/2
            cam.distance = float(max(points.max(0)-points.min(0)))*2.6+.08
            cam.elevation = 80 if part == 0 else -22
            cam.azimuth = 15 if part == 0 else 50
        self.render.update_scene(self.data, cam, scene_option=self.option)
        scene = self.render.scene
        if part is not None:
            frame = self.frame_ids[part]
            origin, rotation = self.data.xpos[frame], self.data.xmat[frame].reshape(3, 3)
            links = list(self.robot.contact_bodies.values())[part][1]
        for i in range(scene.ngeom):
            geom = scene.geoms[i]
            if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid < 0:
                continue
            gid = geom.objid
            is_floor = self.model.geom(gid).name == 'display_floor'
            visual = self.model.geom_group[gid] == 1
            if part is not None:
                selected = self.model.body(self.model.geom_bodyid[gid]).name in links
                geom.pos[:] = rotation.T@(geom.pos-origin)
                geom.mat[:] = rotation.T@geom.mat
                geom.rgba[:] = (*np.array((.68, .72, .78)), .18 if selected and visual and mode!='spheres' else 0)
            elif is_floor:
                pass
            elif mode == 'model':
                geom.rgba[3] = 1 if visual else 0
            elif mode == 'overlay':
                geom.rgba[:] = (.65, .71, .80, .16 if visual else 0)
            else:
                geom.rgba[3] = 0
        if mode != 'model':
            for b, (start, end) in enumerate(self.spheres.bounds):
                if part is not None and b != part:
                    continue
                for j in range(start, end):
                    color = COLORS[b]
                    if b in (4, 5) and j-start >= 24:
                        color = (255, 217, 76)
                    position = self.spheres.pts[j].numpy() if part is not None else self.world[j]
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([float(self.spheres.rad[j]), 0, 0]),
                                       position.astype(np.float64), np.eye(3).ravel(),
                                       np.array([*(np.array(color)/255), 1], dtype=np.float32))
                    geom.specular, geom.shininess = .3, .5
                    scene.ngeom += 1
        return Image.fromarray(self.render.render())

    def close(self):
        self.render.close()


def title(image, heading, subtitle, x=20, y=14):
    draw = ImageDraw.Draw(image)
    draw.text((x, y), heading, font=font(29), fill=TEXT)
    draw.text((x, y+44), subtitle, font=font(18), fill=MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--video', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    viewer = SphereRenderer(600, 860)
    canvas = Image.new('RGB', (1800, 1100), BG)
    title(canvas, 'G1 接触奖励：224 个小球到底分布在哪里', '直接读取当前 BodySpheres 的中心与半径；同一展示姿态、同一相机，球体没有放大。')
    for i, (mode, label) in enumerate([('model', '机器人外观'), ('overlay', '半透明模型 + 实际奖励小球'), ('spheres', '仅显示全部 224 个奖励小球')]):
        canvas.paste(viewer.image(mode), (600*i, 140))
        ImageDraw.Draw(canvas).text((600*i+20, 100), label, font=font(25), fill=TEXT)
    draw = ImageDraw.Draw(canvas)
    legend = [('脚：4 × 2', COLORS[0]), ('膝 / 小腿：24 × 2', COLORS[2]),
              ('手 / 腕：48 × 2', COLORS[4]), ('肘：24 × 2', COLORS[6]), ('骨盆：24', COLORS[8])]
    for i, (label, color) in enumerate(legend):
        x = 30+i*350
        draw.ellipse((x, 1020, x+20, 1040), fill=color)
        draw.text((x+31, 1013), label, font=font(23), fill=TEXT)
    draw.text((30, 1060), '蓝色脚底球半径 5 mm；其他球半径 10 mm。橙色为腕部，黄色为橡胶手。展示姿态用于看清分布。', font=font(20), fill=MUTED)
    canvas.save(args.out/'01_g1_spheres_overview.png')
    # Isolate each part in its own semantic body frame, keeping true sphere radii.
    viewer.close()
    viewer = SphereRenderer(600, 510)
    details = Image.new('RGB', (1800, 1400), BG)
    title(details, '局部近景：几何表面与实际小球', '各窗口单独放大部位，球与模型的相对尺度保持真实；左右对应部位采用相同数量。')
    panels = [(0, '左脚：4 个，半径 5 mm', '从足底看：分别位于前后、左右四角'),
              (4, '左手：48 个，半径 10 mm', '腕部 24 个（橙）+ 橡胶手 24 个（黄）'),
              (2, '左膝组：24 个，半径 10 mm', '沿膝盖和小腿外壳分布，不只在关节处'),
              (6, '左肘组：24 个，半径 10 mm', '围绕肘部 / 前臂这块碰撞网格分布'),
              (8, '骨盆：24 个，半径 10 mm', '分布在骨盆轮廓碰撞网格表面')]
    for i, (part, label, caption) in enumerate(panels):
        col, row = i%3, i//3
        x, y = col*600, 110+row*610
        details.paste(viewer.image('overlay', part=part), (x, y+60))
        draw = ImageDraw.Draw(details)
        draw.text((x+18, y), label, font=font(24), fill=TEXT)
        draw.text((x+18, y+35), caption, font=font(17), fill=MUTED)
    draw = ImageDraw.Draw(details)
    x, y = 1230, 760
    for i, line in enumerate(['总数核对', '脚：4 × 2 = 8', '膝：24 × 2 = 48', '手：48 × 2 = 96',
                              '肘：24 × 2 = 48', '骨盆：24', '合计：224 个', '',
                              '小球用于计算奖励中的接触距离。', '头、胸、大腿没有纳入这 9 组接触奖励。']):
        draw.text((x, y+i*43), line, font=font(24 if i<7 else 18), fill=TEXT if i<7 else MUTED)
    details.save(args.out/'02_g1_spheres_closeups.png')
    # Export the exact centers so every marker is auditable.
    names = list(viewer.robot.contact_bodies)
    rows = []
    for b, (s, e) in enumerate(viewer.spheres.bounds):
        for j in range(s, e):
            local = viewer.spheres.pts[j].numpy()
            rows.append(dict(index=j, body=names[b], frame=viewer.robot.contact_bodies[names[b]][0],
                             local_x=float(local[0]), local_y=float(local[1]), local_z=float(local[2]),
                             radius_m=float(viewer.spheres.rad[j]),
                             world_x=float(viewer.world[j, 0]), world_y=float(viewer.world[j, 1]), world_z=float(viewer.world[j, 2])))
    with (args.out/'g1_reward_spheres.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    viewer.close()
    if args.video:
        viewer = SphereRenderer(640, 720)
        output = args.out/'03_g1_spheres_rotation.mp4'
        process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                    '-s', '1280x840', '-r', '24', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
                                    '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
        try:
            for frame in range(192):
                image = Image.new('RGB', (1280, 840), BG)
                title(image, 'G1 奖励小球：360° 相机对照', '左：半透明模型 + 小球    右：仅 224 个小球    |    真实半径，固定展示姿态')
                az = 35+360*frame/192
                image.paste(viewer.image('overlay', az), (0, 110))
                image.paste(viewer.image('spheres', az), (640, 110))
                process.stdin.write(np.asarray(image).tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError('video encoding failed')
        finally:
            if process.poll() is None:
                process.kill()
            viewer.close()
    manifest = dict(robot='g1', sphere_count=len(rows), rendering='exact BodySpheres centers and radii; camera rotation, not a policy rollout',
                    point_radius_scale=1, pose='default lower body; shoulders opened to +/-0.65 rad and elbows 0.7 rad for visibility',
                    count_by_body={name:e-s for name,(s,e) in zip(names, viewer.spheres.bounds)},
                    source_sha256=hashlib.sha256(Path('supermimic/acp/task.py').read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(Path(viewer.robot.mjcf_path).read_bytes()).hexdigest(),
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.suffix in ('.png', '.mp4', '.csv')})
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
