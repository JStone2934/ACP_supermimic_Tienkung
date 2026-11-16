"""Visualize the screened, unpromoted g1_style_224 Tienkung reward proxy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tools.render_g1_reward_spheres import (
    SphereRenderer, BG, TEXT, MUTED, COLORS, font, title, mujoco, np, Image, ImageDraw,
)
import torch
from supermimic.acp.robots import get_robot
from supermimic.acp.task import BodySpheres, _farthest_points


def candidate(dense):
    points, bounds, offset = [], [], 0
    for b, count in enumerate([28, 28]+[24]*7):
        cloud = dense.hulls[b]['vertices'].numpy()
        chosen = _farthest_points(cloud, count)
        inward = chosen-cloud.mean(0)
        chosen = chosen-.01*inward/np.linalg.norm(inward, axis=-1, keepdims=True).clip(min=1e-6)
        points.append(chosen)
        bounds.append((offset, offset+len(chosen)))
        offset += len(chosen)
    return SimpleNamespace(pts=torch.tensor(np.concatenate(points)),
                           rad=torch.full((offset,), .01), bounds=bounds)


class TienkungRenderer(SphereRenderer):
    def __init__(self, width, height, spheres):
        self.robot = get_robot('tienkung3_29dof')
        self.spheres = spheres
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
        for name, value in [('shoulder_roll_l_joint', .5), ('shoulder_roll_r_joint', -.5),
                            ('elbow_pitch_l_joint', -.5), ('elbow_pitch_r_joint', -.5)]:
            joint = self.model.joint(name).id
            value = np.clip(value, *self.model.jnt_range[joint])
            self.data.qpos[self.model.jnt_qposadr[joint]] = value
        mujoco.mj_forward(self.model, self.data)
        self.frame_ids = [self.model.body(frame).id for frame, _ in self.robot.contact_bodies.values()]
        self.world = self.world_centers()
        self.data.qpos[2] -= float((self.world[:56, 2]-self.spheres.rad[:56].numpy()).min())
        mujoco.mj_forward(self.model, self.data)
        self.world = self.world_centers()
        self.render = mujoco.Renderer(self.model, height, width, max_geom=2000)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 1
        self.camera_distance, self.camera_target = 3.5, (0, 0, .98)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    dense = BodySpheres('tienkung3_29dof')
    spheres = candidate(dense)
    viewer = TienkungRenderer(600, 860, spheres)
    canvas = Image.new('RGB', (1800, 1120), BG)
    title(canvas, '天工 + 224 球：候选接触奖励几何', '按天工真实碰撞凸包计算球心；全部半径 10 mm，未放大。候选方案，尚未用于正式训练。')
    for i, (mode, label) in enumerate([('model', '天工真实模型'), ('overlay', '半透明模型 + 224 个小球'), ('spheres', '只显示候选小球')]):
        canvas.paste(viewer.image(mode), (600*i, 140))
        ImageDraw.Draw(canvas).text((600*i+20, 100), label, font=font(25), fill=TEXT)
    draw = ImageDraw.Draw(canvas)
    labels = ['脚：28 × 2', '膝 / 小腿：24 × 2', '手：24 × 2', '肘 / 前臂：24 × 2', '骨盆：24']
    for i, label in enumerate(labels):
        x = 25+i*355
        draw.ellipse((x, 1020, x+20, 1040), fill=COLORS[i*2])
        draw.text((x+31, 1013), label, font=font(23), fill=TEXT)
    draw.text((25, 1065), '与此前 120 姿态筛查的 g1_style_224 完全相同；该筛查出现 4 次足部接触误判。展示姿态不代表策略表现。', font=font(21), fill=MUTED)
    canvas.save(args.out/'01_tienkung_224_overview.png')
    viewer.close()
    viewer = TienkungRenderer(600, 510, spheres)
    detail = Image.new('RGB', (1800, 1400), BG)
    title(detail, '天工 224 球近景：稀疏球体与真实外壳', '每个窗口单独放大部位，球与模型的相对尺度保持真实。球之间存在空隙，不是连续包覆。')
    panels = [(0, '左脚：28 球', '分布于整块脚部凸包，不只是足底'),
              (4, '左手：24 球', '根据天工 TCP 碰撞几何独立采样'),
              (2, '左膝 / 小腿：24 球', '覆盖膝部和小腿的对应接触几何'),
              (6, '左肘 / 前臂：24 球', '球心向凸包顶点均值内移 10 mm'),
              (8, '骨盆：24 球', '较大部件使用同样数量，覆盖更稀疏')]
    for i, (part, label, caption) in enumerate(panels):
        x, y = (i%3)*600, 110+(i//3)*610
        detail.paste(viewer.image('overlay', part=part), (x, y+60))
        draw = ImageDraw.Draw(detail)
        draw.text((x+18, y), label, font=font(25), fill=TEXT)
        draw.text((x+18, y+36), caption, font=font(18), fill=MUTED)
    draw = ImageDraw.Draw(detail)
    for i, line in enumerate(['总数：56 + 7 × 24 = 224', '每个球半径：10 mm', '左右对应部位数量相同', '',
                              '它不是把 G1 的点直接放大。', '它也不是一个已经验收的碰撞模型。', '',
                              '正式训练目前仍用原密集几何。', '头、胸、大腿不属于这 9 个奖励组。']):
        draw.text((1220, 780+i*44), line, font=font(23 if i<3 else 18), fill=TEXT if i<3 else MUTED)
    detail.save(args.out/'02_tienkung_224_closeups.png')
    rows = []
    names = list(viewer.robot.contact_bodies)
    for b, (s, e) in enumerate(spheres.bounds):
        for j in range(s, e):
            p = spheres.pts[j].numpy()
            rows.append(dict(index=j, body=names[b], frame=viewer.robot.contact_bodies[names[b]][0],
                             local_x=float(p[0]), local_y=float(p[1]), local_z=float(p[2]),
                             radius_m=float(spheres.rad[j]), world_x=float(viewer.world[j, 0]),
                             world_y=float(viewer.world[j, 1]), world_z=float(viewer.world[j, 2])))
    with (args.out/'tienkung_224_candidate.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    viewer.close()
    # Same camera and metric scale, to expose actual morphology differences.
    g1 = SphereRenderer(800, 900)
    tk = TienkungRenderer(800, 900, spheres)
    g1.camera_distance, g1.camera_target = tk.camera_distance, tk.camera_target
    comp = Image.new('RGB', (1600, 1120), BG)
    title(comp, '同为 224 球，体型与分配不同', '使用同一相机距离、同一米制尺度；两者均为便于观察的展示姿态。')
    comp.paste(g1.image('overlay'), (0, 140)); comp.paste(tk.image('overlay'), (800, 140))
    draw = ImageDraw.Draw(comp)
    draw.text((25, 100), 'G1：当前实现的实际 224 球', font=font(26), fill=TEXT)
    draw.text((825, 100), '天工：尚未采用的 224 球候选', font=font(26), fill=TEXT)
    draw.text((25, 1040), '脚 4 / 侧，手 48 / 侧；脚球 r=5 mm，其余 10 mm', font=font(23), fill=MUTED)
    draw.text((825, 1040), '脚 28 / 侧，手 24 / 侧；全部 r=10 mm', font=font(23), fill=MUTED)
    comp.save(args.out/'03_g1_tienkung_224_comparison.png')
    g1.close(); tk.close()
    manifest = dict(robot='tienkung3_29dof', candidate='g1_style_224', promoted=False,
                    spheres=224, radius_m=.01, render_radius_scale=1,
                    algorithm='FPS on each contact convex hull vertices; shift toward vertex mean by 0.01 m',
                    count_by_body={n:e-s for n,(s,e) in zip(names, spheres.bounds)},
                    scope='reward proxy only; static display pose; not training performance',
                    source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                   ['supermimic/acp/task.py', 'tools/probe_stage1_contact_sampling.py',
                                    'tools/render_tienkung_contact_candidate.py', 'tools/render_g1_reward_spheres.py', viewer.robot.mjcf_path]},
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.suffix in ('.png', '.csv')})
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
