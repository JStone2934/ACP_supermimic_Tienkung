"""Review every ranked native pose and nearby source motion; no simulation or selection."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import html
import json
from pathlib import Path
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from supermimic.acp.robots import get_robot


def normalize(rows, anchor):
    """Apply one constant world yaw/XY transform to the whole source clip."""
    w, x, y, z = anchor[3:7]
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = np.cos(yaw), np.sin(yaw)
    out = rows.copy().astype(float)
    out[:, :2] = (out[:, :2]-anchor[:2]) @ np.array([[c, -s], [s, c]])
    qyaw = np.array([np.cos(-yaw/2), 0., 0., np.sin(-yaw/2)])
    for row in out:
        q = np.empty(4)
        mujoco.mju_mulQuat(q, qyaw, row[3:7])
        row[3:7] = q
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--stills-only', action='store_true')
    a = p.parse_args()
    out = a.out or a.base/'pose_gallery'
    out.mkdir(parents=True, exist_ok=False)
    ranked = json.loads((a.base/'pose_candidates_ranked.json').read_text())
    geometry = json.loads((a.base/'pose_geometry.json').read_text())
    geo = {(x['source']['key'], x['source']['frame']): x for x in geometry['candidates']}
    robot = get_robot('tienkung3_29dof')
    spec = robot.get_spec()
    spec.worldbody.add_geom(name='gallery_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=(0, 0, .05), rgba=(.82, .84, .87, 1))
    spec.worldbody.add_light(pos=(2, -3, 5), dir=(-.3, .3, -1))
    spec.worldbody.add_light(pos=(-2, 3, 4), dir=(.3, -.3, -1))
    model = spec.compile()
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 640
    data = mujoco.MjData(model)
    addresses, _ = robot.joint_addresses(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0, 0, .9)
    camera.distance = 3.25
    camera.elevation = -10
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    manifest = []
    def render(renderer, row, azimuth):
        data.qpos[:7] = row[:7]
        data.qpos[addresses] = row[7:]
        mujoco.mj_forward(model, data)
        camera.azimuth = azimuth
        renderer.update_scene(data, camera)
        return Image.fromarray(renderer.render())
    with mujoco.Renderer(model, 640, 640) as renderer:
        for n, candidate in enumerate(ranked, 1):
            ident = f'{n:02d}'
            folder = out/ident
            folder.mkdir()
            g = geo[(candidate['key'], candidate['frame'])]
            with np.load(candidate['motion'], allow_pickle=False) as z:
                robot.validate_metadata(json.loads(str(z['robot_metadata'].item())))
                rows = z['joint_pos']
                frame = candidate['frame']
                assert rows.shape[1] == 36
                assert np.allclose(rows[frame, 7:], candidate['joint_pos'], atol=1e-6)
                pose = normalize(rows[frame:frame+1], rows[frame])[0]
                fps = float(candidate['fps'])
                start = max(0, min(frame-round(fps), len(rows)-round(2*fps)))
                stop = min(len(rows), start+round(2*fps))
            risk = []
            if not g['joint_soft_limits_passed']:
                risk.append('joint soft limits failed')
            if not g['geometry_passed'] and not risk:
                risk.append('geometry check failed; see metrics')
            entry = dict(id=ident, source=candidate, geometry=g,
                         risks=risk, video_start_frame=start, video_stop_frame_exclusive=stop,
                         candidate_time_in_video_s=(frame-start)/fps,
                         video_duration_s=(stop-start)/fps,
                         images={name: f'{ident}/{name}.jpg' for name in ('three_quarter', 'front', 'side')},
                         video=f'{ident}/motion.mp4')
            manifest.append(entry)
            for name, az in [('three_quarter', 35), ('front', 0), ('side', 90)]:
                picture = render(renderer, pose, az)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 640, 33), fill=(240, 243, 247))
                draw.text((10, 6), f'{ident}  {name}  chest {candidate["chest_tilt_deg"]:.1f} deg', font=font, fill='black')
                picture.save(folder/f'{name}.jpg', quality=92)
            print(f'stills {ident}', flush=True)
        contact = Image.new('RGB', (5*320, 7*390), '#f2f4f7')
        for i, entry in enumerate(manifest):
            x, y = (i%5)*320, (i//5)*390
            with Image.open(out/entry['images']['three_quarter']) as im:
                contact.paste(im.resize((320, 320)), (x, y))
            d = ImageDraw.Draw(contact)
            c = entry['source']
            d.text((x+8, y+321), f'{entry["id"]}  {c["key"].split("/")[-1].split("__")[0]}', font=small, fill='black')
            d.text((x+8, y+342), f'chest {c["chest_tilt_deg"]:.1f} deg | frame {c["frame"]}', font=small, fill='black')
            d.text((x+8, y+363), 'GEOMETRY PASS' if entry['geometry']['geometry_passed'] else 'RISK: JOINT SOFT LIMITS', font=small,
                   fill='#176742' if entry['geometry']['geometry_passed'] else '#b12e23')
        contact.save(out/'contact_sheet.jpg', quality=94)
        write_split_sheets(out, manifest)
        document = dict(scope='Source data kinematic replay, not physical stability evidence. All 35 candidates; no selection.',
                        normalization='One constant global yaw and XY transform anchored at candidate; source root Z and all relative joint poses preserved.',
                        order='Original pose_candidates_ranked.json order', candidates=manifest)
        (out/'manifest.json').write_text(json.dumps(document, indent=2))
        cards = []
        for e in manifest:
            c, g = e['source'], e['geometry']
            status = '几何检查通过' if g['geometry_passed'] else '风险：关节软限位未通过'
            cards.append(f'''<article id="p{e['id']}"><h2>{e['id']} · {html.escape(c['key'])}</h2>
<p class="{'pass' if g['geometry_passed'] else 'risk'}">{status}</p>
<p>胸部倾角 {c['chest_tilt_deg']:.2f}° · 骨盆倾角 {c['pelvis_tilt_deg']:.2f}° · 脚高差 {g['foot_height_gap_m']*1000:.2f} mm · 近似支撑裕度 {g['approximate_support_margin_m']*1000:.1f} mm</p>
<p>候选帧 {c['frame']} / 原动作 {c['frame']/c['fps']:.2f} s；视频从帧 {e['video_start_frame']} 开始，候选位于 {e['candidate_time_in_video_s']:.2f} s。保留原始动态，不保证这段动作静止。</p>
<div class="views">{''.join(f'<a href="{e["images"][v]}"><img loading="lazy" src="{e["images"][v]}" alt="{v}"></a>' for v in ('three_quarter','front','side'))}</div>
<video controls preload="none" poster="{e['images']['three_quarter']}" src="{e['video']}"></video>
<details><summary>来源与检查数据</summary><pre>{html.escape(json.dumps(e,ensure_ascii=False,indent=2))}</pre></details></article>''')
        (out/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>35个原始动作姿势候选</title><style>
body{font:16px system-ui;margin:24px;background:#f2f4f7;color:#182330}nav{position:sticky;top:0;background:white;padding:12px;z-index:2}nav a{display:inline-block;padding:6px}article{background:white;padding:20px;margin:24px 0;scroll-margin-top:110px}h2{font-size:20px}.views{display:flex}.views a{width:33.33%}.views img{width:100%}video{width:min(640px,100%)}.pass{color:#176742}.risk{color:#b12e23;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>35个候选姿势 / 原始动作回放</h1><p>请按编号选择；编号固定对应原始排序，包含26个几何通过与9个未通过候选。这里没有替你选择或训练。</p><p>真实原生 tienkung3_29dof 模型，统一去全局 yaw/XY；不修改关节。原始数据动作回放，不是物理仿真稳定性验证。视频固定三分之四镜头；图片依次为三分之四、前视、侧视。</p>
<a href="contact_sheet.jpg">打开完整联系表</a> · <a href="contact_01_12.jpg">01–12</a> · <a href="contact_13_24.jpg">13–24</a> · <a href="contact_25_35.jpg">25–35</a> · <a href="manifest.json">编号映射 / 完整来源</a><nav>'''+''.join(f'<a href="#p{e["id"]}">{e["id"]}</a>' for e in manifest)+'</nav>'+''.join(cards)+'</html>')
        print(f'CONTACT SHEET READY: {out.resolve()}', flush=True)
        if a.stills_only:
            return
        for e in manifest:
            c = e['source']
            with np.load(c['motion'], allow_pickle=False) as z:
                rows = normalize(z['joint_pos'][e['video_start_frame']:e['video_stop_frame_exclusive']], z['joint_pos'][c['frame']])
            fps = c['fps']/2
            pipe = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','640x640',
                                     '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/e['video'])], stdin=subprocess.PIPE)
            for j in range(0, len(rows), 2):
                im = render(renderer, rows[j], 35)
                draw = ImageDraw.Draw(im)
                t = j/c['fps']
                draw.rectangle((0,0,640,62),fill='#f0f3f7')
                draw.text((10,5),f'{e["id"]}  SOURCE DATA REPLAY   t={t:.2f}s',font=font,fill='black')
                draw.text((10,33),f'Candidate frame at {e["candidate_time_in_video_s"]:.2f}s  |  no simulation',font=font,fill='#b12e23' if abs(t-e['candidate_time_in_video_s'])<.06 else 'black')
                pipe.stdin.write(im.tobytes())
            pipe.stdin.close()
            if pipe.wait():
                raise RuntimeError(f'ffmpeg failed for {e["id"]}')
            print(f'video {e["id"]}',flush=True)


def write_split_sheets(out, manifest):
    """Larger review pages with both three-quarter and side views."""
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    for start in (0, 12, 24):
        entries = manifest[start:start+12]
        sheet = Image.new('RGB', (1920, 4*410+55), '#f2f4f7')
        draw = ImageDraw.Draw(sheet)
        draw.text((14, 12), f'Candidates {entries[0]["id"]}-{entries[-1]["id"]} | three-quarter + side | source data replay', font=font, fill='black')
        for j, e in enumerate(entries):
            x, y = (j%3)*640, (j//3)*410+55
            for col, view in enumerate(('three_quarter', 'side')):
                with Image.open(out/e['images'][view]) as im:
                    sheet.paste(im.resize((320,320)), (x+col*320,y))
            c = e['source']
            draw.text((x+9,y+323),f'{e["id"]}  {c["key"].split("/")[-1].split("__")[0]}',font=font,fill='black')
            draw.text((x+9,y+350),f'chest {c["chest_tilt_deg"]:.1f} deg | frame {c["frame"]}',font=font,fill='black')
            draw.text((x+9,y+377),'GEOMETRY PASS' if e['geometry']['geometry_passed'] else 'RISK: SOFT-LIMIT CHECK FAILED',
                      font=font,fill='#176742' if e['geometry']['geometry_passed'] else '#b12e23')
        sheet.save(out/f'contact_{entries[0]["id"]}_{entries[-1]["id"]}.jpg',quality=95)


if __name__ == '__main__':
    main()
