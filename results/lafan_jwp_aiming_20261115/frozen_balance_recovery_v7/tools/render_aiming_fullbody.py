"""Render full v2 physical recordings and plot all-frame errors without a reference dataset."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
from pathlib import Path
import subprocess
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from supermimic.acp.robots import get_robot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory'); p.add_argument('out'); p.add_argument('--scenario', type=int, default=0)
    args = p.parse_args(); z = np.load(args.trajectory)
    s = z['s'][:, args.scenario]; metrics = z['metrics'][:, args.scenario]
    target = z['aim_target_w'][args.scenario]; normal = z['aim_normal_w'][args.scenario]
    rho = z['aim_tolerance'][args.scenario, 0]
    up = np.array([0., 0., 1.]); left = np.cross(up, normal)
    robot = get_robot('tienkung3_29dof'); spec = robot.get_spec()
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, .05), rgba=(.25, .28, .32, 1))
    spec.worldbody.add_light(pos=(0, -3, 5), dir=(0, 0, -1))
    m = spec.compile(); m.vis.global_.offwidth = 960; m.vis.global_.offheight = 540
    data = mujoco.MjData(m); adr, _ = robot.joint_addresses(m); tcp = m.body('right_tcp_link').id
    camera = mujoco.MjvCamera(); camera.lookat[:] = (1.1, 0, 1.)
    camera.distance = 5.; camera.azimuth = 110; camera.elevation = -12
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pipe = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x540',
                             '-r', '25', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', args.out], stdin=subprocess.PIPE)
    def line(scene, a, b, color, width=.006):
        geom = scene.geoms[scene.ngeom]; scene.ngeom += 1
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
    with mujoco.Renderer(m, 540, 960) as render:
        for t in range(0, len(s) - 1, 2):
            if not np.isfinite(s[t]).all():
                raise ValueError(f'Cannot render nonfinite physical state at frame {t}; see trajectory/report')
            data.qpos[:7] = s[t, :7]; data.qpos[adr] = s[t, 13:42]
            mujoco.mj_forward(m, data); render.update_scene(data, camera)
            h = data.xpos[tcp]; direction = -data.xmat[tcp].reshape(3, 3)[:, 2]
            denominator = normal @ direction
            distance = normal @ (target - h) / denominator if denominator > 1e-4 else 3.
            good = bool(metrics[t, 6] and metrics[t, 4] <= rho)
            color = (.15, 1., .25, 1) if good else (1., .2, .1, 1)
            theta = np.linspace(0, 2 * np.pi, 49)
            ring = target + rho * (np.cos(theta)[:, None] * left + np.sin(theta)[:, None] * up)
            for a, b in zip(ring[:-1], ring[1:]): line(render.scene, a, b, (.9, .8, .1, 1))
            line(render.scene, h, h + np.clip(distance, .01, 5) * direction, color, .008)
            frame = Image.fromarray(render.render())
            ImageDraw.Draw(frame).text((12, 12), f'Fullbody v2  t={t*.02:.2f}s  radius={rho:.2f}m  error={metrics[t,4]:.3f}m', fill='white')
            pipe.stdin.write(frame.tobytes())
    pipe.stdin.close()
    if pipe.wait(): raise RuntimeError('ffmpeg failed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    time = np.arange(len(s)) * .02
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(time, metrics[:, 4]); axes[0].axhline(rho, color='r', ls='--'); axes[0].set_ylabel('Aim error (m)')
    for i, label in enumerate(('root', 'left foot', 'right foot')): axes[1].plot(time, metrics[:, i], label=label)
    axes[1].axhline(.05, color='r', ls='--'); axes[1].axhline(.10, color='k', ls='--')
    axes[1].legend(); axes[1].set_ylabel('Drift (m)')
    axes[2].plot(time, metrics[:, 3]); axes[2].axhline(15., color='r', ls='--')
    axes[2].set_ylabel('Tilt (degrees)'); axes[2].set_xlabel('Time since reset (s)')
    fig.tight_layout(); fig.savefig(Path(args.out).with_suffix('.png')); plt.close(fig)


if __name__ == '__main__': main()
