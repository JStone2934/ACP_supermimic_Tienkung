"""Continuous, nominal 30-second standing evaluation, with no automatic resets."""
import json
from pathlib import Path

import numpy as np
import torch

from ..rotations import mat_to_quat


def selection_key(report):
    full = not report['terminated'] and report['elapsed_sim_seconds'] >= 30.0
    excess = max(report['max_root_drift_m'] / .10, max(report['max_foot_drift_m']) / .05)
    return (not full, max(0., excess - 1.), report['max_root_drift_m'] + max(report['max_foot_drift_m']))


@torch.no_grad()
def evaluate(pi, env, task, output, *, iteration=0, checkpoint=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m, ctc, kin, dev = env.motion, task.ctc, task.mdp.kin, env.device
    env.reset(torch.arange(1, device=dev), m.clip_start[:1])
    initial = env.state()
    root0 = initial[0].clone()
    feet0 = kin.forward(*initial)[0][:, task.feet].clone()
    gs = torch.full((1,), 50, device=dev)
    hist = None
    traj = {k: [] for k in ('root_p', 'root_quat', 'q', 'f', 'alive', 'tgt_w', 'tgt_valid')}
    metrics = {k: [] for k in ('root_drift_m', 'foot_drift_m', 'tilt_deg', 'arm_error_deg', 'root_height_m')}
    terminated, steps = False, 0
    was_training = pi.training
    pi.eval()

    def record():
        S, f = env.state(), env.f
        for key, val in [('root_p', S[0]), ('root_quat', mat_to_quat(S[1])), ('q', S[4]), ('f', f),
                         ('alive', torch.tensor([not terminated], device=dev)),
                         ('tgt_w', ctc.next_pos[f, env.v]), ('tgt_valid', ctc.next_valid[f, env.v])]:
            traj[key].append(val.detach().clone())
        fp = kin.forward(*S)[0][:, task.feet]
        metrics['root_drift_m'].append((S[0][:, :2] - root0[:, :2]).norm(dim=-1)[0])
        metrics['foot_drift_m'].append((fp[:, :, :2] - feet0[:, :, :2]).norm(dim=-1)[0])
        metrics['tilt_deg'].append(torch.rad2deg(torch.acos(S[1][0, 2, 2].clamp(-1, 1))))
        metrics['arm_error_deg'].append(torch.rad2deg((S[4][0, task.arm_ids] - task.target_q[task.arm_ids]).abs()))
        metrics['root_height_m'].append(S[0][0, 2].clone())

    try:
        record()
        for step in range(task.episode_steps):
            o = env.observe()
            hist = o[:, None].repeat(1, pi.hist, 1) if hist is None else torch.cat((hist[:, 1:], o[:, None]), 1)
            a = task.compose(pi(hist, ctc.command(env.state(), env.f, env.v, gs, noise=False)), env.xi)
            if not torch.isfinite(a).all():
                raise FloatingPointError('nonfinite standing action')
            out = env.step(a)
            steps = step + 1
            terminated = bool(out['term'][0])
            record()
            if terminated:
                break
            if bool(out['done'][0]) and steps < task.episode_steps:
                raise RuntimeError('unexpected early standing timeout')
    finally:
        pi.train(was_training)

    metrics = {k: torch.stack(v).cpu().numpy() for k, v in metrics.items()}
    report = dict(iteration=iteration, checkpoint=checkpoint, elapsed_sim_seconds=steps * task.mdp.robot.control_dt,
                  terminated=terminated, continuous_without_reset=True, action_noise=0., nominal_physics=True,
                  max_root_drift_m=float(metrics['root_drift_m'].max()),
                  max_foot_drift_m=metrics['foot_drift_m'].max(0).tolist(),
                  max_tilt_deg=float(metrics['tilt_deg'].max()), min_root_height_m=float(metrics['root_height_m'].min()),
                  held_arm_p95_error_deg=np.percentile(metrics['arm_error_deg'], 95, axis=0).tolist(),
                  task_config=task.metadata(), metrics={k: v.tolist() for k, v in metrics.items()})
    c = task.cfg
    report['standing_30s_pass'] = bool(not terminated and steps == task.episode_steps
        and report['max_root_drift_m'] <= c.max_root_drift and max(report['max_foot_drift_m']) <= c.max_foot_drift
        and report['max_tilt_deg'] <= c.max_tilt_deg and max(report['held_arm_p95_error_deg']) <= c.max_arm_p95_deg)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output.with_suffix('.npz'), robot_metadata=json.dumps(task.mdp.robot.metadata()), names=np.array(m.names),
        variant=env.v.cpu().numpy(), boxes=m.clip_boxes[torch.arange(1, device=dev), env.v].cpu().numpy(),
        **{k: torch.stack(v).cpu().numpy() for k, v in traj.items()}, ref_root_p=m.root_p.cpu().numpy(),
        ref_root_R=m.root_R.cpu().numpy(), ref_q=m.q.cpu().numpy(), clip_start=m.clip_start.cpu().numpy(), clip_len=m.clip_len.cpu().numpy())
    return report
