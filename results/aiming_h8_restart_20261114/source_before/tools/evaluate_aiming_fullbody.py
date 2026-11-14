"""Full 30-second static physical evaluation, including frame zero and failed scenes."""
import argparse
from pathlib import Path
import numpy as np
import torch

from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_curriculum import STAGES, fixed_parameters
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.mdp import flatten_state


def runs(mask, dt=.02):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # Samples describe instants, so an interval of k samples spans (k-1)*dt.
    return starts * dt, np.maximum(ends - starts - 1, 0) * dt


def metric_report(x, rho, terminated, full_length, standing_only=False):
    root, feet, tilt, error, angle, valid = x[:, 0], x[:, 1:3], x[:, 3], x[:, 4], x[:, 5], x[:, 6].astype(bool)
    finite_state = x[:, 7].astype(bool)
    hits = valid & np.isfinite(error) & (error <= rho)
    starts, durations = runs(hits)
    _, gaps = runs(~hits)
    capture = starts[(durations >= 5. - 1e-8)]
    captured = bool(len(capture) and capture[0] <= 3. + 1e-8)
    standing = bool(full_length and not terminated and finite_state.all() and root.max() <= .10
                    and feet.max() <= .05 and tilt.max() <= 15.)
    def p95(values):
        # Nearest-rank retains infinity without inf-inf interpolation warnings.
        return float(np.sort(values)[max(0, int(np.ceil(.95 * len(values))) - 1)]) if len(values) else float('inf')
    later = np.arange(len(x)) * .02 >= 3.
    p_all, p_later = p95(error), p95(error[later])
    failures = []
    for bad, reason in ((not full_length, 'incomplete_30s'), (terminated, 'fall'),
                        (not finite_state.all(), 'nonfinite'), (root.max() > .10, 'root_drift'),
                        (feet.max() > .05, 'foot_drift'), (tilt.max() > 15., 'tilt'),
                        (not captured and not standing_only, 'capture_or_hold')):
        if bad: failures.append(reason)
    return dict(passed=standing and (standing_only or captured), standing_pass=standing, aiming_pass=captured,
                failures=failures, root_max_m=float(root.max()), feet_max_m=feet.max(0).tolist(), tilt_max_deg=float(tilt.max()),
                first_5s_hit_start_s=float(capture[0]) if len(capture) else None,
                longest_hit_s=float(durations.max()) if len(durations) else 0.,
                longest_miss_s=float(gaps.max()) if len(gaps) else 0.,
                error_p95_m=p_all, error_p95_finite=bool(np.isfinite(p_all)),
                t_ge_3s_error_p95_m=p_later, t_ge_3s_error_p95_finite=bool(np.isfinite(p_later)),
                angle_p95_deg=p95(angle), valid_ray_fraction=float(valid.mean()),
                t_ge_3s_hit_rate=float(hits[later].mean()) if later.any() else 0., frames=len(x), rho_m=float(rho))


@torch.no_grad()
def evaluate(pi, task, out, level=8):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    parameters = fixed_parameters(level, task.alpha.device)
    n = len(parameters)
    env = FullbodySim(task, n, lambda count: (parameters.clone(), 0. if level == 0 else task.weights['aim']), eval_mode=True)
    ctx = {k: v.clone() for k, v in env.context.items()}
    hist = env.observe()[:, None].repeat(1, 10, 1)
    records, states, actions, contributions, torques = [], [], [], [], []
    terminated = torch.zeros(n, dtype=torch.bool, device=task.alpha.device)
    first_termination = torch.full((n,), -1, device=task.alpha.device, dtype=torch.long)
    def record(S):
        feet = task.kin.forward(*S)[0][:, task.feet]
        e, ang, valid, _ = task.metrics(S, ctx)
        finite = torch.stack([z.flatten(1).isfinite().all(-1) for z in S]).all(0)
        records.append(torch.cat(((S[0][:, :2] - ctx['root_anchor_w'][:, :2]).norm(dim=-1)[:, None],
                                  (feet[..., :2] - ctx['feet_anchor_w'][..., :2]).norm(dim=-1),
                                  torch.rad2deg(torch.acos(S[1][:, 2, 2].clamp(-1, 1)))[:, None],
                                  e[:, None], ang[:, None], valid[:, None], finite[:, None]), -1).cpu().numpy())
        states.append(flatten_state(S).cpu().numpy())
    record(env.state())
    for t in range(task.episode_steps):
        tr = env.step(pi(hist, task.command(env.state(), ctx)))
        first_termination[(first_termination < 0) & tr['terminated']] = t + 1
        terminated |= tr['terminated']
        actions.append(tr['a'].cpu().numpy())
        contributions.append(torch.stack(list(tr['reward_contributions'].values()), -1).cpu().numpy())
        # End-of-control PD diagnostic, not substep energy.
        S = env.state()
        tau = (torch.tensor(task.robot.kp, device=task.alpha.device) * (tr['q_des'] - S[4])
               - torch.tensor(task.robot.kd, device=task.alpha.device) * S[5])
        torques.append(tau.cpu().numpy())
        record(S)
        hist = torch.cat((hist[:, 1:], env.observe()[:, None]), 1)
    x = np.stack(records)
    reports = [metric_report(x[:, i], float(parameters[i, 3]), bool(terminated[i]), len(x) == 1501, level == 0) for i in range(n)]
    for i, report in enumerate(reports):
        report['parameters'] = parameters[i].cpu().tolist()
        report['first_termination_step'] = int(first_termination[i])
    report = dict(level=level, stage=STAGES[level]['name'], passed=all(r['passed'] for r in reports),
                  scenarios=reports, episode_seconds=30., task_config=task.metadata(),
                  reward_term_names=list(task.weights), trajectory=str(out.with_suffix('.npz')))
    np.savez_compressed(out.with_suffix('.npz'), s=np.stack(states), a_exec=np.stack(actions), metrics=x,
                        reward_contributions=np.stack(contributions), end_step_pd_torque=np.stack(torques),
                        parameters=parameters.cpu().numpy(), first_termination=first_termination.cpu().numpy(),
                        **{k: v.cpu().numpy() for k, v in ctx.items()})
    json_save(out.with_suffix('.json'), report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--stage', choices=[s['name'] for s in STAGES], default='C3b')
    a = p.parse_args(); torch.set_num_threads(8)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    task = FullbodyTask('cuda:0', ck['config']); validate_checkpoint(ck, task)
    pi = FullbodyPolicy(task.robot).to(task.alpha.device); pi.load_state_dict(ck['pi']); pi.eval()
    report = evaluate(pi, task, a.out, [s['name'] for s in STAGES].index(a.stage))
    from supermimic.acp.robots import sha256
    report.update(checkpoint=str(Path(a.ckpt).resolve()), checkpoint_sha256=sha256(a.ckpt))
    json_save(Path(a.out).with_suffix('.json'), report)
    print(dict(passed=report['passed'], stage=a.stage, scenarios=len(report['scenarios'])))


if __name__ == '__main__':
    main()
