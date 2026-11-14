"""Held-out episodes for open-loop WM validation; never inserted into training replay."""
from pathlib import Path
import numpy as np
import torch
from .aiming_fullbody import flat_boxes
from .aiming_fullbody_env import FullbodySim
from .aiming_fullbody_io import json_save
from .mdp import unflatten_state, normalize_xi


@torch.no_grad()
def validate_world_model(pi, wm, task, sampler, cfg, out):
    env = FullbodySim(task, cfg['wm_validation_envs'], sampler)
    history = env.observe()[:, None].repeat(1, 10, 1)
    rows = []
    noise = task.alpha.new_full((29,), cfg['body_noise_rad'])
    noise[task.arm_ids] = cfg['arm_noise_rad']
    for _ in range(cfg['wm_validation_steps']):
        o = env.observe()
        history = torch.where((env.ep_step == 0)[:, None, None], o[:, None].expand_as(history), history)
        raw = pi(history, task.command(env.state(), env.context)) + torch.randn_like(env.a_prev) * noise / task.alpha
        tr = env.step(raw)
        rows.append({k: tr[k].clone() for k in ('s', 's1', 'a', 'xi', 'done', 'ep_step', 'aim_target_w', 'aim_tolerance', 'aim_weight')})
        history = torch.cat((history[:, 1:], env.observe()[:, None]), 1)
    data = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    horizons = sorted(set([1, 8, 16, 32, cfg['horizon']]))
    reports = {}
    for horizon in horizons:
        # Nonoverlapping validation windows; retain the terminating transition itself.
        starts = torch.arange(0, len(rows) - horizon + 1, horizon, device=task.alpha.device)
        if not len(starts):
            raise ValueError('Validation trajectory shorter than requested horizon')
        S = unflatten_state(data['s'][starts].flatten(0, 1))
        alive = torch.ones(len(S[0]), dtype=torch.bool, device=task.alpha.device)
        for j in range(horizon):
            a = data['a'][starts + j].flatten(0, 1)
            xi = data['xi'][starts + j].flatten(0, 1)
            S = wm.step(S, a, normalize_xi(xi), task.mdp.scan(S, flat_boxes(S)))
            if j < horizon - 1:
                alive &= ~data['done'][starts + j].flatten()
        truth = unflatten_state(data['s1'][starts + horizon - 1].flatten(0, 1))
        feet = task.kin.forward(*S)[0][:, task.feet]
        true_feet = task.kin.forward(*truth)[0][:, task.feet]
        h, d, _ = task.hand(S); ht, dt, _ = task.hand(truth)
        target = data['aim_target_w'][starts].flatten(0, 1)
        distance = (target - ht).norm(dim=-1)
        errors = dict(root_m=(S[0] - truth[0]).norm(dim=-1),
                      feet_m=(feet - true_feet).norm(dim=-1).amax(-1),
                      joints_rad=(S[4] - truth[4]).square().mean(-1).sqrt(),
                      tcp_m=(h - ht).norm(dim=-1),
                      direction_deg=torch.rad2deg(torch.acos((d * dt).sum(-1).clamp(-1, 1))),
                      tip_m=distance * (d - dt).norm(dim=-1))
        report = dict(windows=int(alive.sum()), total_windows=len(alive))
        for key, values in errors.items():
            values = values[alive]
            report[key + '_p95'] = float(values.quantile(.95)) if len(values) else float('inf')
            report[key + '_rmse'] = float(values.square().mean().sqrt()) if len(values) else float('inf')
        reports[str(horizon)] = report
    thresholds = dict(root_m_p95=cfg['wm_gate_root_m'], feet_m_p95=cfg['wm_gate_feet_m'])
    if (data['aim_weight'] > 0).any():
        thresholds['tip_m_p95'] = min(cfg['wm_gate_tip_m'], float(data['aim_tolerance'].min()) / 2)
    passed_by_horizon = {h: r['windows'] > 0 and all(np.isfinite(r[k]) and r[k] <= limit for k, limit in thresholds.items())
                         for h, r in reports.items()}
    # Longer predictions remain visible, but do not label an H8 model by H32 accuracy.
    passed = all(ok for h, ok in passed_by_horizon.items() if int(h) <= cfg['horizon'])
    result = dict(passed=passed, active_horizon=cfg['horizon'], passed_by_horizon=passed_by_horizon,
                  horizons=reports, thresholds=thresholds,
                  thresholds_status='stage-specific diagnostic, does not block optimizer updates or replace physical acceptance',
                  independent_episodes=True, exploration_rad=dict(arm=cfg['arm_noise_rad'], body=cfg['body_noise_rad']))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'), **{k: x.cpu().numpy() for k, x in data.items()})
    json_save(out.with_suffix('.json'), result)
    return result
