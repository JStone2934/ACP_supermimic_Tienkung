"""Read-only checkpoint diagnostics: gradient decomposition and physical directional tests.

No optimizer updates, checkpoint writes, or training services are performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_fullbody import FullbodyTask, FullbodyPolicy, flat_boxes, validate_checkpoint
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.mdp import unflatten_state, flatten_state, normalize_xi
from supermimic.acp.networks import WorldModel, Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import bound_bootstrap
from supermimic.rotations import mat_to_quat


def samples(task, path, batch):
    z = np.load(path)
    d = {k: torch.from_numpy(z[k]).to(task.alpha.device) for k in z.files}
    if d['done'].any():
        raise ValueError('This diagnostic expects uninterrupted validation episodes')
    n = d['s'].shape[1]
    ctx0 = task.context(unflatten_state(d['s'][0]), task.alpha.new_tensor([[3., 0., 0., .2]]).repeat(n, 1), d['aim_weight'][0])
    ctx0['aim_target_w'] = d['aim_target_w'][0]
    ctx0['aim_tolerance'] = d['aim_tolerance'][0]
    times = torch.linspace(16, len(d['s']) - 16, batch, device=task.alpha.device).long()
    envs = torch.arange(batch, device=task.alpha.device) % n
    ts = times[:, None] - torch.arange(9, -1, -1, device=task.alpha.device)[None]
    hs = unflatten_state(d['s'][ts, envs[:, None]].flatten(0, 1))
    history = task.mdp.obs(hs, d['a'][ts - 1, envs[:, None]].flatten(0, 1),
                           d['xi'][ts, envs[:, None]].flatten(0, 1)).reshape(batch, 10, -1)
    return dict(S=unflatten_state(d['s'][times, envs]), history=history, ap=d['a'][times-1, envs],
                xi=d['xi'][times, envs], t=d['ep_step'][times, envs].long(),
                ctx={k: v[envs] for k, v in ctx0.items()},
                actions=torch.stack([d['a'][times+j, envs] for j in range(8)]), times=times, envs=envs)


def imagined(pi, wm, critic, task, cfg, b, actions=None):
    S, hist, ap, t = b['S'], b['history'], b['ap'], b['t']
    alive = torch.ones_like(t, dtype=torch.float32)
    stand = torch.zeros_like(alive); request = torch.zeros_like(alive)
    for j in range(cfg['horizon']):
        raw = pi(hist, task.command(S, b['ctx'])) if actions is None else actions[j]
        a, _ = task.compose_action(raw, b['xi'])
        S = wm.step(S, a, normalize_xi(b['xi']), task.mdp.scan(S, flat_boxes(S)))
        _, terms = task.reward_terms(S, a, ap, b['ctx'], raw=raw, xi=b['xi'])
        stand = stand + cfg['gamma']**j * alive * sum(v for k, v in terms.items() if k != 'request_limit')
        request = request + cfg['gamma']**j * alive * terms['request_limit']
        alive = alive * (1 - task.soft_termination(S, t + 1, b['ctx']))
        hist = torch.cat((hist[:, 1:], task.mdp.obs(S, a, b['xi'])[:, None]), 1)
        ap, t = a, t + 1
    features = task.critic_features(S, hist, ap, b['xi'], t, b['ctx'])
    raw_value = critic(features)
    tail = cfg['gamma']**cfg['horizon'] * alive * bound_bootstrap(raw_value, t, task, cfg)
    return dict(standing=stand, request=request, bootstrap=tail), features, raw_value, dict(S=S, hist=hist, ap=ap, t=t, alive=alive)


def dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def gradient_report(pi, wm, critic, task, cfg, b, optimizer_state):
    parts, features, values, tail_state = imagined(pi, wm, critic, task, cfg, b)
    params = list(pi.parameters())
    grads = {k: torch.autograd.grad(v.mean(), params, retain_graph=True) for k, v in parts.items()}
    total = tuple(sum(g[i] for g in grads.values()) for i in range(len(params)))
    norm = lambda g: dot(g, g).sqrt()
    stats = {k: dict(mean=float(parts[k].mean()), gradient_norm=float(norm(g)),
                    cosine_with_total=float(dot(g, total) / (norm(g) * norm(total)).clamp_min(1e-12)))
             for k, g in grads.items()}
    cosines = {a + '__' + c: float(dot(grads[a], grads[c]) / (norm(grads[a]) * norm(grads[c])).clamp_min(1e-12))
               for a in grads for c in grads if a < c}
    tail_grads = {}
    S, hist, ap, t, alive = (tail_state[k] for k in ['S', 'hist', 'ap', 't', 'alive'])
    for path in ['state', 'history', 'previous_action', 'survival']:
        F = task.critic_features(S if path == 'state' else tuple(x.detach() for x in S),
                                hist if path == 'history' else hist.detach(),
                                ap if path == 'previous_action' else ap.detach(), b['xi'], t, b['ctx'])
        v = bound_bootstrap(critic(F), t, task, cfg)
        obj = cfg['gamma']**cfg['horizon'] * (alive if path == 'survival' else alive.detach()) * v
        g = torch.autograd.grad(obj.mean(), params, retain_graph=True, allow_unused=True)
        tail_grads[path] = tuple(torch.zeros_like(p) if x is None else x for p, x in zip(params, g))
    reconstructed = tuple(sum(g[i] for g in tail_grads.values()) for i in range(len(params)))
    residual = tuple(x-y for x,y in zip(reconstructed, grads['bootstrap']))
    paths = {k: dict(gradient_norm=float(norm(g)), cosine_with_bootstrap=float(dot(g, grads['bootstrap']) /
                    (norm(g) * norm(grads['bootstrap'])).clamp_min(1e-12))) for k,g in tail_grads.items()}
    paths['relative_reconstruction_error'] = float(norm(residual) / norm(grads['bootstrap']).clamp_min(1e-12))
    del tail_grads, reconstructed, residual
    # Predict a single Adam parameter update without mutating any network or optimizer.
    pg = optimizer_state['param_groups'][0]; beta1,beta2 = pg['betas']
    updates = {}
    for mode in ['full_saved_moments', 'full_fresh_moments', 'no_bootstrap_fresh_moments']:
        objective_grad = total if mode != 'no_bootstrap_fresh_moments' else tuple(x+y for x,y in zip(grads['standing'],grads['request']))
        clip = min(1., 1. / (float(norm(objective_grad)) + 1e-6))
        deltas = []
        for i,g in enumerate(objective_grad):
            g = (-g * clip).clamp(-cfg['actor_grad_clip_value'], cfg['actor_grad_clip_value'])
            state = optimizer_state['state'].get(pg['params'][i], {}) if mode == 'full_saved_moments' else {}
            step = float(state.get('step', 0)) + 1
            m = state['exp_avg'].to(g.device) if state else torch.zeros_like(g)
            v = state['exp_avg_sq'].to(g.device) if state else torch.zeros_like(g)
            m = beta1*m + (1-beta1)*g; v=beta2*v + (1-beta2)*g.square()
            deltas.append(-pg['lr'] * (m/(1-beta1**step)) / ((v/(1-beta2**step)).sqrt()+pg['eps']))
        updates[mode] = dict(parameter_delta_norm=float(norm(deltas)),
                            first_order_objective_changes={k:float(dot(g,deltas)) for k,g in grads.items()})
    fg = torch.autograd.grad(parts['bootstrap'].mean(), features)[0]
    lengths = [('state', task.robot.inv_dim), ('command', 107), ('root_anchor', 3), ('yaw', 1),
               ('height', 1), ('feet', 6), ('previous_action', 29), ('xi', task.robot.xi_dim),
               ('scan', task.mdp.scan_dim), ('remaining_time', 1), ('history', 990), ('aim_weight', 1)]
    groups = {}; start = 0
    std = critic.norm.std()
    for name, length in lengths:
        g = fg[:, start:start+length]
        groups[name] = dict(feature_gradient_norm=float(g.norm()), std_min=float(std[start:start+length].min()),
                            near_constant_features=int((std[start:start+length] < .011).sum()))
        start += length
    assert start == features.shape[-1]
    capped = bound_bootstrap(values, b['t'] + cfg['horizon'], task, cfg)
    return dict(objectives=stats, cosines=cosines, total_gradient_norm=float(norm(total)),
                critic_features=groups, bootstrap_paths=paths, predicted_adam_updates=updates,
                raw_value_mean=float(values.mean()), raw_value_max=float(values.max()),
                capped_fraction=float((values > capped + 1e-6).float().mean()))


@torch.no_grad()
def physical_directions(pi, wm, critic, task, cfg, b, directions, epsilon):
    labels = ['baseline'] + [f'{k}_{sign}' for k in directions for sign in ['plus', 'minus']]
    action_sets = [b['actions']]
    for direction in directions.values():
        action_sets += [b['actions'] + epsilon * direction, b['actions'] - epsilon * direction]
    actions = torch.cat(action_sets, 1)
    groups, batch = len(labels), len(b['t'])
    repeat = lambda x: torch.cat([x] * groups, 0)
    S = tuple(repeat(x) for x in b['S'])
    ctx = {k: repeat(v) for k, v in b['ctx'].items()}
    sampler = lambda n: (task.alpha.new_tensor([[3., 0., 0., .2]]).repeat(n, 1), 0.)
    env = FullbodySim(task, groups * batch, sampler, eval_mode=True)
    ids = torch.arange(env.N, device=task.alpha.device)
    env.sim.reset(ids); env.scene.reset(ids)
    p, R, v, w, q, qd = S
    env.robot.write_root_link_pose_to_sim(torch.cat((p + env.scene.env_origins, mat_to_quat(R)), -1), env_ids=ids)
    env.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=ids)
    env.robot.write_joint_position_to_sim(q[:, env.to_internal], env_ids=ids)
    env.robot.write_joint_velocity_to_sim(qd[:, env.to_internal], env_ids=ids)
    env.robot.set_joint_position_target(q[:, env.to_internal])
    env.sim.forward()
    env.xi = repeat(b['xi']); env.a_prev = repeat(b['ap']); env.ep_step = repeat(b['t']); env.context = ctx
    hist = repeat(b['history'])
    restored_error = float((flatten_state(env.state()) - flatten_state(S)).abs().max())
    if restored_error > 1e-4:
        raise ValueError(f'Failed to restore diagnostic state: {restored_error}')
    total = torch.zeros(env.N, device=task.alpha.device); alive = torch.ones_like(total)
    eight = None
    for j in range(72):
        raw = actions[j] if j < 8 else pi(hist, task.command(env.state(), ctx))
        tr = env.step(raw)
        reward = sum(v for k, v in tr['reward_contributions'].items() if k != 'request_limit')
        total += cfg['gamma']**j * alive * reward
        alive *= (~tr['done']).float()
        hist = torch.cat((hist[:, 1:], env.observe()[:, None]), 1)
        if j == 7: eight = total.clone()
    augmented = dict(b, S=S, ctx=ctx, xi=repeat(b['xi']), ap=repeat(b['ap']), t=repeat(b['t']), history=repeat(b['history']))
    model_parts, _, _, _ = imagined(pi, wm, critic, task, cfg, augmented, actions=actions)
    result = dict(epsilon_rad=epsilon, restored_state_max_error=restored_error, cases={})
    for index, label in enumerate(labels):
        sl = slice(index*batch, (index+1)*batch)
        result['cases'][label] = dict(physical_reward_8=float(eight[sl].mean()), physical_reward_72=float(total[sl].mean()),
                                     model_reward_8=float(model_parts['standing'][sl].mean()),
                                     model_bootstrap=float(model_parts['bootstrap'][sl].mean()),
                                     alive_fraction_72=float(alive[sl].mean()))
    result['directions'] = {}
    for index, name in enumerate(directions):
        plus, minus = (1 + 2 * index) * batch, (2 + 2 * index) * batch
        delta = eight[plus:plus+batch] - eight[minus:minus+batch]
        result['directions'][name] = dict(physical_plus_minus_reward_8=float(delta.mean()),
            positive_physical_fraction_8=float((delta > 0).float().mean()),
            physical_plus_minus_reward_72=float((total[plus:plus+batch] - total[minus:minus+batch]).mean()),
            model_plus_minus_reward_8=float((model_parts['standing'][plus:plus+batch] - model_parts['standing'][minus:minus+batch]).mean()),
            model_plus_minus_bootstrap=float((model_parts['bootstrap'][plus:plus+batch] - model_parts['bootstrap'][minus:minus+batch]).mean()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True); p.add_argument('--trajectory', required=True); p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=64); p.add_argument('--physical', action='store_true')
    a = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(710)
    import warp as wp
    wp.config.quiet = True
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False); cfg = ck['config']
    task = FullbodyTask('cuda:0', cfg); validate_checkpoint(ck, task)
    pi = FullbodyPolicy().cuda(); pi.load_state_dict(ck['pi'])
    wm = WorldModel(task.mdp.scan_dim, robot=task.robot).cuda(); wm.load_state_dict(ck['wm']); wm.requires_grad_(False)
    critic = Critic(task.critic_dim).cuda(); critic.load_state_dict(ck['critic']); critic.requires_grad_(False)
    freeze_normalizers(pi, wm, critic)
    b = samples(task, a.trajectory, a.batch)
    result = dict(checkpoint=str(Path(a.ckpt).resolve()), checkpoint_iteration=ck['iteration'],
                  trajectory=str(Path(a.trajectory).resolve()), batch=a.batch,
                  gradient_decomposition=gradient_report(pi, wm, critic, task, cfg, b, ck['optimizers'][0]))
    json_save(a.out, result); print(json.dumps(result['gradient_decomposition']), flush=True)
    if a.physical:
        b = samples(task, a.trajectory, 16)
        actions = b['actions'].clone().requires_grad_(True)
        parts, _, _, _ = imagined(pi, wm, critic, task, cfg, b, actions=actions)
        directions = {}
        for key in ['standing', 'bootstrap']:
            grad = torch.autograd.grad(parts[key].sum(), actions, retain_graph=True)[0]
            gq = grad / task.alpha
            directions[key] = (gq / gq.abs().amax(dim=(0, 2), keepdim=True).clamp_min(1e-12) / task.alpha).detach()
        del parts, actions
        result['physical_direction_tests'] = []
        for eps in [.005, .02]:
            probe = physical_directions(pi, wm, critic, task, cfg, b, directions, eps)
            result['physical_direction_tests'].append(probe)
            json_save(a.out, result); print(json.dumps(probe), flush=True)


if __name__ == '__main__':
    main()
