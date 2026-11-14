"""Fullbody v2: physical replay, supervised WM, differentiable actor and finite-episode critic."""
import argparse
import fcntl
import json
from itertools import count
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .aiming_fullbody import (DEFAULT_CONFIG, TASK_NAME, CONTEXT_FIELDS, REPLAY_FIELDS, FullbodyTask,
                             FullbodyPolicy, FullbodyReplay, flat_boxes, validate_checkpoint)
from .aiming_fullbody_curriculum import Curriculum, StageExploration, STAGES
from .aiming_fullbody_env import FullbodySim
from .aiming_fullbody_io import code_identity, json_save, save_checkpoint
from .aiming_fullbody_validation import validate_world_model
from .mdp import unflatten_state, normalize_xi
from .networks import WorldModel, Critic
from .standing import freeze_normalizers


def wm_loss(wm, task, windows):
    S = unflatten_state(windows['s'][:, 0])
    loss = S[0].new_zeros(())
    scale = wm.out_norm.std().clamp_min(.003)
    for j in range(windows['a'].shape[1]):
        # Invalid tails must not propagate arbitrary post-reset actions through the old episode.
        truth = unflatten_state(windows['s'][:, j])
        S = tuple(torch.where(windows['mask'][:, j].reshape(-1, *([1] * (x.ndim - 1))).bool(), x, y)
                  for x, y in zip(S, truth))
        S = wm.step(S, windows['a'][:, j], normalize_xi(windows['xi'][:, j]), task.mdp.scan(S, flat_boxes(S)))
        err = (wm.state_error(S, unflatten_state(windows['s1'][:, j])) / scale).square().mean(-1)
        loss += (err * windows['mask'][:, j]).sum()
    return loss / windows['mask'].sum().clamp_min(1)


def bound_bootstrap(value, step, task, cfg):
    """Only cap bootstrap estimates; critic regression retains its uncapped gradient."""
    if not cfg.get('bound_value_bootstrap', False):
        return value
    remaining = (task.episode_steps - step).clamp_min(0).to(value.dtype)
    gamma = cfg['gamma']
    upper = task.max_step_reward * (remaining if gamma == 1. else (1. - gamma ** remaining) / (1. - gamma))
    return torch.minimum(value, upper)


def update_actor(pi, wm, critic, task, buffer, optim_pi, optim_v, cfg, *, train_actor=True):
    start, histories = buffer.sample_starts(cfg['pi_batch'], 10)
    S, history = unflatten_state(start['s']), histories['o']
    xi, ap, t = start['xi'], start['a_prev'], start['ep_step'].long()
    start_step = t.clone()
    ctx = {k: start[k] for k in CONTEXT_FIELDS}
    alive = torch.ones(len(t), device=t.device)
    objective = torch.zeros_like(alive)
    xs, rewards, dones, masks = [], [], [], []
    gamma = cfg['gamma']
    wm.requires_grad_(False); critic.requires_grad_(False)
    optim_pi.zero_grad(set_to_none=True)
    actor_grad = 0.
    def imagined_step(S, history, ap, t):
        raw = pi(history, task.command(S, ctx))
        a, _ = task.compose_action(raw, xi)
        next_state = wm.step(S, a, normalize_xi(xi), task.mdp.scan(S, flat_boxes(S)))
        reward = task.reward(next_state, a, ap, ctx, raw=raw, xi=xi)
        done = task.soft_termination(next_state, t + 1, ctx)
        next_history = torch.cat((history[:, 1:], task.mdp.obs(next_state, a, xi)[:, None]), 1)
        return next_state, next_history, a, reward, done
    try:
        with torch.set_grad_enabled(train_actor):
            for j in range(cfg['horizon']):
                with torch.no_grad():
                    xs.append(task.critic_features(S, history, ap, xi, t, ctx))
                masks.append(alive.detach())
                if train_actor:
                    S, history, ap, r, d = checkpoint(imagined_step, S, history, ap, t, use_reentrant=False)
                else:
                    S, history, ap, r, d = imagined_step(S, history, ap, t)
                objective = objective + gamma**j * alive * r
                alive = alive * (1 - d); t = t + 1
                rewards.append(r.detach()); dones.append(d.detach())
            last = task.critic_features(S, history, ap, xi, t, ctx)
            objective = objective + gamma**cfg['horizon'] * alive * bound_bootstrap(critic(last), t, task, cfg)
            actor_loss = -objective.mean()
        if train_actor:
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(pi.parameters(), 1., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(pi.parameters(), cfg.get('actor_grad_clip_value', .1))
            optim_pi.step()
    finally:
        wm.requires_grad_(True); critic.requires_grad_(True)
    X, mask = torch.stack(xs), torch.stack(masks)
    critic.norm.update(X)
    with torch.no_grad():
        V = critic(torch.cat((X, last.detach()[None]), 0))
        steps = start_step[None] + torch.arange(cfg['horizon'] + 1, device=t.device)[:, None]
        V = bound_bootstrap(V, steps, task, cfg)
        ret = V[-1]; targets = []
        for j in reversed(range(cfg['horizon'])):
            ret = rewards[j] + gamma * (1 - dones[j]) * ((1 - cfg['lambda_return']) * V[j + 1] + cfg['lambda_return'] * ret)
            targets.append(ret)
        Y = torch.stack(targets[::-1])
    X, Y, mask = X.flatten(0, 1), Y.flatten(), mask.flatten()
    for _ in range(cfg['critic_epochs']):
        for ids in torch.randperm(len(X), device=X.device).chunk(cfg['critic_minibatches']):
            optim_v.zero_grad()
            critic_loss = ((critic(X[ids]) - Y[ids]).square() * mask[ids]).sum() / mask[ids].sum().clamp_min(1)
            critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            optim_v.step()
    return dict(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                actor_grad=float(actor_grad), critic_grad=float(critic_grad),
                bootstrap_max=float(V.max()), return_target_max=float(Y.max()))


def training_phase(iteration, start, cfg, critic_warmup_done, *, resume=False):
    if iteration < cfg['wm_only_iters'] or (resume and iteration - start < cfg['resume_wm_iters']):
        return 'wm'
    return 'critic' if critic_warmup_done < cfg.get('critic_warmup_iters', 0) else 'joint'


def exploration_fraction(updates, cfg):
    return max(cfg.get('noise_min_fraction', 0.), 1. - updates / cfg['noise_decay_updates'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=DEFAULT_CONFIG); p.add_argument('--log-dir', '--log_dir', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--init'); group.add_argument('--resume')
    p.add_argument('--horizon', type=int, choices=(8, 16, 32))
    p.add_argument('--iterations', type=int, help='Total iteration limit; 0 runs until curriculum acceptance, including on resume')
    p.add_argument('--smoke', action='store_true', help='5 iterations covering WM, critic warmup and joint updates')
    p.add_argument('--capacity-check', action='store_true', help='With --smoke: use configured env/batch counts and horizon for a short capacity check')
    args = p.parse_args()
    ck = torch.load(args.resume or args.init, map_location='cpu', weights_only=False)
    if ck.get('task') != TASK_NAME:
        raise ValueError('Prepare an identified v2 checkpoint first; old task resume is forbidden')
    if args.resume and ck.get('stage') != 'training':
        raise ValueError('--resume requires a complete v2 training checkpoint')
    if args.init and ck.get('stage') != 'initialization':
        raise ValueError('--init requires the prepared v2 initialization')
    cfg = dict(ck['config'] if args.resume else json.loads(Path(args.config).read_text()))
    cfg['source'] = ck['config']['source']
    if args.iterations is not None:
        cfg['iterations'] = args.iterations
    if not isinstance(cfg['iterations'], int) or cfg['iterations'] < 0:
        raise ValueError('iterations must be a nonnegative integer; 0 means unlimited')
    # Older v2 checkpoints keep their saved schedule and optimizer settings on resume.
    cfg.setdefault('critic_warmup_iters', 0)
    cfg.setdefault('wm_validation_every', cfg['ckpt_every'])
    cfg.setdefault('log_every', 100)
    cfg.setdefault('noise_min_fraction', 0.)
    cfg.setdefault('actor_betas', [.9, .999])
    cfg.setdefault('actor_grad_clip_value', .1)
    if cfg.get('task') != TASK_NAME or cfg.get('version') != 2 or cfg.get('hist') != 10:
        raise ValueError('Invalid v2 configuration')
    if args.resume and ck.get('smoke') and not args.smoke:
        raise ValueError('Smoke checkpoints cannot seed formal training; use the prepared initialization')
    if args.capacity_check and (not args.smoke or args.resume):
        raise ValueError('--capacity-check requires --smoke and a fresh --init')
    if args.smoke:
        capacity = {k: cfg[k] for k in ('num_envs', 'wm_batch', 'pi_batch', 'horizon', 'buffer_steps')}
        cfg.update(num_envs=8, wm_batch=8, pi_batch=4, buffer_steps=100, wm_batches=1,
                   initial_collect_steps=32, wm_only_iters=2, critic_warmup_iters=1, resume_wm_iters=2,
                   iterations=5, ckpt_every=5, critic_epochs=1, critic_minibatches=1,
                   wm_validation_steps=64, wm_validation_envs=2, wm_validation_every=2, log_every=1)
        if args.capacity_check:
            cfg.update(capacity)
    if args.horizon:
        cfg['horizon'] = args.horizon
    if cfg['horizon'] not in (4, 8, 16, 32):
        raise ValueError('Supported imagination lengths: 4, 8, 16, 32')
    out = Path(args.log_dir)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('New training requires an empty output directory; use --resume for an existing run')
    out.mkdir(parents=True, exist_ok=True)
    run_lock = (out / '.trainer.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another trainer owns {out}') from exc
    torch.set_num_threads(8)
    import warp as wp
    wp.config.quiet = True
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    task = FullbodyTask('cuda:0', cfg); validate_checkpoint(ck, task)
    pi = FullbodyPolicy(task.robot).to(task.alpha.device); pi.load_state_dict(ck['pi'])
    wm = WorldModel(task.mdp.scan_dim, robot=task.robot).to(task.alpha.device); wm.load_state_dict(ck['wm'])
    freeze_normalizers(pi, wm)
    critic = Critic(task.critic_dim).to(task.alpha.device)
    optim = [torch.optim.Adam(model.parameters(), lr=cfg[name]) for model, name in
             ((pi, 'actor_lr'), (wm, 'wm_lr'), (critic, 'critic_lr'))]
    optim[0].param_groups[0]['betas'] = tuple(cfg['actor_betas'])
    cur = Curriculum(task.alpha.device, ck.get('curriculum'))
    exploration = None
    if cfg.get('stage_exploration'):
        if args.resume and 'exploration' not in ck:
            raise ValueError('Stage exploration requires an explicit migrated initialization')
        exploration = StageExploration(cfg['stage_exploration'], cur.level, ck.get('exploration'))
    start = ck['iteration'] if args.resume else 0
    updates = ck['actor_updates'] if args.resume else 0
    critic_warmup_done = ck.get('critic_warmup_done', cfg['critic_warmup_iters'] if updates else 0) if args.resume else 0
    critic_updates = ck.get('critic_updates', updates * cfg['critic_epochs'] * cfg['critic_minibatches']) if args.resume else 0
    best = ck.get('best') if args.resume else None
    if args.resume:
        critic.load_state_dict(ck['critic'])
        for opt, state in zip(optim, ck['optimizers']): opt.load_state_dict(state)
        torch.set_rng_state(ck['rng_torch']); torch.cuda.set_rng_state_all(ck['rng_cuda'])
        np.random.set_state(ck['rng_numpy']); random.setstate(ck['rng_python'])
    sampler = lambda n: cur.sample(n, task.weights['aim'])
    env = FullbodySim(task, cfg['num_envs'], sampler)
    buffer = FullbodyReplay(cfg['buffer_steps'], env.N, env.device, extra=REPLAY_FIELDS, robot=task.robot)
    history = env.observe()[:, None].repeat(1, 10, 1)
    noise = task.alpha.new_full((29,), cfg['body_noise_rad']); noise[task.arm_ids] = cfg['arm_noise_rad']
    reward_stats = {}
    action_stats = {}
    def noise_fraction():
        return exploration.fraction() if exploration else exploration_fraction(updates, cfg)

    @torch.no_grad()
    def collect(n):
        nonlocal history, reward_stats, action_stats
        sums = {}
        clipped = torch.zeros_like(task.alpha)
        excess_rad = torch.zeros_like(task.alpha)
        fraction = noise_fraction()
        for _ in range(n):
            o = env.observe()
            history = torch.where((env.ep_step == 0)[:, None, None], o[:, None].expand_as(history), history)
            raw = pi(history, task.command(env.state(), env.context))
            scales = noise * fraction
            if exploration:
                # Preserve standing with small body noise; the right arm follows stage progress.
                scales = task.alpha.new_full((29,), cfg['body_noise_rad'])
                scales[task.arm_ids] = cfg['arm_noise_rad'] * fraction
            raw += torch.randn_like(raw) * scales / task.alpha
            tr = env.step(raw); buffer.add(tr)
            excess = raw - tr['a']
            clipped += (excess.abs() > 1e-5).float().mean(0) / n
            excess_rad += (excess * task.alpha).abs().mean(0) / n
            if not torch.stack([torch.isfinite(tr[k]).all() for k in ('s', 's1', 'a', 'r')]).all():
                raise FloatingPointError('Nonfinite physical transition; refusing optimizer update')
            for prefix, terms in (('raw/', tr['reward_terms']), ('weighted_dt/', tr['reward_contributions'])):
                for key, value in terms.items(): sums[prefix + key] = sums.get(prefix + key, 0.) + value.mean() / n
            history = torch.cat((history[:, 1:], env.observe()[:, None]), 1)
        reward_stats = sums
        action_stats = dict(clipped_fraction=clipped.tolist(), request_excess_rad=excess_rad.tolist(),
                            arm_noise_rad=cfg['arm_noise_rad'] * fraction,
                            body_noise_rad=cfg['body_noise_rad'] * (1. if exploration else fraction))
    identity = code_identity()
    json_save(out / 'config.json', cfg)
    json_save(out / 'code_identity.json', identity)
    json_save(out / 'runtime.json', dict(checkpoint=str(Path(args.resume or args.init).resolve()), smoke=args.smoke,
                                       replay_restart=True, physics='nominal', source=ck['migration'],
                                       requested_envs=cfg['num_envs'], wm_batch=cfg['wm_batch'], actor_batch=cfg['pi_batch']))
    print(f"envs={env.N} horizon={cfg['horizon']} batch={cfg['wm_batch']}/{cfg['pi_batch']} "
          f"warmup={cfg['wm_only_iters']}+{cfg['critic_warmup_iters']} budget={cfg['iterations'] or 'unlimited'}", flush=True)
    collect(cfg['initial_collect_steps'])
    stop = start + 5 if args.smoke else cfg['iterations']
    model_ready = False
    began = time.monotonic()
    previous_phase = None
    for it in range(start, stop) if stop else count(start):
        phase = training_phase(it, start, cfg, critic_warmup_done, resume=bool(args.resume))
        collect(cfg['collect_steps'])
        for _ in range(cfg['wm_batches']):
            optim[1].zero_grad()
            loss = wm_loss(wm, task, buffer.sample_windows(cfg['wm_batch'], cfg['wm_window']))
            loss.backward(); wm_grad = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1., error_if_nonfinite=True)
            optim[1].step()
        check_model = (it + 1) % cfg['wm_validation_every'] == 0 or it + 1 == stop
        if check_model:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(cfg['seed'] + 710)
                validation = validate_world_model(pi, wm, task, sampler, cfg, out / 'wm_validation' / f'{it + 1:06d}')
            model_ready = validation['passed']
        stats = {}
        if phase != 'wm':
            stats = update_actor(pi, wm, critic, task, buffer, optim[0], optim[2], cfg, train_actor=phase == 'joint')
            updates += int(phase == 'joint')
            if exploration and phase == 'joint':
                exploration.step()
            critic_warmup_done += int(phase == 'critic')
            critic_updates += cfg['critic_epochs'] * cfg['critic_minibatches']
        if (it + 1) % cfg['log_every'] == 0 or phase != previous_phase or it + 1 == stop:
            rewards = dict(zip(reward_stats, torch.stack(list(reward_stats.values())).cpu().tolist()))
            row = dict(iteration=it + 1, phase=phase, actor_updates=updates, critic_updates=critic_updates,
                       critic_warmup_done=critic_warmup_done, wm_loss=float(loss.detach()), wm_grad=float(wm_grad),
                       model_ready=model_ready, stage=STAGES[cur.level]['name'], elapsed_s=time.monotonic() - began,
                       exploration_fraction=noise_fraction(),
                       exploration=exploration.state_dict() if exploration else None,
                       action_diagnostics=action_stats, reward=rewards, **stats)
            with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False) + '\n')
            json_save(out / 'status.json', row)
            print(f"iter={it + 1} phase={phase} stage={row['stage']} wm={row['wm_loss']:.5f} "
                  f"actor_updates={updates} model_ok={model_ready}", flush=True)
        previous_phase = phase
        if (it + 1) % cfg['ckpt_every'] == 0 or it + 1 == stop:
            report = None
            if not args.smoke:
                from tools.evaluate_aiming_fullbody import evaluate
                with torch.random.fork_rng(devices=[0]):
                    report = evaluate(pi, task, out / 'evaluation' / f'{it + 1:06d}', cur.level)
                cur.settle(report)
                if exploration:
                    exploration.settle(report, cur.level)
            saved = dict(task=TASK_NAME, stage='training', smoke=args.smoke, pi=pi.state_dict(), wm=wm.state_dict(),
                         critic=critic.state_dict(), optimizers=[o.state_dict() for o in optim], iteration=it + 1,
                         actor_updates=updates, critic_updates=critic_updates, critic_warmup_done=critic_warmup_done,
                         config=cfg, task_config=task.metadata(), robot_metadata=task.robot.metadata(),
                         migration=ck['migration'], curriculum=cur.state_dict(), best=best, code_identity=identity,
                         rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                         rng_numpy=np.random.get_state(), rng_python=random.getstate(), model_ready=model_ready)
            if exploration:
                saved['exploration'] = exploration.state_dict()
            if report:
                key = [report['level'], sum(r['passed'] for r in report['scenarios']) / len(report['scenarios']),
                       -sum(r['root_max_m'] + max(r['feet_max_m']) for r in report['scenarios']) / len(report['scenarios'])]
                if best is None or tuple(key) > tuple(best):
                    best = key; saved['best'] = best; save_checkpoint(out / 'best.pt', saved)
            save_checkpoint(out / 'last.pt', saved)
            save_checkpoint(out / f'ckpt_{it + 1:06d}.pt', saved)
            if report and report['level'] == len(STAGES) - 1 and report['passed'] and report['level'] in cur.mastered:
                subprocess.run([sys.executable, '-m', 'tools.evaluate_aiming_fullbody', '--ckpt', str(out / 'last.pt'),
                                '--out', str(out / 'evaluation' / 'independent_reload'), '--stage', 'C3b'], check=True)
                confirmed = json.loads((out / 'evaluation' / 'independent_reload.json').read_text())
                if confirmed['passed']:
                    subprocess.run([sys.executable, '-m', 'tools.render_aiming_fullbody',
                                    str(out / 'evaluation' / 'independent_reload.npz'),
                                    str(out / 'evaluation' / 'independent_reload.mp4'), '--scenario', '13'], check=True)
                    subprocess.run([sys.executable, '-m', 'tools.export_aiming_fullbody', '--ckpt', str(out / 'last.pt'),
                                    '--out', str(out / f'export_{it + 1:06d}')], check=True)
                    json_save(out / 'completion.json', dict(passed=True, iteration=it + 1, independent_reload=True))
                    return
    json_save(out / 'completion.json', dict(passed=False, wiring_complete=bool(args.smoke and updates > 0),
                                            actor_updates=updates, model_ready=model_ready,
                                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                                            reason='smoke wiring only' if args.smoke else 'iteration budget reached'))


if __name__ == '__main__':
    main()
