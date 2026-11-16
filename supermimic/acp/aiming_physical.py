"""Physical-return supervision and a single, versioned GPU sampling worker.

Only CPU artifacts cross the process boundary. Policies stay fixed for an entire
request; deterministic tails after exploratory prefixes label the current policy.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .aiming_fullbody import FullbodyPolicy, FullbodyTask, flat_boxes
from .aiming_fullbody_curriculum import Curriculum, fixed_parameters
from .aiming_fullbody_env import FullbodySim
from .aiming_fullbody_io import json_save, save_checkpoint
from .mdp import flatten_state, normalize_xi, unflatten_state
from .standing import freeze_normalizers
from .aiming_runtime import memory_snapshot


BACKEND = 'physical_mc_dual_v1'


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(v, device) for v in value)
    return value


def execution_setup(cfg):
    from . import networks
    networks.AMP_DTYPE = torch.bfloat16 if cfg.get('amp') == 'bf16' else None
    allow_tf32 = cfg.get('allow_tf32', True)
    torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    return execution_precision()


def execution_precision():
    from . import networks
    return dict(network_amp_dtype=str(networks.AMP_DTYPE) if networks.AMP_DTYPE is not None else None,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                default_dtype=str(torch.get_default_dtype()))


class RolloutStore:
    """Reusable GPU storage; expose valid rows by index instead of copying a full episode."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffers = None
        self.size = 0

    def reset(self):
        self.size = 0

    def append(self, sample):
        def allocate(value):
            if isinstance(value, dict):
                return {k: allocate(v) for k, v in value.items()}
            return value.new_empty((self.capacity, *value.shape))
        def write(target, value):
            if isinstance(value, dict):
                for k, v in value.items():
                    write(target[k], v)
            else:
                target[self.size].copy_(value)
        if self.size >= self.capacity:
            raise ValueError('Physical rollout storage overflow')
        if self.buffers is None:
            self.buffers = allocate(sample)
        write(self.buffers, sample)
        self.size += 1

    def view(self):
        def flatten(value):
            if isinstance(value, dict):
                return {k: flatten(v) for k, v in value.items()}
            return value[:self.size].flatten(0, 1)
        return flatten(self.buffers)


def actor_batches(data, cfg, count):
    """Draw independent, prebatched actor starts from valid training environments only."""
    valid = data.get('valid', torch.ones_like(data['env_id'], dtype=torch.bool))
    train = (valid & (data['env_id'] < cfg['critic_train_envs'])).nonzero().flatten()
    if not len(train):
        raise ValueError('No valid actor training starts')
    starts = {k: data[k] for k in ('s', 'history', 'ap', 'xi', 't', 'ctx')}
    return [cpu_tree(take(starts, train[torch.randint(len(train), (cfg['pi_batch'],), device=train.device)]))
            for _ in range(count)]


def check_baseline(cache, payload):
    if cache is None:
        raise ValueError('No paired baseline exists')
    expected = dict(job_id=payload['baseline_job_id'], digest=payload['baseline_digest'],
                    version=payload['version'], level=payload['level'], seed=payload['seed'])
    if any(cache[k] != v for k, v in expected.items()):
        raise ValueError('Stale paired baseline')


def policy_digest(state):
    h = hashlib.sha256()
    for name, x in sorted(state.items()):
        h.update(name.encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def physical_returns(rewards, done, gamma):
    """Include terminal reward, and never bootstrap a true terminal or task end."""
    out = torch.empty_like(rewards)
    tail = torch.zeros_like(rewards[-1])
    for j in reversed(range(len(rewards))):
        tail = rewards[j] + gamma * (~done[j]) * tail
        out[j] = tail
    return out


def take(data, ids):
    return {k: ({n: x[ids] for n, x in v.items()} if isinstance(v, dict) else v[ids])
            for k, v in data.items()}


def initialize_standing_policy(pi, task):
    """Random network, with the last bias centered on the physical reset pose."""
    with torch.no_grad():
        pi.dec[-1].bias.copy_((task.initial_q - task.mdp.default_q) / task.alpha)


def snapshot_actor(pi, optimizer):
    return cpu_tree(pi.state_dict()), cpu_tree(optimizer.state_dict())


def restore_actor(pi, optimizer, snapshot):
    pi.load_state_dict(snapshot[0])
    optimizer.load_state_dict(snapshot[1])
    optimizer.zero_grad(set_to_none=True)


def action_noise(task, cfg, mastered=False):
    body = cfg['body_noise_after_standing_rad'] if mastered else cfg['body_noise_rad']
    scale = torch.full_like(task.alpha, body)
    scale[task.arm_ids] = cfg['arm_noise_rad']
    return scale


@torch.no_grad()
def rollout(pi, task, env, cfg, *, seed, steps=1500, keep_data=False, prefixes=False, storage=None):
    """One first episode per slot. Failed slots never contribute after a reset."""
    torch.manual_seed(seed)
    dev, n = task.alpha.device, env.N
    env.reset(torch.arange(n, device=dev))
    hist = env.observe()[:, None].repeat(1, 10, 1)
    alive = torch.ones(n, device=dev, dtype=torch.bool)
    falls = torch.zeros_like(alive)
    finite_all = torch.ones_like(alive)
    length = torch.zeros(n, device=dev, dtype=torch.long)
    root = torch.zeros(n, device=dev)
    feet, tilt, hits, angle_sum = (torch.zeros_like(root) for _ in range(4))
    rewards, dones, samples, features, masks, times = [], [], [], [], [], []
    if storage is not None:
        storage.reset()
    prefix = torch.randint(0, cfg['critic_prefix_steps'] + 1, (n,), device=dev) if prefixes else torch.zeros(n, device=dev, dtype=torch.long)
    # Always include nominal, no-prefix episodes in both train and validation groups.
    prefix[::4] = 0
    scales = action_noise(task, cfg)
    began = time.monotonic()

    def record(active):
        S = env.state()
        finite = torch.stack([x.flatten(1).isfinite().all(-1) for x in S]).all(0)
        finite_all.logical_and_(~active | finite)
        if not finite[active].all():
            raise FloatingPointError('Nonfinite physical state')
        pos = task.kin.forward(*S)[0][:, task.feet]
        r = (S[0][:, :2] - env.context['root_anchor_w'][:, :2]).norm(dim=-1)
        f = (pos[..., :2] - env.context['feet_anchor_w'][..., :2]).norm(dim=-1).amax(-1)
        t = torch.rad2deg(torch.acos(S[1][:, 2, 2].clamp(-1, 1)))
        root.copy_(torch.maximum(root, torch.where(active, r, 0.)))
        feet.copy_(torch.maximum(feet, torch.where(active, f, 0.)))
        tilt.copy_(torch.maximum(tilt, torch.where(active, t, 0.)))
        return S

    record(alive)  # The initial frame is part of the acceptance interval.
    for j in range(steps):
        if keep_data and j % cfg['critic_stride'] == 0:
            S = env.state()
            sample = dict(s=flatten_state(S), history=hist, ap=env.a_prev, xi=env.xi,
                          t=env.ep_step, ctx=env.context)
            x = task.critic_features(S, hist, env.a_prev, env.xi, env.ep_step, env.context)
            valid = alive & (j >= prefix)
            if storage is not None:
                storage.append(dict(sample, x=x, valid=valid))
            else:
                samples.append({k: ({name: v.clone() for name, v in value.items()}
                                    if isinstance(value, dict) else value.clone())
                                for k, value in sample.items()})
                features.append(x)
                masks.append(valid)
            times.append(j)
        raw = pi(hist, task.command(env.state(), env.context))
        if prefixes:
            raw += torch.randn_like(raw) * scales / task.alpha * (j < prefix)[:, None]
        tr = env.step(raw)
        if not tr['r'][alive].isfinite().all():
            raise FloatingPointError('Nonfinite physical reward')
        record(alive)
        error, angle, valid, _ = task.metrics(env.state(), env.context)
        hits += (alive & valid & (error <= env.context['aim_tolerance'][:, 0])).float()
        angle_sum += torch.where(alive, angle, 0.)
        length += alive.long()
        falls |= alive & tr['terminated']
        rewards.append(torch.where(alive, tr['r'], 0.))
        dones.append(tr['done'] | ~alive)
        alive &= ~tr['done']
        hist = torch.cat((hist[:, 1:], env.observe()[:, None]), 1)
        ended = tr['done'].nonzero().flatten()
        if len(ended):
            env.reset(ended)
            hist[ended] = env.observe()[ended, None].expand(-1, 10, -1)
        if not alive.any():
            break
    if keep_data and alive.any():
        raise ValueError('A truncated rollout cannot supply complete-return labels')
    R, D = torch.stack(rewards), torch.stack(dones)
    G = physical_returns(R, D, cfg['gamma'])
    passed = finite_all & ~falls & (length == task.episode_steps) & (root <= .10) & (feet <= .05) & (tilt <= 15.)
    values = dict(root_max_m=root, feet_max_m=feet, tilt_max_deg=tilt,
                  survival_s=length.float() * task.dt, return_=G[0],
                  undiscounted_return=R.sum(0), fell=falls.float(), standing_pass=passed.float(),
                  hit_rate=hits / length.clamp_min(1), angle_deg=angle_sum / length.clamp_min(1))
    summary = {k: float(v.mean()) for k, v in values.items()}
    summary.update(per_environment={k: v.cpu().tolist() for k, v in values.items()},
                   finite=bool(finite_all.all()), complete=not bool(alive.any()),
                   requested_steps=steps, control_steps=len(R), environments=n,
                   simulated_transitions=len(R) * n, valid_transitions=int(length.sum()),
                   wall_seconds=time.monotonic() - began)
    if not keep_data:
        return None, summary
    if storage is not None:
        data = storage.view()
        data.update(y=G[times].flatten(), env_id=torch.arange(n, device=dev).repeat(len(times)))
        if not data['valid'].any():
            raise ValueError('No valid deterministic-tail samples')
        return data, summary
    data = {k: ({name: torch.stack([row[k][name] for row in samples]).flatten(0, 1) for name in samples[0][k]}
                if k == 'ctx' else torch.stack([row[k] for row in samples]).flatten(0, 1)) for k in samples[0]}
    data.update(x=torch.stack(features).flatten(0, 1), y=G[times].flatten(),
                env_id=torch.arange(n, device=dev).repeat(len(times)))
    ids = torch.stack(masks).flatten().nonzero().flatten()
    if not len(ids):
        raise ValueError('No valid deterministic-tail samples')
    return cpu_tree(take(data, ids)), summary


def compare_candidate(base, candidate, *, protected=False, anchor=None, short=False):
    """Stage-aware engineering guard; does not change formal physical acceptance."""
    reasons = []
    if not candidate['finite'] or (not short and not candidate['complete']):
        return dict(accepted=False, reasons=['nonfinite_or_incomplete'], tolerances={})
    tolerances = {}
    def worse(key, floor, higher_bad=True, reference=base):
        delta = np.asarray(candidate['per_environment'][key]) - np.asarray(reference['per_environment'][key])
        # Report a paired variability heuristic, not a statistical guarantee.
        allowance = max(floor, 2 * float(delta.std(ddof=1)) / np.sqrt(len(delta))) if len(delta) > 1 else floor
        tolerances[key] = max(tolerances.get(key, 0.), allowance)
        return float(delta.mean()) * (1 if higher_bad else -1) > allowance
    if protected:
        for reference in ([base, anchor] if anchor is not None else [base]):
            for key, floor, bad in [('fell', 1 / 64, True), ('root_max_m', .005, True),
                                    ('feet_max_m', .0025, True), ('tilt_max_deg', .5, True)]:
                if worse(key, floor, bad, reference):
                    reasons.append(key)
            if not short and worse('standing_pass', 2 / 64, False, reference):
                reasons.append('standing_pass')
        if worse('return_', .05, False):
            reasons.append('return')
    else:
        if worse('survival_s', .1, False):
            reasons.append('survival')
        survival_gain = candidate['survival_s'] - base['survival_s'] > tolerances.get('survival_s', .1)
        # A longer surviving episode can have larger max drift: don't reject that progress.
        if not survival_gain and worse('undiscounted_return', .1, False):
            reasons.append('physical_return')
    return dict(accepted=not reasons, reasons=sorted(set(reasons)), tolerances=tolerances,
                protected=protected, short=short)


def actor_step(pi, wm, critic, task, data, optimizer, cfg, *, prebatched=False):
    from torch.utils.checkpoint import checkpoint
    from .train_aiming_fullbody import bound_bootstrap
    b = data if prebatched else take(data, torch.randint(len(data['s']), (cfg['pi_batch'],), device=task.alpha.device))
    S, history, ap, t = unflatten_state(b['s']), b['history'], b['ap'], b['t']
    alive = torch.ones_like(t, dtype=torch.float32)
    objective = torch.zeros_like(alive)
    wm.requires_grad_(False); critic.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    step = getattr(pi, '_physical_imagined_step', None)
    if step is None:
        def step(S, history, ap, t, xi, ctx):
            raw = pi(history, task.command(S, ctx))
            action, _ = task.compose_action(raw, xi)
            next_state = wm.step(S, action, normalize_xi(xi), task.mdp.scan(S, flat_boxes(S)))
            reward = task.reward(next_state, action, ap, ctx, raw=raw, xi=xi)
            done = task.soft_termination(next_state, t + 1, ctx)
            h = torch.cat((history[:, 1:], task.mdp.obs(next_state, action, xi)[:, None]), 1)
            return next_state, h, action, reward, done
        if cfg.get('compile_training'):
            step = torch.compile(step, dynamic=False)
        pi._physical_imagined_step = step
    try:
        for j in range(cfg['horizon']):
            if cfg.get('actor_recompute', True):
                S, history, ap, r, done = checkpoint(step, S, history, ap, t, b['xi'], b['ctx'], use_reentrant=False)
            else:
                S, history, ap, r, done = step(S, history, ap, t, b['xi'], b['ctx'])
            objective += cfg['gamma'] ** j * alive * r
            alive = alive * (1 - done); t = t + 1
        value = bound_bootstrap(critic(task.critic_features(S, history, ap, b['xi'], t, b['ctx'])), t, task, cfg)
        loss = -(objective + cfg['gamma'] ** cfg['horizon'] * alive * value).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(pi.parameters(), 1., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_value_(pi.parameters(), cfg['actor_grad_clip_value'])
        optimizer.step()
        return dict(loss=float(loss.detach()), gradient_norm=float(norm))
    finally:
        wm.requires_grad_(True); critic.requires_grad_(True)


def fit_critic(critic, optimizer, data, cfg, *, first=False):
    from .train_aiming_fullbody import bound_bootstrap
    mask = data.get('valid', torch.ones_like(data['env_id'], dtype=torch.bool))
    train = (mask & (data['env_id'] < cfg['critic_train_envs'])).nonzero().flatten()
    valid = (mask & (data['env_id'] >= cfg['critic_train_envs'])).nonzero().flatten()
    if not len(train) or not len(valid):
        raise ValueError('Both training and validation need deterministic-tail samples')
    if first:
        critic.norm.max_count = 2e8
        for ids in train.split(4096):
            critic.norm.update(data['x'][ids])
        critic.norm.var.clamp_(min=.05 ** 2)
    freeze_normalizers(critic)
    @torch.no_grad()
    def measure():
        predictions = []
        for ids in valid.split(1024):
            x = data['x'][ids]
            if cfg.get('compile_training') and len(ids) < 1024:
                x = torch.cat((x, x[:1].expand(1024 - len(ids), -1)))
            predictions.append(bound_bootstrap(critic(x)[:len(ids)], data['t'][ids], NoneTask, cfg))
        return float((torch.cat(predictions) - data['y'][valid]).abs().mean())
    # Only the finite-horizon upper-bound attributes are needed by the calibration metric.
    class NoneTask:
        episode_steps = 1500
        max_step_reward = cfg['max_step_reward']
    before = measure()
    best = before
    saved = cpu_tree(critic.state_dict()), cpu_tree(optimizer.state_dict())
    limit = cfg['critic_initial_updates'] if first else cfg['critic_refresh_updates']
    stale = 0; curve = []; actual = 0
    critic.requires_grad_(True)
    for update in range(1, limit + 1):
        ids = train[torch.randint(len(train), (cfg['critic_batch'],), device=train.device)]
        optimizer.zero_grad(set_to_none=True)
        loss = (critic(data['x'][ids]) - data['y'][ids]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); actual = update
        if update % cfg['critic_check_every'] == 0 or update == limit:
            mae = measure()
            curve.append(dict(update=update, validation_mae=mae, train_loss=float(loss.detach())))
            if np.isfinite(mae) and mae < best:
                significant = mae < best - max(1e-4, abs(best) * .001)
                best = mae; saved = cpu_tree(critic.state_dict()), cpu_tree(optimizer.state_dict())
                stale = 0 if significant else stale + 1
            else:
                stale += 1
            if stale >= cfg['critic_patience']:
                break
    critic.load_state_dict(saved[0]); optimizer.load_state_dict(saved[1])
    improved = np.isfinite(best) and (best < before - 1e-6 if first else best <= before)
    return dict(before_mae=before, best_mae=best, updates=actual, train_samples=len(train),
                validation_samples=len(valid), improved=bool(improved), curve=curve)


def _worker_main(connection, cfg):
    torch.set_num_threads(cfg['worker_threads'])
    torch.cuda.set_device(cfg['sampler_device'])
    execution_setup(cfg)
    import warp as wp
    wp.config.quiet = True
    task = FullbodyTask(cfg['sampler_device'], cfg)
    pi = FullbodyPolicy(task.robot).to(task.alpha.device).eval()
    critic = optimizer = None
    storage = RolloutStore((task.episode_steps + cfg['critic_stride'] - 1) // cfg['critic_stride'])
    baseline = None
    envs = {}
    official_env = validation_env = None
    env_creations = 0
    context = {'level': 0, 'kind': 'collect'}
    def sampler(n):
        if context['kind'] in ('collect', 'collect_fit'):
            c = Curriculum(task.alpha.device, dict(level=context['level'], consecutive=0, mastered=[]))
            return c.sample(n, task.weights['aim'])
        params = fixed_parameters(context['level'], task.alpha.device)
        return params[torch.arange(n, device=task.alpha.device) % len(params)], (0. if context['level'] == 0 else task.weights['aim'])
    try:
        connection.send({'ready': True, 'device': str(task.alpha.device), 'precision': execution_precision()})
        while True:
            request = connection.recv()
            if request is None:
                break
            directory = Path(request['directory'])
            try:
                payload = torch.load(directory / 'request.pt', map_location='cpu', weights_only=False)
                if policy_digest(payload['pi']) != payload['digest']:
                    raise ValueError('Policy payload digest mismatch')
                pi.load_state_dict(payload['pi'])
                context.update(level=payload['level'], kind=payload['kind'])
                kind = payload['kind']
                result = dict(job_id=request['job_id'], version=payload['version'], digest=payload['digest'],
                              level=payload['level'], device=str(task.alpha.device))
                if kind in ('collect', 'collect_fit', 'pair', 'baseline', 'candidate'):
                    n = cfg['critic_envs'] if kind in ('collect', 'collect_fit') else cfg['gate_envs']
                    if n not in envs:
                        envs[n] = FullbodySim(task, n, sampler, eval_mode=True)
                        env_creations += 1
                    env = envs[n]
                    if kind in ('collect', 'collect_fit'):
                        data, summary = rollout(pi, task, env, cfg, seed=payload['seed'], keep_data=True,
                                                prefixes=True, storage=storage if kind == 'collect_fit' else None)
                        result['physical'] = summary
                        if kind == 'collect_fit':
                            from .networks import Critic
                            if critic is None:
                                critic = Critic(task.critic_dim).to(task.alpha.device)
                                optimizer = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'],
                                                             fused=cfg.get('fused_adam', False))
                                if cfg.get('compile_training'):
                                    critic.compile(dynamic=False)
                            critic.load_state_dict(payload['critic'])
                            optimizer.load_state_dict(payload['critic_optimizer'])
                            calibration = fit_critic(critic, optimizer, data, cfg, first=payload['first'])
                            result.update(calibration=calibration, critic=cpu_tree(critic.state_dict()),
                                          critic_optimizer=cpu_tree(optimizer.state_dict()),
                                          actor_batches=actor_batches(data, cfg, payload['actor_count'])
                                              if calibration['improved'] else [])
                            del data
                        else:
                            result['data'] = data
                    elif kind == 'baseline':
                        _, base = rollout(pi, task, env, cfg, seed=payload['seed'])
                        baseline = dict(job_id=request['job_id'], digest=payload['digest'], version=payload['version'],
                                        level=payload['level'], seed=payload['seed'], base=base)
                        result['baseline'] = base
                    elif kind == 'candidate':
                        check_baseline(baseline, payload)
                        _, candidate = rollout(pi, task, env, cfg, seed=payload['seed'])
                        result.update(base=baseline['base'], candidate=candidate,
                                      decision=compare_candidate(baseline['base'], candidate,
                                          protected=payload['protected'], anchor=payload.get('anchor')))
                        baseline = None
                    else:
                        _, base = rollout(pi, task, env, cfg, seed=payload['seed'])
                        pi.load_state_dict(payload['candidate'])
                        _, candidate = rollout(pi, task, env, cfg, seed=payload['seed'])
                        result.update(base=base, candidate=candidate,
                                      decision=compare_candidate(base, candidate, protected=payload['protected'], anchor=payload.get('anchor')))
                elif kind == 'official':
                    from tools.evaluate_aiming_fullbody import evaluate
                    n = len(fixed_parameters(payload['level'], task.alpha.device))
                    if official_env is not None and official_env.N != n:
                        official_env = None
                        gc.collect()
                    if official_env is None:
                        official_env = FullbodySim(task, n, sampler, eval_mode=True)
                        env_creations += 1
                    if payload.get('checkpoint'):
                        ck = torch.load(payload['checkpoint'], map_location='cpu', weights_only=False)
                        from .aiming_fullbody import validate_checkpoint
                        validate_checkpoint(ck, task)
                        if policy_digest(ck['pi']) != payload['digest']:
                            raise ValueError('Reloaded checkpoint does not match requested policy')
                        pi.load_state_dict(ck['pi'])
                        del ck
                    torch.manual_seed(payload['seed'])
                    # Deployment export is FP32. Formal acceptance must use that same precision.
                    from . import networks
                    networks.AMP_DTYPE = None
                    try:
                        result['report'] = evaluate(pi, task, directory / 'official', payload['level'], env=official_env)
                        result['report']['policy_precision'] = 'float32'
                        result['report']['precision'] = execution_precision()
                    finally:
                        execution_setup(cfg)
                elif kind == 'validate_wm':
                    from .networks import WorldModel
                    from .aiming_fullbody_validation import validate_world_model
                    wm = WorldModel(task.mdp.scan_dim, robot=task.robot).to(task.alpha.device)
                    wm.load_state_dict(payload['wm']); freeze_normalizers(wm)
                    context['kind'] = 'collect'
                    torch.manual_seed(payload['seed'])
                    if validation_env is None:
                        validation_env = FullbodySim(task, cfg['wm_validation_envs'], sampler)
                        env_creations += 1
                    result['report'] = validate_world_model(pi, wm, task, sampler, cfg, directory / 'wm', env=validation_env)
                    del wm
                else:
                    raise ValueError(f'Unknown physical worker request: {kind}')
                if kind in ('official', 'validate_wm'):
                    gc.collect()
                result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                result['memory'] = memory_snapshot()
                result['simulator_cache'] = dict(creations=env_creations, sampling_counts=sorted(envs),
                                                official_count=official_env.N if official_env is not None else None,
                                                validation_count=validation_env.N if validation_env is not None else None)
                json_save(directory / 'summary.json', {k: v for k, v in result.items()
                          if k not in ('data', 'actor_batches', 'critic', 'critic_optimizer')})
                save_checkpoint(directory / 'result.pt', result)
                connection.send(dict(job_id=request['job_id'], ok=True))
                del result, payload
            except BaseException:
                error = traceback.format_exc()
                json_save(directory / 'error.json', dict(error=error))
                connection.send(dict(job_id=request['job_id'], ok=False, error=error))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class PhysicalWorker:
    def __init__(self, cfg, out):
        self.cfg, self.out, self.pending = cfg, Path(out), None
        self.sequence = time.time_ns()
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker_main, args=(child, cfg), daemon=True)
        self.process.start(); child.close()
        deadline = time.monotonic() + cfg['worker_timeout_s']
        while not self.connection.poll(.1):
            if not self.process.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError('Physical worker failed to initialize')
        self.runtime = self.connection.recv()

    def submit(self, kind, pi, *, version, level, seed, **extra):
        if self.pending is not None:
            raise RuntimeError('Only one physical request may be in flight')
        self.sequence += 1
        directory = self.out / f'{self.sequence}_{kind}'
        directory.mkdir(parents=True, exist_ok=False)
        state = cpu_tree(pi.state_dict()) if hasattr(pi, 'state_dict') else cpu_tree(pi)
        payload = dict(kind=kind, pi=state, version=version, level=level, seed=seed,
                       digest=policy_digest(state), **cpu_tree(extra))
        save_checkpoint(directory / 'request.pt', payload)
        self.pending = dict(job_id=self.sequence, directory=str(directory), version=version,
                            level=level, digest=payload['digest'], began=time.monotonic(), kind=kind)
        self.connection.send(dict(job_id=self.sequence, directory=str(directory)))

    def ready(self):
        if self.connection.poll():
            return True
        if not self.process.is_alive():
            raise RuntimeError('Physical worker exited; refusing further policy updates')
        if self.pending and time.monotonic() - self.pending['began'] > self.cfg['worker_timeout_s']:
            raise TimeoutError('Physical worker request timed out')
        return False

    def receive(self):
        if not self.ready():
            return None
        notification = self.connection.recv()
        if notification['job_id'] != self.pending['job_id']:
            raise ValueError('Mismatched worker notification')
        if not notification['ok']:
            raise RuntimeError(notification['error'])
        directory = Path(self.pending['directory'])
        result = torch.load(directory / 'result.pt', map_location='cpu', weights_only=False)
        for key in ('job_id', 'version', 'level', 'digest'):
            if result[key] != self.pending[key]:
                raise ValueError(f'Stale physical result: {key}')
        result['request_wall_seconds'] = time.monotonic() - self.pending['began']
        # Small JSON evidence stays; large IPC payloads have no continuing owner.
        (directory / 'request.pt').unlink(); (directory / 'result.pt').unlink()
        self.pending = None
        return result

    def close(self):
        try:
            if self.process.is_alive() and self.pending is None:
                self.connection.send(None)
                self.process.join(5)
            if self.process.is_alive():
                self.process.terminate(); self.process.join(5)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self.connection.close()
