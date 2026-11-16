"""Fullbody learning with H8/H16 imagination and a dedicated physical-return GPU.

Example: python -m supermimic.acp.train_aiming_fullbody --from-scratch
    --config docs/config/standing_aiming_fullbody_v2/h8_physical_dual_scratch.json
    --log-dir logs/tienkung3_29dof/standing_aiming_fullbody_v2/<new-run>
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .aiming_fullbody import FullbodyPolicy, FullbodyReplay, FullbodyTask, REPLAY_FIELDS, TASK_NAME, flat_boxes, validate_checkpoint
from .aiming_fullbody_curriculum import Curriculum, StageExploration, STAGES
from .aiming_fullbody_env import FullbodySim
from .aiming_fullbody_io import code_identity, json_save, save_checkpoint
from .aiming_physical import (BACKEND, PhysicalWorker, action_noise, actor_step, cpu_tree, fit_critic,
                             execution_setup, execution_precision, initialize_standing_policy, policy_digest, restore_actor,
                             snapshot_actor, take, to_device)
from .mdp import normalize_xi, unflatten_state
from .networks import WorldModel, Critic
from .aiming_runtime import MemoryMonitor
from .standing import freeze_normalizers
from .train_aiming_fullbody import wm_loss


DEFAULT = 'docs/config/standing_aiming_fullbody_v2/h8_physical_dual_scratch.json'


def apply_precision_override(cfg, precision):
    """Change execution precision without replacing a resumed training configuration."""
    if precision == 'fp32':
        cfg.update(amp='none', allow_tf32=False)
    elif precision == 'bf16':
        cfg.update(amp='bf16', allow_tf32=True)
    elif precision is not None:
        raise ValueError(f'Unsupported precision: {precision}')


def official_due(state, cfg):
    if state.get('official_pending') or not state['initial_evaluated']:
        return True
    blocks = state['accepted_blocks']
    return (blocks > state.get('last_official_accepted_blocks', 0)
            and blocks % cfg['formal_eval_blocks'] == 0)


def validate_config(cfg):
    if cfg['task'] != TASK_NAME or cfg['version'] != 2 or cfg['training_backend'] != BACKEND:
        raise ValueError('Expected the physical-MC fullbody v2 configuration')
    if cfg['horizon'] not in (8, 16) or cfg['wm_window'] != 8 or cfg['hist'] != 10:
        raise ValueError('Require H=8 or H=16, WM window=8 and history=10')
    if cfg['learner_device'] == cfg['sampler_device']:
        raise ValueError('The learner and physical worker require distinct GPUs')
    for key in ('num_envs', 'wm_batch', 'pi_batch', 'buffer_steps', 'collect_steps', 'initial_collect_steps',
                'wm_batches', 'wm_only_iters', 'critic_envs', 'critic_batch', 'critic_initial_updates',
                'critic_refresh_updates', 'critic_check_every', 'critic_patience', 'actor_block_size',
                'critic_stride', 'gate_envs', 'worker_timeout_s', 'formal_eval_blocks'):
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if not 0 < cfg['critic_train_envs'] < cfg['critic_envs']:
        raise ValueError('Whole training and validation environment groups are required')
    if cfg['buffer_steps'] < 10 or cfg['initial_collect_steps'] < 10:
        raise ValueError('Replay must contain at least one real observation history')
    if not 0 < cfg['gamma'] < 1 or cfg['actor_lr'] <= 0 or cfg['critic_prefix_steps'] < 0:
        raise ValueError('Invalid return, optimizer or prefix configuration')
    if cfg.get('amp', 'none') not in ('none', 'bf16'):
        raise ValueError('Only FP32 or BF16 network execution is supported')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=DEFAULT)
    p.add_argument('--log-dir', '--log_dir', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--from-scratch', action='store_true')
    g.add_argument('--resume')
    p.add_argument('--physical-critic', action='store_true', help='Select this backend when resuming via the legacy entrypoint')
    p.add_argument('--horizon', type=int, choices=[8, 16],
                   help='Override actor imagination length; WM fitting window remains 8')
    p.add_argument('--iterations', type=int, help='Finite learner-round budget; 0 is unlimited')
    p.add_argument('--actor-budget', type=int, help='Total attempted actor updates; 0 is unlimited')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--capacity-check', action='store_true')
    p.add_argument('--precision', choices=['fp32', 'bf16'],
                   help='Override execution precision on a fresh or resumed run; fp32 also disables TF32')
    args = p.parse_args()
    ck = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if ck and (ck.get('training_backend') != BACKEND or ck.get('stage') != 'training'):
        raise ValueError('Resume requires a complete physical-MC training checkpoint')
    if ck and ck.get('smoke') and not args.smoke:
        raise ValueError('Test checkpoints cannot initialize formal training')
    cfg = copy.deepcopy(ck['config'] if ck else json.loads(Path(args.config).read_text()))
    apply_precision_override(cfg, args.precision)
    if args.horizon is not None:
        cfg['horizon'] = args.horizon
    configuration_history = copy.deepcopy(ck.get('configuration_history', []) if ck else [])
    if args.capacity_check and (not args.smoke or args.resume):
        raise ValueError('Capacity check requires a fresh smoke run')
    if args.smoke and not args.resume:
        cfg.update(wm_only_iters=2, critic_initial_updates=100, critic_refresh_updates=100,
                   critic_check_every=20, critic_patience=3, log_every=1, formal_eval_blocks=1,
                   warmup_validate_every=0, wm_validation_steps=64, wm_validation_envs=2,
                   actor_budget=10, iterations=0)
        if not args.capacity_check:
            cfg.update(num_envs=8, wm_batch=8, pi_batch=8, buffer_steps=64, initial_collect_steps=32,
                       wm_batches=1, critic_envs=8, critic_train_envs=6, critic_batch=32,
                       critic_initial_updates=40, critic_refresh_updates=20, gate_envs=8,
                       actor_block_size=2, actor_budget=4, critic_prefix_steps=16)
    for name in ('iterations', 'actor_budget'):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
        if cfg[name] < 0:
            raise ValueError(f'{name} must be nonnegative')
    validate_config(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Two visible CUDA GPUs are required')
    out = Path(args.log_dir).resolve()
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('A from-scratch run requires a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    run_lock = (out / '.trainer.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # One owner across checkouts and frozen runners, including finite tests.
    owner_path = Path.home() / '.local/state/standing-aiming/physical-trainer.lock'
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    owner_lock = owner_path.open('a')
    fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(cfg['learner_threads'])
    torch.cuda.set_device(cfg['learner_device'])
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    execution_setup(cfg)
    import warp as wp
    wp.config.quiet = True
    task = FullbodyTask(cfg['learner_device'], cfg)
    cfg['max_step_reward'] = task.max_step_reward
    pi = FullbodyPolicy(task.robot, freeze_norm=False).to(task.alpha.device)
    wm = WorldModel(task.mdp.scan_dim, robot=task.robot).to(task.alpha.device)
    critic = Critic(task.critic_dim).to(task.alpha.device)
    opt_options = dict(fused=cfg.get('fused_adam', False))
    op = torch.optim.Adam(pi.parameters(), lr=cfg['actor_lr'], betas=tuple(cfg['actor_betas']), **opt_options)
    ow = torch.optim.Adam(wm.parameters(), lr=cfg['wm_lr'], **opt_options)
    ov = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'], **opt_options)
    state = dict(iteration=0, wm_updates=0, critic_updates=0, actor_updates=0, attempted_actor_updates=0,
                 blocks=0, accepted_blocks=0, rejected_blocks=0, consecutive_rejections=0, version=0, collection_round=0,
                 physical_transitions=0, worker_transitions=0, valid_worker_transitions=0,
                 warmup_done=0, normalizers_ready=False, critic_ready=False, calibration_failures=0,
                 standing_protected=False, initial_evaluated=False, anchor=None, completed=False, last_wm_loss=None,
                 official_pending=False, last_official_accepted_blocks=0,
                 last_decision=None, last_calibration=None, best_key=None, wm_validation=None)
    cur = Curriculum(task.alpha.device)
    exploration = StageExploration(cfg['stage_exploration'], 0)
    if ck:
        validate_checkpoint(ck, task)
        pi.load_state_dict(ck['pi']); wm.load_state_dict(ck['wm']); critic.load_state_dict(ck['critic'])
        for opt, saved in zip((op, ow, ov), ck['optimizers']):
            opt.load_state_dict(saved)
        state.update(ck['physical_state'])
        cur = Curriculum(task.alpha.device, ck['curriculum'])
        exploration = StageExploration(cfg['stage_exploration'], cur.level, ck['exploration'])
        torch.set_rng_state(ck['rng_torch']); torch.cuda.set_rng_state(ck['rng_learner_cuda'], task.alpha.device)
        np.random.set_state(ck['rng_numpy']); random.setstate(ck['rng_python'])
        if state['normalizers_ready']:
            freeze_normalizers(pi, wm)
        if state['critic_ready']:
            freeze_normalizers(critic)
    else:
        initialize_standing_policy(pi, task)
    del ck
    identity = code_identity()
    began = time.monotonic()
    stop_requested = False
    def stop_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    worker = None; candidate_snapshot = None
    phase_context = {'phase': 'initializing'}
    def memory_context():
        return dict(phase_context, iteration=state['iteration'],
                    worker_pid=worker.process.pid if worker is not None else None,
                    worker_task=worker.pending.get('kind') if worker is not None and worker.pending else None)
    memory_monitor = MemoryMonitor(out, memory_context)
    memory_monitor.start()
    sampler = lambda n: cur.sample(n, task.weights['aim'])
    env = FullbodySim(task, cfg['num_envs'], sampler)
    buffer = FullbodyReplay(cfg['buffer_steps'], env.N, env.device, REPLAY_FIELDS, robot=task.robot)
    history = env.observe()[:, None].repeat(1, 10, 1)
    loss_function = lambda windows: wm_loss(wm, task, windows)
    if cfg.get('compile_training'):
        loss_function = torch.compile(loss_function, dynamic=False)

    def checkpoint(name='last.pt'):
        artifact = dict(task=TASK_NAME, stage='training', training_backend=BACKEND, smoke=bool(args.smoke),
                        pi=cpu_tree(pi.state_dict()), wm=cpu_tree(wm.state_dict()), critic=cpu_tree(critic.state_dict()),
                        optimizers=cpu_tree([op.state_dict(), ow.state_dict(), ov.state_dict()]),
                        config=cfg, task_config=task.metadata(), robot_metadata=task.robot.metadata(),
                        curriculum=cur.state_dict(), exploration=exploration.state_dict(),
                        physical_state=copy.deepcopy(state), iteration=state['iteration'], actor_updates=state['actor_updates'],
                        critic_updates=state['critic_updates'], migration=dict(kind='random_initialization', source=None,
                            actor_bias='physical_reset_pose', inherited_weights=False), code_identity=identity,
                        resume_checkpoint=str(Path(args.resume).resolve()) if args.resume else None,
                        configuration_history=configuration_history,
                        execution_precision=execution_precision(),
                        rng_torch=torch.get_rng_state(), rng_learner_cuda=torch.cuda.get_rng_state(task.alpha.device),
                        rng_numpy=np.random.get_state(), rng_python=random.getstate())
        save_checkpoint(out / name, artifact)

    def log(phase, **extra):
        phase_context['phase'] = phase
        row = dict(phase=phase, stage=STAGES[cur.level]['name'], elapsed_s=time.monotonic() - began,
                   learner_device=cfg['learner_device'], sampler_device=cfg['sampler_device'],
                   precision=execution_precision(),
                   actor_lr=op.param_groups[0]['lr'], **state, **extra)
        json_save(out / 'status.json', row)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(f"phase={phase} iter={state['iteration']} stage={row['stage']} "
              f"actor={state['actor_updates']}/{state['attempted_actor_updates']} "
              f"wm={state['last_wm_loss']} blocks={state['accepted_blocks']}/{state['rejected_blocks']}", flush=True)

    @torch.no_grad()
    def collect(n):
        nonlocal history
        scale = action_noise(task, cfg, 0 in cur.mastered)
        scale[task.arm_ids] *= exploration.fraction()
        for _ in range(n):
            o = env.observe()
            history = torch.where((env.ep_step == 0)[:, None, None], o[:, None].expand_as(history), history)
            raw = pi(history, task.command(env.state(), env.context))
            raw += torch.randn_like(raw) * scale / task.alpha
            tr = env.step(raw)
            if not all(torch.isfinite(tr[k]).all() for k in ('s', 's1', 'a', 'r')):
                raise FloatingPointError('Nonfinite replay transition')
            buffer.add(tr)
            history = torch.cat((history[:, 1:], env.observe()[:, None]), 1)
            state['physical_transitions'] += env.N

    @torch.no_grad()
    def initialize_norms():
        # Real data only; no validation episodes contribute normalization statistics.
        rows = {k: v[:buffer.size].flatten(0, 1) for k, v in buffer.data.items()}
        for ids in torch.arange(len(rows['s']), device=task.alpha.device).split(1024):
            S, S1 = unflatten_state(rows['s'][ids]), unflatten_state(rows['s1'][ids])
            ctx = {k: rows[k][ids] for k in env.context}
            pi.obs_norm.update(rows['o'][ids])
            pi.goal_norm.update(task.command(S, ctx)[:, :93])
            wm.in_norm.update(wm.inputs(S, rows['a'][ids], normalize_xi(rows['xi'][ids]), task.mdp.scan(S, flat_boxes(S))))
            wm.out_norm.update(wm.targets(S, S1))
        # Avoid amplifying previously constant channels when exploration broadens.
        pi.obs_norm.var.clamp_(min=.05 ** 2); pi.goal_norm.var.clamp_(min=.05 ** 2)
        wm.in_norm.var.clamp_(min=.05 ** 2); wm.out_norm.var.clamp_(min=.003 ** 2)
        freeze_normalizers(pi, wm)
        state['normalizers_ready'] = True

    def wm_round():
        collect(cfg['collect_steps'])
        for _ in range(cfg['wm_batches']):
            ow.zero_grad(set_to_none=True)
            loss = loss_function(buffer.sample_windows(cfg['wm_batch'], cfg['wm_window']))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1., error_if_nonfinite=True)
            ow.step(); state['wm_updates'] += 1
        state['iteration'] += 1
        state['last_wm_loss'] = float(loss.detach())

    def await_worker(wm_rounds=0):
        # A fixed data/update quota, independent of how quickly GPU 1 finishes.
        for _ in range(wm_rounds):
            if stop_requested:
                raise InterruptedError('Stop requested')
            wm_round()
            worker.ready()  # Check worker health, even while doing useful GPU 0 work.
        while not worker.ready():
            if stop_requested:
                raise InterruptedError('Stop requested')
            time.sleep(.05)
        result = worker.receive()
        for name in ('physical', 'base', 'candidate'):
            if name in result:
                state['worker_transitions'] += result[name]['simulated_transitions']
                state['valid_worker_transitions'] += result[name]['valid_transitions']
        return result

    def submit_collection(count):
        extra = {}
        kind = 'collect'
        if cfg.get('critic_on_worker'):
            kind = 'collect_fit'
            extra = dict(critic=critic.state_dict(), critic_optimizer=ov.state_dict(),
                         first=not state['critic_ready'], actor_count=count)
        worker.submit(kind, pi, version=state['version'], level=cur.level,
                      seed=cfg['seed'] + 10000 + state['collection_round'], **extra)
        state['collection_round'] += 1

    def official():
        nonlocal critic, ov, buffer, history
        old_level = cur.level
        state['official_pending'] = True
        checkpoint()
        log('official_evaluation_pending')
        worker.submit('official', pi, version=state['version'], level=cur.level, seed=cfg['seed'] + 90000 + cur.level)
        report = await_worker()['report']
        state['initial_evaluated'] = True
        state['official_pending'] = False
        state['last_official_accepted_blocks'] = state['accepted_blocks']
        state['worker_transitions'] += len(report['scenarios']) * 1500
        # Formal evaluation has its original complete-episode, all-scenario criteria.
        cur.settle(report)
        exploration.settle(report, cur.level)
        if report['passed'] and old_level == 0:
            state['standing_protected'] = True
        if cur.level != old_level:
            state['anchor'] = None; state['best_key'] = None
            state['critic_ready'] = False; state['calibration_failures'] = 0
            state['consecutive_rejections'] = 0
            critic = Critic(task.critic_dim).to(task.alpha.device)
            ov = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'], **opt_options)
            # Refresh all contexts at the curriculum boundary; no old-stage policy starts.
            env.reset(torch.arange(env.N, device=task.alpha.device))
            history = env.observe()[:, None].repeat(1, 10, 1)
            buffer = FullbodyReplay(cfg['buffer_steps'], env.N, env.device, REPLAY_FIELDS, robot=task.robot)
            collect(cfg['initial_collect_steps'])
        checkpoint()
        if report['passed'] and old_level == len(STAGES) - 1 and old_level in cur.mastered:
            worker.submit('official', pi, version=state['version'], level=old_level,
                          seed=cfg['seed'] + 99000, checkpoint=str(out / 'last.pt'))
            confirmed = await_worker()['report']
            state['worker_transitions'] += len(confirmed['scenarios']) * 1500
            state['completed'] = bool(confirmed['passed'])
            if state['completed']:
                checkpoint('best.pt')
        log('official_evaluation', formal_passed=report['passed'])

    json_save(out / 'config.json', cfg)
    json_save(out / 'code_identity.json', identity)
    json_save(out / 'runtime.json', dict(training_backend=BACKEND, from_scratch=not bool(args.resume), smoke=args.smoke,
                                       initialization='random networks; reset-pose actor output bias',
                                       learner_device=cfg['learner_device'], sampler_device=cfg['sampler_device'],
                                       num_envs=cfg['num_envs'], critic_envs=cfg['critic_envs'], gate_envs=cfg['gate_envs'],
                                       critic_training_device=cfg['sampler_device'] if cfg.get('critic_on_worker') else cfg['learner_device'],
                                       amp=cfg.get('amp', 'none'), compile_training=cfg.get('compile_training', False),
                                       precision=execution_precision(), precision_override=args.precision,
                                       resume_checkpoint=str(Path(args.resume).resolve()) if args.resume else None,
                                       actor_recompute=cfg.get('actor_recompute', True),
                                       overlap_baseline=cfg.get('overlap_baseline', False),
                                       horizon=cfg['horizon'], replay_restart=True, seed=cfg['seed'],
                                       configuration_history=configuration_history))
    reason = 'finite_budget'
    try:
        if not args.resume:
            checkpoint('initial.pt')
        log('initial_collection')
        collect(cfg['initial_collect_steps'])
        if not state['normalizers_ready']:
            initialize_norms()
        worker = PhysicalWorker(cfg, out / 'worker')
        submit_collection(cfg['actor_block_size'])
        log('wm_warmup', worker=worker.runtime)
        while state['warmup_done'] < cfg['wm_only_iters']:
            if stop_requested:
                raise InterruptedError('Stop requested')
            wm_round(); state['warmup_done'] += 1
            worker.ready()
            if state['warmup_done'] == 1 or state['warmup_done'] % cfg['log_every'] == 0 or state['warmup_done'] == cfg['wm_only_iters']:
                log('wm_warmup'); checkpoint()
        first_result = await_worker()
        if not args.smoke:
            worker.submit('validate_wm', pi, version=state['version'], level=cur.level,
                          seed=cfg['seed'] + 710, wm=wm.state_dict())
            state['wm_validation'] = await_worker()['report']
            log('wm_validation')
        if official_due(state, cfg):
            old_level = cur.level
            official()
            if cur.level != old_level:
                first_result = None
        while not stop_requested and not state['completed']:
            if cfg['actor_budget'] and state['attempted_actor_updates'] >= cfg['actor_budget']:
                break
            if cfg['iterations'] and state['iteration'] >= cfg['iterations']:
                break
            count = cfg['actor_block_size']
            if cfg['actor_budget']:
                count = min(count, cfg['actor_budget'] - state['attempted_actor_updates'])
            if first_result is not None:
                result = first_result; first_result = None
                for _ in range(count):
                    wm_round()
            else:
                submit_collection(count)
                result = await_worker(wm_rounds=count)
            before_fit = state['critic_ready']
            if cfg.get('critic_on_worker'):
                calibration = result['calibration']
                critic.load_state_dict(result['critic']); ov.load_state_dict(result['critic_optimizer'])
                freeze_normalizers(critic)
                batches = result['actor_batches']
                if calibration['improved'] and len(batches) < count:
                    raise ValueError('Worker supplied too few independent actor batches')
            else:
                data = to_device(result['data'], task.alpha.device)
                calibration = fit_critic(critic, ov, data, cfg, first=not before_fit)
            del result
            state['critic_updates'] += calibration['updates']
            state['last_calibration'] = calibration
            state['critic_ready'] = before_fit or calibration['improved']
            log('critic_calibration')
            if not calibration['improved']:
                state['calibration_failures'] += 1
                if not cfg.get('critic_on_worker'):
                    del data
                if state['calibration_failures'] >= 3:
                    raise RuntimeError('Three physical critic calibrations failed; actor remains unchanged')
                continue
            state['calibration_failures'] = 0
            # Actor starts match the deterministic-tail distribution used by its critic.
            if not cfg.get('critic_on_worker'):
                actor_data = take(data, (data['env_id'] < cfg['critic_train_envs']).nonzero().flatten())
            candidate_snapshot = snapshot_actor(pi, op)
            gate_seed = cfg['seed'] + 50000 + cur.level
            if cfg.get('overlap_baseline'):
                worker.submit('baseline', candidate_snapshot[0], version=state['version'], level=cur.level, seed=gate_seed)
            actor_logs = []
            for batch_index in range(count):
                if cfg.get('critic_on_worker'):
                    actor_data = to_device(batches[batch_index], task.alpha.device)
                actor_logs.append(actor_step(pi, wm, critic, task, actor_data, op, cfg,
                                             prebatched=cfg.get('critic_on_worker', False)))
                state['attempted_actor_updates'] += 1
            del actor_data
            if cfg.get('critic_on_worker'):
                del batches
            else:
                del data
            if cfg.get('overlap_baseline'):
                baseline = await_worker()
                worker.submit('candidate', pi, version=state['version'], level=cur.level, seed=gate_seed,
                              baseline_job_id=baseline['job_id'], baseline_digest=baseline['digest'],
                              protected=state['standing_protected'], anchor=state['anchor'])
            else:
                worker.submit('pair', candidate_snapshot[0], version=state['version'], level=cur.level,
                              seed=gate_seed, candidate=pi.state_dict(),
                              protected=state['standing_protected'], anchor=state['anchor'])
            log('candidate_evaluation')
            result = await_worker()
            decision = result['decision']
            state['last_decision'] = decision
            state['blocks'] += 1
            if decision['accepted']:
                state['actor_updates'] += count; state['accepted_blocks'] += 1
                state['version'] += 1; state['consecutive_rejections'] = 0
                for _ in range(count):
                    exploration.step()
                c = result['candidate']
                key = [c['standing_pass'], -c['fell'], c['survival_s'], c['return_'], -c['feet_max_m']]
                if state['best_key'] is None or tuple(key) > tuple(state['best_key']):
                    state['best_key'] = key; state['anchor'] = c
                    checkpoint('best.pt')
            else:
                restore_actor(pi, op, candidate_snapshot)
                state['rejected_blocks'] += 1; state['consecutive_rejections'] += 1
            candidate_snapshot = None
            checkpoint()
            log('candidate_accepted' if decision['accepted'] else 'candidate_rejected', actor_batch=actor_logs,
                baseline=result['base'], candidate=result['candidate'])
            budget_exhausted = cfg['actor_budget'] and state['attempted_actor_updates'] >= cfg['actor_budget']
            if state['consecutive_rejections'] >= 3 and not budget_exhausted:
                # Reduce only after repeated failures, preserving Adam moments on rollback.
                op.param_groups[0]['lr'] = max(cfg['actor_min_lr'], op.param_groups[0]['lr'] * .5)
                for _ in range(cfg['recovery_wm_rounds']):
                    if stop_requested:
                        raise InterruptedError('Stop requested')
                    wm_round()
                state['consecutive_rejections'] = 0
                log('model_recovery'); checkpoint()
            if decision['accepted'] and state['accepted_blocks'] % cfg['formal_eval_blocks'] == 0:
                official()
                if not state['completed'] and not args.smoke:
                    worker.submit('validate_wm', pi, version=state['version'], level=cur.level,
                                  seed=cfg['seed'] + 710, wm=wm.state_dict())
                    state['wm_validation'] = await_worker()['report']
                    log('wm_validation')
        reason = 'curriculum_accepted' if state['completed'] else ('stopped' if stop_requested else 'finite_budget')
    except InterruptedError:
        reason = 'stopped'
    except BaseException as exc:
        if candidate_snapshot is not None:
            restore_actor(pi, op, candidate_snapshot)
            candidate_snapshot = None
        checkpoint()
        log('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if candidate_snapshot is not None:
            restore_actor(pi, op, candidate_snapshot)
        if worker is not None:
            worker.close()
        memory_monitor.close()
    checkpoint()
    log('completed' if state['completed'] else 'stopped', reason=reason)
    json_save(out / 'completion.json', dict(passed=state['completed'], reason=reason, smoke=bool(args.smoke),
                                          wiring_complete=state['attempted_actor_updates'] > 0,
                                          state=state, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))
    if state['completed']:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'tools.export_aiming_fullbody', '--ckpt', str(out / 'last.pt'),
                        '--out', str(out / 'export')], check=True)


if __name__ == '__main__':
    main()
