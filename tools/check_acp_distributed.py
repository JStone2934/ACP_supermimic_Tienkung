"""Reproducible short ACP training/benchmark on isolated numerical fixtures.

These fixtures and relaxed hindsight admission thresholds test execution only;
they do not establish standing, locomotion, or goal-reaching ability.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from supermimic.acp.robots import get_robot


def prepare(root):
    robot = get_robot('tienkung3_29dof')
    motions = root / 'motions'
    contacts = root / 'contacts'
    folder = motions / 'synthetic'
    folder.mkdir(parents=True, exist_ok=True)
    jp = np.zeros((41, 7 + robot.num_joints))
    jp[:, 2] = robot.stand_height; jp[:, 3] = 1; jp[:, 7:] = robot.default_q
    for i in range(10):
        path = folder / f'standing_{i:02d}_motion.npz'
        if not path.exists():
            np.savez_compressed(path, joint_pos=jp, joint_names=np.array(robot.joint_names), fps=np.array([50.]),
                                robot_metadata=json.dumps(robot.metadata()), data_version='distributed-fixture-v1',
                                source='synthetic constant pose', purpose='distributed numerical integration only')
            np.save(folder / f'standing_{i:02d}_terrain.npy', np.zeros((0, 1, 10)))
    if not (contacts / 'summary.json').exists():
        subprocess.run([sys.executable, '-m', 'supermimic.acp.make_labels', '--robot', robot.robot_id,
                        '--motion_dir', str(motions), '--out', str(contacts), '--workers', '2'], check=True)
    (root / 'fixture.json').write_text(json.dumps(dict(robot=robot.metadata(), frames_per_clip=41, clips=10,
          purpose='numerical integration only; no policy capability claim',
          planner_test_overrides=dict(start_max=1., reach_min=0., min_entries=16),
          learning_test_overrides=dict(lr=1e-6, lr_fs=1e-6, wm_only_iters=0)), indent=2))
    return motions, contacts


def check_resumes(root, env, port, stages, label):
    """Exercise full checkpoint transitions without touching benchmark runs."""
    result = {}
    source_dir = root / ('dual_' + label if label else 'dual')
    result_dir = root / ('resume_' + label if label else 'resume')
    result_dir.mkdir(exist_ok=True)
    for stage in stages:
        source = source_dir / stage / 'last.pt'
        original = json.loads((source_dir / stage / 'command.json').read_text())
        module = original[original.index('--module') + 1]
        options = original[original.index('--robot'):]
        for transition, dual in (('dual_to_dual', True), ('dual_to_single', False), ('single_to_dual', True)):
            ck = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
            previous = ck.get('step', ck.get('iteration'))
            out = result_dir / stage / transition
            out.mkdir(parents=True, exist_ok=True)
            opts = list(options)
            for flag in ('--init', '--planner_init', '--tracker', '--benchmark_steps', '--log_dir', '--iterations', '--steps'):
                if flag in opts:
                    index = opts.index(flag); del opts[index:index+2]
            opts += ['--resume', str(source), '--log_dir', str(out), '--benchmark_steps', '0',
                     '--steps' if stage == 'planner' else '--iterations', str(previous + (20 if stage == 'planner' else 2))]
            command = [sys.executable, '-u']
            if dual:
                command += ['-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2', '--master-addr=127.0.0.1',
                            f'--master-port={port}', '--max-restarts=0', '--module', module]
            else:
                command += ['-m', module]
                opts += ['--device', 'cuda:1']
            command += opts
            (out / 'command.json').write_text(json.dumps(command, indent=2))
            print(f'RESUME {stage}/{transition}', flush=True)
            with (out / 'stdout.log').open('w') as log:
                subprocess.run(command, env=env | {'CUDA_VISIBLE_DEVICES':'0,1'}, stdout=log, stderr=subprocess.STDOUT, check=True)
            new = torch.load(out / 'last.pt', map_location='cpu', weights_only=True, mmap=True)
            assert new['sim_steps'] > ck['sim_steps']
            assert new['args']['num_envs'] == ck['args']['num_envs']
            for key, value in ck['distributed']['updates'].items():
                assert new['distributed']['updates'][key] > value, (stage, key)
            assert new['distributed']['skipped'] == ck['distributed']['skipped']
            if stage == 'stage3':
                assert new['curriculum_level'].shape == ck['curriculum_level'].shape
            result[f'{stage}/{transition}'] = dict(status='PASS', sim_steps=new['sim_steps'], distributed=new['distributed'])
            # Keep dual_to_dual and dual_to_single independent; the last transition resumes the single run.
            if not dual:
                source = out / 'last.pt'
        if stage == 'stage3':
            bad = result_dir / 'stage3_changed_environments'
            bad.mkdir(exist_ok=True)
            invalid = list(command)
            invalid[invalid.index('--num_envs')+1] = str(2 * ck['args']['num_envs'])
            invalid[invalid.index('--log_dir')+1] = str(bad)
            with (bad / 'stdout.log').open('w') as log:
                failed = subprocess.run(invalid, env=env | {'CUDA_VISIBLE_DEVICES':'0,1'}, stdout=log, stderr=subprocess.STDOUT)
            assert failed.returncode != 0 and not (bad / 'last.pt').exists()
            assert 'same global environment/group counts' in (bad / 'stdout.log').read_text()
            result['stage3/changed_environments_rejected'] = dict(status='PASS')
    for stage in (s for s in stages if s != 'planner'):
        source = source_dir / stage / 'last.pt'
        module = 'supermimic.acp.evaluate_goal' if stage == 'stage3' else 'supermimic.acp.evaluate'
        out = result_dir / f'{stage}_evaluation.json'
        command = [sys.executable, '-m', module, '--ckpt', str(source), '--out', str(out)]
        if stage == 'stage2':
            command += ['--planner', str(source)]
        if stage == 'stage3':
            command += ['--episodes', '1']
        with out.with_suffix('.log').open('w') as log:
            subprocess.run(command, env=env | {'CUDA_VISIBLE_DEVICES':'1'}, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert out.exists()
        result[stage + '/evaluation_load'] = dict(status='PASS', output=str(out))
    legacy = Path('logs/tienkung3_29dof/integration_stage1_v3/last.pt')
    if legacy.exists() and 'stage1' in stages:
        old = torch.load(legacy, map_location='cpu', weights_only=True, mmap=True)
        out = result_dir / 'legacy_stage1_resume'
        out.mkdir(exist_ok=True)
        options = dict(old['args'], iterations=old['iteration']+2, log_dir=str(out), resume=str(legacy))
        command = [sys.executable, '-m', 'supermimic.acp.train_stage1', '--verify_sync', '--device', 'cuda:1']
        for key, value in options.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    command.append('--'+key)
                elif key in ('blind', 'backward_curriculum'):
                    command.append('--no-'+key)
            else:
                command += ['--'+key, str(value)]
        with (out / 'stdout.log').open('w') as log:
            subprocess.run(command, env=env | {'CUDA_VISIBLE_DEVICES':'0,1'}, stdout=log, stderr=subprocess.STDOUT, check=True)
        loaded = torch.load(out / 'last.pt', map_location='cpu', weights_only=True, mmap=True)
        assert loaded['iteration'] == old['iteration'] + 2 and loaded['sim_steps'] > old['sim_steps']
        assert loaded['distributed']['updates']['pi'] == 2
        result['legacy_stage1_resume'] = dict(status='PASS', checkpoint=str(legacy))
        out = result_dir / 'legacy_stage1_init'
        out.mkdir(exist_ok=True)
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
                   '--master-addr=127.0.0.1', f'--master-port={port}', '--max-restarts=0', '--module', 'supermimic.acp.train_stage1',
                   '--robot', 'tienkung3_29dof', '--init', str(legacy), '--contacts_dir', 'data/tienkung3_29dof/contacts/bootstrap',
                   '--num_envs', '4', '--wm_batch', '4', '--pi_batch', '4', '--hist', '2', '--horizon', '2', '--wm_window', '2',
                   '--buffer_steps', '16', '--warmup_steps', '4', '--wm_batches', '1', '--critic_epochs', '1', '--critic_minibatches', '1',
                   '--lr', '0.000001', '--iterations', '2', '--log_every', '1', '--verify_sync', '--log_dir', str(out)]
        with (out / 'stdout.log').open('w') as log:
            subprocess.run(command, env=env | {'CUDA_VISIBLE_DEVICES':'0,1'}, stdout=log, stderr=subprocess.STDOUT, check=True)
        loaded = torch.load(out / 'last.pt', map_location='cpu', weights_only=True, mmap=True)
        assert loaded['distributed']['updates']['pi'] == 2
        result['legacy_stage1_init'] = dict(status='PASS', checkpoint=str(legacy))
    (result_dir / 'results.json').write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.cache/dual-5090/acceptance')
    ap.add_argument('--mode', choices=('dual', 'single', 'both'), default='both')
    ap.add_argument('--stages', default='stage1,planner,stage2,stage3')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--resume-only', action='store_true', help='test existing dual checkpoints on 2→2, 2→1, 1→2 and evaluation')
    ap.add_argument('--port', type=int, default=29547)
    ap.add_argument('--iterations', type=int, default=80)
    ap.add_argument('--benchmark-steps', type=int, default=50)
    ap.add_argument('--profile', choices=('short', 'local512'), default='short', help='local512 checks the documented 512/256/256/512 configuration')
    ap.add_argument('--reward-chunk-size', type=int, default=0)
    ap.add_argument('--label', default='', help='suffix for separate runs sharing the same fixtures/initial checkpoints')
    args = ap.parse_args()
    if args.profile == 'local512' and not args.label:
        args.label = 'local512'
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    motions, contacts = prepare(root)
    if args.prepare_only:
        return
    env = os.environ | dict(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONNOUSERSITE='1',
                            TORCHINDUCTOR_COMPILE_THREADS='8')
    if args.resume_only:
        check_resumes(root, env, args.port, args.stages.split(','), args.label)
        return
    modes = ['dual', 'single'] if args.mode == 'both' else [args.mode]
    stages = args.stages.split(',')
    for mode in modes:
        dual = mode == 'dual'
        directory = root / (mode + '_' + args.label if args.label else mode); directory.mkdir(exist_ok=True)
        # Same starting checkpoints for the single/dual performance comparison.
        source = root / ('dual_' + args.label) if args.profile == 'local512' else root / 'dual'
        if args.profile != 'local512' and args.label and not (source / 'stage1/last.pt').exists():
            source = root / ('dual_' + args.label)
        if not (source / 'stage1/last.pt').exists():
            source = directory
        tracker = source / 'stage1/last.pt'
        planner = source / 'planner/last.pt'
        stage2 = source / 'stage2/last.pt'
        base = ['--robot', 'tienkung3_29dof', '--num_envs', '64', '--seed', '0', '--log_every', '10',
                '--ckpt_every', str(args.iterations), '--verify_sync', '--benchmark_warmup', '10', '--benchmark_steps', str(args.benchmark_steps)]
        learning = ['--iterations', str(args.iterations), '--wm_batch', '64', '--pi_batch', '64', '--horizon', '8',
                    '--wm_window', '2', '--wm_batches', '1', '--critic_epochs', '1', '--critic_minibatches', '2',
                    '--warmup_steps', '48', '--action_noise', '0.01', '--lr', '0.000001', '--reward_chunk_size', str(args.reward_chunk_size)]
        planning = ['--planner_batch', '64', '--planner_sub_batches', '2', '--planner_min_entries', '16', '--val_every', '40']
        configs = {
            'stage1': ('supermimic.acp.train_stage1', [*base, *learning, '--hist', '5', '--buffer_steps', '64',
                        '--wm_only_iters', '0', '--no-backward_curriculum', '--motion_dir', str(motions), '--contacts_dir', str(contacts)]),
            'planner': ('supermimic.acp.train_planner_offline', [*base, *planning, '--tracker', str(tracker), '--steps', str(args.iterations),
                        '--buffer_steps', '64', '--capacity', '2048', '--start_max', '1', '--reach_min', '0', '--action_noise', '0.01']),
            'stage2': ('supermimic.acp.train_stage2', [*base, *learning, *planning, '--init', str(tracker), '--planner_init', str(planner),
                        '--buffer_steps', '64', '--planner_capacity', '2048', '--planner_updates', '1', '--u_obs', '2', '--u_rew', '4',
                        '--n_ann', '20', '--hold_window', '4']),
            'stage3': ('supermimic.acp.train_stage3', [*base, *learning, *planning, '--init', str(stage2), '--buffer_steps', '768',
                        '--planner_capacity', '2048', '--planner_updates', '1', '--terrain_instances', '1', '--critic_warmup', '2', '--lr_fs', '0.000001']),
        }
        for stage in stages:
            module, options = configs[stage]
            if args.profile == 'local512':
                options = list(options)
                sizes = dict(num_envs=512, wm_batch=256, pi_batch=256, planner_batch=512, planner_sub_batches=8,
                             planner_min_entries=1000, horizon=32, hist=10, wm_window=8, wm_batches=10,
                             critic_epochs=5, critic_minibatches=4, warmup_steps=64, planner_capacity=300000,
                             capacity=300000, terrain_instances=16, buffer_steps=1350 if stage == 'stage3' else 1000)
                if stage == 'planner':
                    sizes['steps'] = max(20, args.iterations)
                for key, value in sizes.items():
                    if '--'+key in options:
                        options[options.index('--'+key)+1] = str(value)
            out = directory / stage
            if (out / 'acceptance.json').exists():
                continue
            command = [sys.executable, '-u']
            if dual:
                command += ['-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2', '--master-addr=127.0.0.1',
                            f'--master-port={args.port}', '--max-restarts=0', '--module', module]
            else:
                command += ['-m', module]
            command += [*options, '--log_dir', str(out)]
            out.mkdir(exist_ok=True)
            (out / 'command.json').write_text(json.dumps(command, indent=2))
            print(f'START {mode}/{stage}', flush=True)
            started = time.time()
            with (directory / f'{stage}.log').open('w') as log:
                subprocess.run(command, env=env | {'CUDA_VISIBLE_DEVICES': '0,1' if dual else '1'},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=True, mmap=True)
            counts = ck['distributed']['updates']
            expected = ('planner',) if stage == 'planner' else ('wm', 'pi', 'critic_m', 'critic_fs', 'planner') if stage == 'stage3' else ('wm', 'pi', 'critic', 'planner') if stage == 'stage2' else ('wm', 'pi', 'critic')
            assert all(counts.get(k, 0) >= 2 for k in expected), counts
            optimizer_keys = dict(wm='opt_wm', pi='opt_pi', critic='opt_v', critic_m='opt_vm', critic_fs='opt_vfs',
                                  planner='opt' if stage == 'planner' else 'opt_planner')
            for name in expected:
                states = ck[optimizer_keys[name]]['state'].values()
                assert any('exp_avg' in state and bool(state['exp_avg'].abs().max() > 0) for state in states), name
            assert not ck['distributed']['skipped'], ck['distributed']
            assert ck['distributed'].get('state_sha256')
            benchmark = json.loads((out / 'benchmark.json').read_text()) if args.benchmark_steps else None
            if benchmark is not None:
                assert benchmark['measured_iterations'] == args.benchmark_steps
            report = dict(status='PASS', elapsed_s=time.time()-started, updates=counts,
                          synchronized_sha256=ck['distributed']['state_sha256'], benchmark=benchmark)
            (out / 'acceptance.json').write_text(json.dumps(report, indent=2))
            print(f'PASS {mode}/{stage}: {counts}', flush=True)


if __name__ == '__main__':
    main()
