"""CPU collective regression tests; real MuJoCo/CUDA acceptance is separate."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from supermimic.acp.distributed import Runtime
from supermimic.acp.networks import RunningNorm
from supermimic.acp.planner import (ContactPlanner, PlannerBuffer, PlannerCfg, PlannerScan,
                                    MixedPlannerBuffer, GroupedPlannerBuffer, global_sample, planner_update,
                                    planner_loss, distributed_planner_metrics)


def worker(root, fail=False):
    torch.set_num_threads(1)
    rt = Runtime(SimpleNamespace(device='cpu', log_dir=root, seed=4, verify_sync=True, num_envs=4), backend='gloo')
    if fail:
        if rt.rank == 1:
            raise RuntimeError('injected worker failure')
        rt.sum(1)
        raise AssertionError('peer failure must abort the collective')
    checks = []
    model = torch.nn.Linear(3, 2)
    ref = copy.deepcopy(model)
    opt, ref_opt = torch.optim.Adam(model.parameters(), lr=.01), torch.optim.Adam(ref.parameters(), lr=.01)
    rt.register({'model': model}, {'opt': opt})
    x = torch.arange(24).view(8, 3).float() / 10
    for _ in range(3):
        opt.zero_grad(); ref_opt.zero_grad()
        model(x.chunk(2)[rt.rank]).square().mean().backward()
        ref(x).square().mean().backward()
        rt.step(opt, model, 'model', max_norm=.7)
        torch.nn.utils.clip_grad_norm_(ref.parameters(), .7)
        ref_opt.step()
        for a, b in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    rt.check_sync(); checks.append('global_gradient_and_adam')
    before = copy.deepcopy(model.state_dict())
    opt.zero_grad(); model(x[:2]).sum().backward()
    if rt.rank == 1:
        next(model.parameters()).grad[0, 0] = float('nan')
    rt.step(opt, model, 'model')
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert rt.skipped['model'] == 1
    rt.check_sync(); checks.append('nonfinite_collective_skip')

    sparse = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(3)])
    sparse_opt = torch.optim.SGD(sparse.parameters(), lr=.1)
    sparse[rt.rank].sum().backward()
    rt.step(sparse_opt, sparse, 'partial')
    torch.testing.assert_close(sparse[0], torch.full((2,), .95))
    torch.testing.assert_close(sparse[1], torch.full((2,), .95))
    assert sparse[2].grad is None
    checks.append('missing_and_unused_gradients')

    norm = RunningNorm(3)
    rt.register({'norm': norm}, {})
    norm.update(x[:3] if rt.rank == 0 else x[3:])
    torch.testing.assert_close(norm.mean, x.mean(0))
    torch.testing.assert_close(norm.var, x.var(0, unbiased=False))
    norm.update(x[:0] if rt.rank == 0 else x[:2])
    all_x = torch.cat((x, x[:2]))
    torch.testing.assert_close(norm.var, all_x.var(0, unbiased=False))
    assert norm.count == 10
    norm.max_count = 10
    norm.update(x)
    assert norm.count == 10
    rt.check_sync(); checks.append('weighted_norm_empty_and_freeze')

    deferred = RunningNorm(3)
    rt.register({'norm': deferred}, {}, deferred=(deferred,))
    if rt.rank == 1:
        deferred.update(x[:2]); deferred.update(x[2:])
    rt.flush_norm(deferred)
    torch.testing.assert_close(deferred.mean, x.mean(0))
    torch.testing.assert_close(deferred.var, x.var(0, unbiased=False))
    rt.flush_norm(deferred)
    assert deferred.count == 8
    checks.append('deferred_norm_empty_rank')

    # Unequal numbers of valid contacts must match a concatenated loss, not a mean of means.
    logits = torch.nn.Parameter(torch.arange(4*9*3).view(4,9,3).float()/100)
    occ_logits = torch.nn.Parameter(torch.zeros(4,9))
    pts = torch.zeros(4,3,3); tgt = torch.zeros(4,9,3)
    occ = torch.zeros(4,9, dtype=torch.bool); occ[2:, :3] = True
    idx = slice(rt.rank*2, (rt.rank+1)*2)
    la, lo = planner_loss(logits[idx].log_softmax(-1), occ_logits[idx], pts[idx], tgt[idx], occ[idx], .1)
    n = occ[idx].sum().float()
    local_loss = la * n * 2 / rt.sum(n).clamp(min=1) + lo
    local_loss.backward()
    grads = [rt.average(p.grad) for p in (logits, occ_logits)]
    full = [p.detach().clone().requires_grad_() for p in (logits, occ_logits)]
    a, o = planner_loss(full[0].log_softmax(-1), full[1], pts, tgt, occ, .1)
    (a+o).backward()
    for actual, p in zip(grads, full):
        torch.testing.assert_close(actual, p.grad)
    checks.append('masked_planner_loss_denominator')

    cfg = PlannerCfg(batch=16, sub_batches=2, capacity=100, min_entries=1)
    scan = PlannerScan(cfg, 'cpu')
    planner = ContactPlanner(3, cfg)
    opt = torch.optim.Adam(planner.parameters(), lr=1e-4)
    buf = PlannerBuffer(50, cfg.hist, 3, scan.num_z, 'cpu')
    rt.register({'planner': planner}, {'opt_planner': opt}, deferred=(planner.prop_norm,))
    if rt.rank == 1:
        data = {k: v[:12].clone() for k, v in buf.data.items()}
        data['prop'].normal_(); data['occ'][:, :3] = True
        data['clip'].fill_(17)
        buf.add(data); planner.prop_norm.update(data['prop'].float())
    rt.flush_norm(planner.prop_norm)
    batch = global_sample(buf, 16, rt)
    assert (batch['clip'] == 17).all()
    before = next(planner.parameters()).detach().clone()
    loss = planner_update(planner, opt, buf, scan, cfg, rt)
    assert torch.isfinite(loss).all() and not torch.equal(before, next(planner.parameters()))
    rt.check_sync()
    mt = distributed_planner_metrics(planner, buf, scan, cfg, rt, 8)
    assert mt['n_label'][:3] == [8., 8., 8.]
    checks.append('real_planner_update_with_empty_rank_and_global_validation')

    goal = GroupedPlannerBuffer(2, 100, cfg.hist, 3, scan.num_z, 'cpu', runtime=rt)
    # Unequal shard occupancy: group 0 has 1 sample, group 1 has 20; both get ~half the goal samples.
    data = {k: v[:1].clone() for k, v in buf.data.items()}
    data['clip'].fill_(rt.rank)
    if rt.rank == 1:
        data = {k: v.repeat((20,) + (1,)*(v.ndim-1)) for k, v in data.items()}
    goal.add(data, torch.full((len(data['clip']),), rt.rank))
    sample = global_sample(goal, 2000, rt)
    share = float((sample['clip'] == 0).float().mean())
    assert .44 < share < .56, share
    checks.append('global_terrain_balancing')
    bin_G = torch.zeros(2); weights = torch.full((2,), .5); phase = torch.tensor([.1, .8])
    bins = torch.tensor([0, 0]) if rt.rank == 0 else torch.tensor([0, 1])
    returns = torch.tensor([1., 3.]) if rt.rank == 0 else torch.tensor([5., 7.])
    delta = torch.tensor([-.2, .2]) if rt.rank == 0 else torch.tensor([.05, -.1])
    rt.reference_sampling(bin_G, bins, returns, phase=phase, delta=delta, phase_max=.9,
                          bin_w_avg=weights, G_max=10., update_weights=True, ended=2)
    torch.testing.assert_close(bin_G, torch.tensor([.6, 1.4]))
    torch.testing.assert_close(phase, torch.tensor([0., .9]))
    torch.testing.assert_close(weights, .999 * torch.full((2,), .5) + .001 * torch.softmax(-25*bin_G/10, 0))
    # A rank without ended episodes must enter the same collective.
    rt.reference_sampling(bin_G, torch.tensor([0]) if rt.rank else torch.empty(0, dtype=torch.long),
                          torch.tensor([4.]) if rt.rank else torch.empty(0), phase=phase, delta=torch.zeros(2),
                          phase_max=.9, bin_w_avg=weights, G_max=10., update_weights=False, ended=rt.rank)
    torch.testing.assert_close(bin_G, torch.tensor([1.28, 1.4]))
    checks.append('global_sampling_curriculum_and_empty_episode_rank')
    if rt.main:
        Path(root, 'passed.json').write_text(json.dumps(checks))
    rt.close()


class DistributedTests(unittest.TestCase):
    def test_label_geometry_chunks_preserve_contacts(self):
        from supermimic.acp.robots import get_robot
        from supermimic.acp.kinematics import RobotKinematics
        from supermimic.acp.motion import MotionLib
        from supermimic.acp.mdp import MDP
        from supermimic.acp.task import ContactTask, ContactCfg
        from supermimic.acp.planner import label_geometry
        robot = get_robot('tienkung3_29dof')
        kin = RobotKinematics('cpu', robot)
        motion = MotionLib.from_dir(robot.motion_dir, kin, 'cpu')
        mdp = MDP(kin, motion, torch.tensor(robot.default_q))
        contacts = str(Path(robot.motion_dir).parents[1] / 'contacts/bootstrap')
        ctc = ContactTask(ContactCfg(contacts_dir=contacts), mdp, robot.motion_dir, 'cpu')
        state = motion.ref_state(torch.arange(7))
        boxes = torch.tensor([0., 0., .6, 1., 0., .3, .2, .3]).expand(7, 1, 8)
        motion.goal_boxes = boxes[:1]  # include the hull/box path even for a flat reference fixture
        fk = kin.forward(*state)
        expected = ctc.body_distance(state, boxes, fk) <= ctc.cfg.touch_dist, ctc.contact_point(state, boxes, fk)
        actual = label_geometry(ctc, state, boxes, chunk_size=3)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_reward_chunks_preserve_outputs_and_gradients(self):
        from supermimic.acp.memory import compile_reward, reward_batch
        torch.manual_seed(0)
        x = torch.randn(11, 3, requires_grad=True)
        w = torch.randn(3, 3, requires_grad=True)
        scalar = torch.tensor(.7)

        def reward(state, previous, coefficient):
            y = (state[0] @ w).sin() * previous[0]
            return y.square().sum(-1) * coefficient, y.sigmoid().mean(-1)

        plain = reward((x,), (x * .3,), scalar)
        grads = torch.autograd.grad(sum(v.square().mean() for v in plain), (x, w))
        chunked = reward_batch(compile_reward(reward, recompute=True), (x,), (x * .3,), scalar, chunk_size=4)
        actual = torch.autograd.grad(sum(v.square().mean() for v in chunked), (x, w))
        for a, b in zip(chunked, plain):
            torch.testing.assert_close(a, b)
        for a, b in zip(actual, grads):
            torch.testing.assert_close(a, b)

    def test_collectives(self):
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close()
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
                   '--master-addr=127.0.0.1', f'--master-port={port}', '-m', 'tools.test_acp_distributed', '--worker', '--out', root]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=os.environ | {'OMP_NUM_THREADS':'1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(json.loads(Path(root, 'passed.json').read_text())), 9)

    def test_invalid_global_partition(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {'WORLD_SIZE':'2', 'RANK':'0', 'LOCAL_RANK':'0'}):
            with self.assertRaisesRegex(ValueError, 'divisible'):
                Runtime(SimpleNamespace(device=None, num_envs=3))
            with self.assertRaisesRegex(ValueError, 'only for single'):
                Runtime(SimpleNamespace(device='cuda:0', num_envs=4))
            with self.assertRaisesRegex(ValueError, 'goal group'):
                Runtime(SimpleNamespace(device=None, num_envs=6, pi_batch=8, goal_frac=.5, critic_minibatches=1, horizon=2))
            with self.assertRaisesRegex(ValueError, 'critic_minibatches'):
                Runtime(SimpleNamespace(device=None, num_envs=8, pi_batch=12, goal_frac=.5, critic_minibatches=4, horizon=2))
        rt = Runtime.__new__(Runtime)
        rt.args, rt.world_size = {}, 2
        with self.assertRaisesRegex(ValueError, 'planner_batch'):
            rt.planner_config(PlannerCfg(batch=10, sub_batches=2))

    def test_worker_failure_terminates_job(self):
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close()
            cmd = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
                   '--master-addr=127.0.0.1', f'--master-port={port}', '--max-restarts=0',
                   '-m', 'tools.test_acp_distributed', '--worker', '--fail-worker', '--out', root]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45, env=os.environ | {'OMP_NUM_THREADS':'1'})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('injected worker failure', result.stderr)
            self.assertFalse(Path(root, 'last.pt').exists())


if __name__ == '__main__':
    if '--worker' in sys.argv:
        ap = argparse.ArgumentParser(); ap.add_argument('--worker', action='store_true'); ap.add_argument('--out'); ap.add_argument('--fail-worker', action='store_true')
        args = ap.parse_args(); worker(args.out, args.fail_worker)
    else:
        unittest.main()
