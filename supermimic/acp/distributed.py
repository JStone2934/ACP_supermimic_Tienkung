"""Synchronous ACP data parallelism, with communication outside compiled graphs.

CLI sizes are global. Only environment and imagination/model batches become local;
replay time windows, optimizer update counts and schedules never scale with ranks.
"""
from __future__ import annotations

import atexit
import copy
import datetime
import faulthandler
import hashlib
import io
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist


def add_arguments(parser, *, planner=False, reward=True):
    parser.add_argument('--device', default=None, help='single-process device; torchrun uses LOCAL_RANK')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None)
    parser.add_argument('--dist_timeout', type=int, default=1800, help='collective timeout in seconds, including first compilation')
    parser.add_argument('--verify_sync', action='store_true', help='verify model and optimizer equality at checkpoints')
    parser.add_argument('--benchmark_warmup', type=int, default=10)
    parser.add_argument('--benchmark_steps', type=int, default=0, help='measure this many updates after warmup; 0 disables')
    if reward:
        parser.add_argument('--reward_chunk_size', type=int, default=0, help='checkpoint analytic rewards in chunks of state rows; 0 keeps original execution')
    if planner:
        parser.add_argument('--planner_batch', type=int, default=None, help='global planner batch (default: checkpoint or 4096)')
        parser.add_argument('--planner_sub_batches', type=int, default=None)
        parser.add_argument('--planner_min_entries', type=int, default=None, help='minimum GLOBAL hindsight samples before learning')


def merge_moments(parts):
    """Chan merge of rows [count, mean..., M2...], including empty shards."""
    d = (parts.shape[-1] - 1) // 2
    n = parts[:, :1]
    count = n.sum()
    mean = (parts[:, 1:1+d] * n).sum(0) / count.clamp(min=1)
    m2 = (parts[:, 1+d:] + n * (parts[:, 1:1+d] - mean).square()).sum(0)
    return torch.cat((count.reshape(1), mean, m2))


def moments(x):
    x = x.reshape(-1, x.shape[-1]).double()
    if len(x) == 0:
        return torch.zeros(1 + 2 * x.shape[-1], device=x.device, dtype=torch.float64)
    mean = x.mean(0)
    return torch.cat((x.new_tensor([len(x)]), mean, (x - mean).square().sum(0)))


class Runtime:
    def __init__(self, args, *, backend=None):
        if os.environ.get('ACP_DISTRIBUTED_DEBUG') == '1':
            faulthandler.register(signal.SIGUSR1, all_threads=True)
            faulthandler.dump_traceback_later(120, repeat=True)
        self.args = copy.deepcopy(vars(args))
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.active = self.world_size > 1
        requested = getattr(args, 'device', None)
        if self.active and requested is not None and backend != 'gloo':
            raise ValueError('--device is only for single-process runs; torchrun binds LOCAL_RANK')
        self.device = requested or f'cuda:{self.local_rank if self.active else 0}'
        self.main = self.rank == 0
        self.updates, self.skipped = {}, {}
        self.models, self.optimizers = {}, {}
        self._bench_count = 0
        self._bench_start = None
        self._bench_done = False
        self._device_peak = 0
        for key in ('num_envs', 'wm_batch', 'pi_batch'):
            if hasattr(args, key):
                setattr(args, key, self.split(getattr(args, key), key))
        if hasattr(args, 'pi_batch'):
            groups = [args.pi_batch]
            if hasattr(args, 'goal_frac'):
                for key in ('num_envs', 'pi_batch'):
                    total = self.args[key]
                    goal = round(total * args.goal_frac)
                    self.split(goal, key + ' goal group')
                    self.split(total - goal, key + ' reference group')
                goal = round(args.pi_batch * args.goal_frac)
                groups = [goal, args.pi_batch - goal]
            if args.critic_minibatches < 1 or any(n * args.horizon % args.critic_minibatches for n in groups):
                raise ValueError('each local imagination group × horizon must be divisible by critic_minibatches')
            if any(n < 1 for n in groups):
                raise ValueError('each imagination group must have at least one sample on every rank')
        for key in ('buffer_steps', 'horizon', 'wm_window', 'wm_batches', 'critic_epochs', 'log_every', 'val_every', 'ckpt_every', 'utd'):
            if hasattr(args, key) and getattr(args, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if hasattr(args, 'hist') and args.hist < 2:
            raise ValueError('hist must be >= 2 for the previous two actions')
        if hasattr(args, 'wm_window') and args.buffer_steps < args.wm_window:
            raise ValueError('buffer_steps must hold wm_window steps')
        if getattr(args, 'reward_chunk_size', 0) < 0:
            raise ValueError('reward_chunk_size must be nonnegative')
        if self.device.startswith('cuda'):
            torch.cuda.set_device(self.device)
        if self.active:
            dist.init_process_group(backend or 'nccl', timeout=datetime.timedelta(seconds=getattr(args, 'dist_timeout', 1800)))
            atexit.register(self.close)
        self.seed(self.args.get('seed', 0))
        self.log_dir = Path(self.args.get('log_dir', '.'))
        if self.main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.write_json('args.json', self.args)
            self.write_json('runtime.json', dict(world_size=self.world_size, global_sizes={k: self.args[k] for k in ('num_envs', 'wm_batch', 'pi_batch') if k in self.args},
                                                local_sizes={k: getattr(args, k) for k in ('num_envs', 'wm_batch', 'pi_batch') if hasattr(args, k)},
                                                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), device=self.device))

    def split(self, total, name):
        if total < self.world_size or total % self.world_size:
            raise ValueError(f'global {name}={total} must be positive and divisible by world_size={self.world_size}')
        return total // self.world_size

    def capacity(self, total):
        if total < self.world_size:
            raise ValueError(f'global buffer capacity {total} must be >= world_size={self.world_size}')
        return total // self.world_size + int(self.rank < total % self.world_size)

    def planner_config(self, cfg):
        for key in ('batch', 'sub_batches', 'min_entries'):
            value = self.args.get('planner_' + key)
            if value is not None:
                setattr(cfg, key, value)
        if min(cfg.batch, cfg.sub_batches, cfg.min_entries) < 1 or cfg.batch % (cfg.sub_batches * self.world_size):
            raise ValueError('global planner_batch must be divisible by planner_sub_batches × world_size; sizes must be positive')
        minimum = self.world_size * 5
        if 'goal_frac' in self.args:
            from .goal import TERRAIN_TYPES
            minimum = max(minimum, self.world_size * len(TERRAIN_TYPES))
        if cfg.capacity < max(cfg.min_entries, minimum):
            raise ValueError('planner capacity must hold min_entries and a nonempty validation shard')
        return cfg

    def seed(self, value):
        random.seed(value)
        np.random.seed(value % 2**32)
        torch.manual_seed(value)

    def register(self, models, optimizers, *, deferred=()):
        self.models, self.optimizers = models, optimizers
        from .networks import RunningNorm
        for module in models.values():
            if self.active:
                for value in module.state_dict().values():
                    dist.broadcast(value, 0)
            for child in module.modules():
                if isinstance(child, RunningNorm):
                    child._runtime = self
                    child._deferred = any(child is norm for norm in deferred)
        self.seed(self.args.get('seed', 0) + self.rank)

    def sum(self, value):
        value = torch.as_tensor(value, device=self.device).clone()
        if self.active:
            dist.all_reduce(value)
        return value

    def average(self, value):
        return self.sum(value) / self.world_size

    def ratio(self, numerator, denominator):
        pair = self.sum(torch.stack((numerator, denominator)))
        return pair[0] / pair[1].clamp(min=1)

    def gather(self, tensor):
        if not self.active:
            return [tensor]
        values = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(values, tensor.contiguous())
        return values

    def objects(self, value):
        if not self.active:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    @torch.no_grad()
    def norm_update(self, norm, x):
        if norm.count >= norm.max_count:
            return
        part = moments(x)
        if getattr(norm, '_deferred', False):
            old = getattr(norm, '_pending', None)
            norm._pending = part if old is None else merge_moments(torch.stack((old, part)))
        else:
            self._apply_norm(norm, part)

    @torch.no_grad()
    def _apply_norm(self, norm, part):
        combined = merge_moments(torch.stack(self.gather(part)))
        n = combined[0]
        if n == 0:
            return
        d = len(norm.mean)
        old = torch.cat((norm.count.double().reshape(1), norm.mean.double(), norm.var.double() * norm.count.double()))
        result = merge_moments(torch.stack((old, combined)))
        if not torch.isfinite(result).all():
            raise FloatingPointError('non-finite global normalization statistics')
        norm.count.copy_(result[0])
        norm.mean.copy_(result[1:1+d])
        norm.var.copy_(result[1+d:] / result[0])

    @torch.no_grad()
    def flush_norm(self, norm):
        if norm.count >= norm.max_count:
            return
        part = getattr(norm, '_pending', None)
        if part is None:
            part = torch.zeros(1 + 2 * len(norm.mean), dtype=torch.float64, device=norm.mean.device)
        self._apply_norm(norm, part)
        norm._pending = None

    def global_std(self, x):
        part = merge_moments(torch.stack(self.gather(moments(x.detach().reshape(-1, 1)))))
        return (part[2] / (part[0] - 1).clamp(min=1)).sqrt().float()

    @torch.no_grad()
    def reference_sampling(self, bin_G, bins, returns, *, phase, delta, phase_max, bin_w_avg, G_max, update_weights, ended):
        """Update shared sampling statistics before any rank resets its environments."""
        sums, counts = torch.zeros_like(bin_G), torch.zeros_like(bin_G)
        sums.index_add_(0, bins, returns)
        counts.index_add_(0, bins, torch.ones_like(returns))
        combined = self.sum(torch.cat((sums, counts, delta, sums.new_tensor([ended]))))
        n = len(bin_G)
        sums, counts = combined[:n], combined[n:2*n]
        updated = counts > 0
        bin_G[updated] = .8 * bin_G[updated] + .2 * sums[updated] / counts[updated]
        phase.add_(combined[2*n:-1]).clamp_(0., phase_max)
        if update_weights and combined[-1] > 0:
            soft = torch.softmax(-25. * bin_G / G_max, 0)
            bin_w_avg.mul_(.999).add_(.001 * soft)

    @torch.no_grad()
    def step(self, optimizer, module, name, *, max_norm=None, clip_value=None):
        """Average gradients before clipping; all ranks either update or skip."""
        params = list(module.parameters())
        present = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=self.device)
        finite = torch.ones((), dtype=torch.int32, device=self.device)
        for p in params:
            if p.grad is not None:
                finite.mul_(torch.isfinite(p.grad).all().int())
        if self.active:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            dist.all_reduce(present, op=dist.ReduceOp.MAX)
        if not finite:
            self.skipped[name] = self.skipped.get(name, 0) + 1
            optimizer.zero_grad(set_to_none=True)
            self.print(f'warning: skipped {name} update: non-finite gradient on at least one rank', flush=True)
            return torch.tensor(float('nan'), device=self.device)
        selected = []
        for p, used in zip(params, present.tolist()):
            if used:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                selected.append(p)
        # Deterministic buckets bound the temporary communication allocation.
        buckets, bucket, size = [], [], 0
        for p in selected:
            if bucket and (size + p.numel() * p.element_size() > 25 * 1024**2 or bucket[0].dtype != p.dtype):
                buckets.append(bucket)
                bucket, size = [], 0
            bucket.append(p)
            size += p.numel() * p.element_size()
        if bucket:
            buckets.append(bucket)
        if self.active:
            for bucket in buckets:
                flat = torch.cat([p.grad.reshape(-1) for p in bucket])
                flat.div_(self.world_size)
                dist.all_reduce(flat)
                offset = 0
                for p in bucket:
                    p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
                    offset += p.numel()
        gn = torch.nn.utils.clip_grad_norm_(selected, max_norm if max_norm is not None else float('inf'))
        if not torch.isfinite(gn):
            self.skipped[name] = self.skipped.get(name, 0) + 1
            optimizer.zero_grad(set_to_none=True)
            self.print(f'warning: skipped {name} update: non-finite reduced gradient norm', flush=True)
            return gn
        if clip_value is not None and selected:
            torch.nn.utils.clip_grad_value_(selected, clip_value)
        optimizer.step()
        self.updates[name] = self.updates.get(name, 0) + 1
        return gn

    def aggregate(self, stats):
        """Merge event lists for exact sums/ratios, including empty ranks."""
        def plain(x):
            return x.detach().cpu() if isinstance(x, torch.Tensor) else x
        parts = self.objects({k: [plain(v) for v in values] for k, values in stats.items()})
        return {k: [x for part in parts for x in part[k]] for k in stats}

    def global_size(self, buffer):
        return int(self.sum(buffer.size))

    def print(self, *args, **kwargs):
        if self.main:
            print(*args, **kwargs)

    def open_log(self, path, mode='a'):
        return open(path, mode) if self.main else io.StringIO()

    def write_json(self, name, value):
        if self.main:
            path = self.log_dir / name
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_text(json.dumps(value, indent=2) + '\n')
            tmp.replace(path)

    def restore(self, ck):
        self.updates = dict(ck.get('distributed', {}).get('updates', {}))
        self.skipped = dict(ck.get('distributed', {}).get('skipped', {}))

    def check_sync(self, extra=None):
        def digest(value, h):
            if isinstance(value, torch.Tensor):
                h.update(str((value.dtype, tuple(value.shape))).encode())
                h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(value, dict):
                for key in sorted(value, key=str):
                    h.update(str(key).encode())
                    digest(value[key], h)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    digest(item, h)
            else:
                h.update(repr(value).encode())
        state = {**{k: v.state_dict() for k, v in self.models.items()}, **{k: v.state_dict() for k, v in self.optimizers.items()}}
        state['sampling_and_schedule'] = extra or {}
        h = hashlib.sha256()
        digest(state, h)
        hashes = self.objects(h.hexdigest())
        if len(set(hashes)) != 1:
            raise RuntimeError(f'model/optimizer divergence between ranks: {hashes}')
        return hashes[0]

    def save(self, ck, *, numbered, small_keys=None, full=False):
        ck['args'] = self.args
        ck['distributed'] = dict(version=1, world_size=self.world_size, updates=dict(self.updates), skipped=dict(self.skipped))
        if self.device.startswith('cuda'):
            ck['distributed']['ranks'] = self.objects(dict(rank=self.rank, device=self.device,
                **self.memory_stats()))
        if self.args.get('verify_sync'):
            keys = ('curriculum_phase', 'bin_G', 'bin_w_avg', 'admission_totals', 'admission_history',
                    'sched_it', 'admit_base', 'scale', 'curriculum_level', 'curriculum_success')
            ck['distributed']['state_sha256'] = self.check_sync({k: ck[k] for k in keys if k in ck})
        if self.main:
            for name, content in [('last.pt', ck), (numbered, ck if full or small_keys is None else {k: ck[k] for k in small_keys} | {'distributed': ck['distributed']})]:
                path = self.log_dir / name
                torch.save(content, str(path) + '.tmp')
                os.replace(str(path) + '.tmp', path)
            self.write_json('updates.json', ck['distributed'])

    def memory_stats(self):
        import warp as wp
        free, total = torch.cuda.mem_get_info(self.device)
        return dict(torch_peak_allocated=torch.cuda.max_memory_allocated(self.device),
                    torch_peak_reserved=torch.cuda.max_memory_reserved(self.device), device_used=total-free,
                    warp_pool_used=wp.get_mempool_used_mem_current(self.device),
                    warp_pool_peak=wp.get_mempool_used_mem_high(self.device))

    def tick(self, step, sim_steps, *, ready=True):
        target = self.args.get('benchmark_steps', 0)
        if target <= 0 or self._bench_done or not ready:
            return
        torch.cuda.synchronize(self.device)
        self._bench_count += 1
        warmup = self.args.get('benchmark_warmup', 10)
        if self._bench_start is None and self._bench_count >= warmup:
            if self.active:
                dist.barrier()
            torch.cuda.reset_peak_memory_stats(self.device)
            self._bench_start = (time.perf_counter(), sim_steps, dict(self.updates), self._bench_count)
            return
        if self._bench_start is None:
            return
        free, total = torch.cuda.mem_get_info(self.device)
        self._device_peak = max(self._device_peak, total - free)
        if self._bench_count - self._bench_start[3] < target:
            return
        elapsed = time.perf_counter() - self._bench_start[0]
        ranks = self.objects(dict(rank=self.rank, elapsed_s=elapsed, **self.memory_stats(), device_peak_used=self._device_peak))
        wall = max(r['elapsed_s'] for r in ranks)
        self.write_json('benchmark.json', dict(world_size=self.world_size, measured_iterations=target, warmup_updates=warmup,
                                              elapsed_s=wall, iterations_per_s=target/wall, transitions_per_s=(sim_steps-self._bench_start[1])/wall,
                                              updates={k: v-self._bench_start[2].get(k, 0) for k, v in self.updates.items()}, ranks=ranks, args=self.args))
        self._bench_done = True

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
