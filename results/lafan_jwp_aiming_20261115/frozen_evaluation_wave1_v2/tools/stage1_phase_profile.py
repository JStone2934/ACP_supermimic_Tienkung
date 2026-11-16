"""Optional probe-only timings; CUDA events measure stream elapsed time, including launch gaps."""
import functools
import json
import time
import torch


class PhaseProfile:
    def __init__(self, runtime):
        self.runtime = runtime
        self.active = False
        self.records = []
        self.stack = []
        self.steps = set()

    def iteration(self, local_step):
        # Profile after initial compilation, before the benchmark's 25-update warmup ends.
        self.active = 10 <= local_step < 15
        if self.active:
            self.steps.add(local_step)

    def section(self, name):
        def decorate(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                if not self.active:
                    return fn(*args, **kwargs)
                parent = self.stack[-1] if self.stack else 'iteration'
                self.stack.append(name)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before = time.perf_counter()
                begin.record()
                try:
                    return fn(*args, **kwargs)
                finally:
                    end.record()
                    self.records.append((name, parent, time.perf_counter()-before, begin, end))
                    self.stack.pop()
            return wrapped
        return decorate

    def finish(self):
        self.active = False
        torch.cuda.synchronize(self.runtime.device)
        result = {}
        for name, parent, cpu, begin, end in self.records:
            key = parent+'/'+name
            entry = result.setdefault(key, dict(calls=0, host_wall_s=0., stream_elapsed_s=0.))
            entry['calls'] += 1
            entry['host_wall_s'] += cpu
            entry['stream_elapsed_s'] += begin.elapsed_time(end)/1000
        for entry in result.values():
            entry['host_s_per_iteration'] = entry['host_wall_s']/max(len(self.steps), 1)
            entry['stream_s_per_iteration'] = entry['stream_elapsed_s']/max(len(self.steps), 1)
        out = dict(rank=self.runtime.rank, measured_local_steps=sorted(self.steps), sections=result,
                   interpretation='Nested inclusive timings; do not sum parent and child. CUDA event elapsed includes launch gaps, not SM occupancy. Profiling ends before the separate clean 50-iteration benchmark.')
        (self.runtime.log_dir/f'phase_profile_rank{self.runtime.rank}.json').write_text(json.dumps(out, indent=2)+'\n')
