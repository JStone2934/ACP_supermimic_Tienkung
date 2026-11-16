"""Small, persistent host-memory diagnostics for the physical trainer."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import threading


def process_memory(pid):
    try:
        fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        return {key: int(fields[key].split()[0]) * 1024 for key in ('VmRSS', 'VmHWM', 'VmSwap') if key in fields}
    except (OSError, ValueError):
        return {}


def memory_snapshot():
    result = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(),
                  process=process_memory(os.getpid()))
    try:
        group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        root = Path('/sys/fs/cgroup') / group.lstrip('/')
        result['cgroup'] = group
        for name in ('memory.current', 'memory.peak', 'memory.high', 'memory.max', 'memory.swap.current', 'memory.swap.max'):
            path = root / name
            if path.exists():
                value = path.read_text().strip()
                result[name] = int(value) if value.isdigit() else value
        for name in ('memory.events', 'memory.stat'):
            fields = dict(line.split() for line in (root / name).read_text().splitlines())
            result[name] = {k: int(v) for k, v in fields.items()
                            if name == 'memory.events' or k in ('anon', 'file', 'kernel', 'pagetables', 'slab')}
        result['memory.pressure'] = (root / 'memory.pressure').read_text().strip()
    except (OSError, ValueError, StopIteration):
        pass
    return result


class MemoryMonitor:
    def __init__(self, directory, context, interval=5.):
        self.directory = Path(directory)
        self.context = context
        self.interval = interval
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, name='training-memory-log', daemon=True)

    def start(self):
        self.thread.start()

    def sample(self):
        row = memory_snapshot()
        row.update(self.context())
        if row.get('worker_pid'):
            row['worker_process'] = process_memory(row['worker_pid'])
        with (self.directory / 'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        temporary = self.directory / 'memory_latest.json.tmp'
        temporary.write_text(json.dumps(row, indent=2) + '\n')
        temporary.replace(self.directory / 'memory_latest.json')

    def run(self):
        while not self.stopped.is_set():
            try:
                self.sample()
            except Exception as error:
                print(f'memory monitor: {type(error).__name__}: {error}', flush=True)
            self.stopped.wait(self.interval)

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2.)
