"""Stream Stage1 CSV history and continuation metrics into TensorBoard."""
import argparse
import csv
import io
import math
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    writer = EventFileWriter(str(args.out))
    seen = set()
    windows = {}
    try:
        while True:
            for path in args.csv:
                if not path.exists():
                    continue
                content = path.read_text()
                # Ignore a partially written final row; retry on the next poll.
                content = content[:content.rfind('\n') + 1]
                for row in csv.DictReader(io.StringIO(content)):
                    try:
                        step = int(row['iteration'])
                        values = {k: float(v) for k, v in row.items() if k != 'iteration'}
                    except (ValueError, TypeError):
                        continue
                    if step in seen:
                        continue
                    scalars = []
                    for key, value in values.items():
                        if not math.isfinite(value):
                            continue
                        group = ('curriculum' if key.startswith('phase_') else
                                 'performance' if key in ('ep_ret_norm', 'term_frac', 'touch_precision', 'touch_recall', 'ep_len', 'sim_rew') else
                                 'optimization')
                        scalars.append(Summary.Value(tag=f'{group}/{key}', simple_value=value))
                        window = windows.setdefault(key, [])
                        window.append((step, value))
                        window[:] = [(s, v) for s, v in window if s > step - 1000]
                        scalars.append(Summary.Value(tag=f'{group}_mean1000/{key}', simple_value=sum(v for _, v in window)/len(window)))
                    writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=scalars)))
                    seen.add(step)
            writer.flush()
            time.sleep(10)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
