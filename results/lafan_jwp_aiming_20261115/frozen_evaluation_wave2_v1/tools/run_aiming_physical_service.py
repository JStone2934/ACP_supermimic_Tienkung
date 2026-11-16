"""Resume the latest atomic checkpoint on each systemd start; never create a second trainer."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def resume_command(directory, python=sys.executable):
    directory = Path(directory).resolve()
    checkpoint = directory / 'last.pt'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f'Missing restart checkpoint: {checkpoint}')
    return [python, '-u', '-m', 'supermimic.acp.train_aiming_physical',
            '--resume', str(checkpoint), '--precision', 'fp32', '--log-dir', str(directory)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    args = parser.parse_args()
    owner = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'php-codex.service']).returncode == 0
    if owner and os.environ.get('PHP_OVERNIGHT_OWNER') != '1':
        print('PHP background owner is active; a single-owner handoff is required.', flush=True)
        raise SystemExit(78)
    try:
        command = resume_command(args.train_dir)
    except FileNotFoundError as error:
        print(str(error), flush=True)
        raise SystemExit(78)
    checkpoint = Path(command[command.index('--resume') + 1])
    record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(),
                  invocation_id=os.environ.get('INVOCATION_ID'), command=command,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    with (args.train_dir / 'service_starts.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print('Resuming latest checkpoint: ' + str(checkpoint), flush=True)
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
