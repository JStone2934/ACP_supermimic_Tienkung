"""Bounded read-only two-seed physical stage evaluation; never promotes a checkpoint."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','frozen-runner','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterations',type=int,nargs='+',required=True);p.add_argument('--level',type=int,required=True)
    p.add_argument('--seeds',type=int,nargs=2,default=[17,53]);p.add_argument('--timeout-hours',type=float,default=6)
    a=p.parse_args();assert len(set(a.seeds))==2;a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (a.out/'worker.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
        deadline=time.monotonic()+a.timeout_hours*3600
        for iteration in a.iterations:
            checkpoint=a.run/f'ckpt_{iteration:06d}.pt'
            while not checkpoint.exists():
                if time.monotonic()>deadline:raise TimeoutError(str(checkpoint))
                time.sleep(5)
            digest=sha(checkpoint);out=a.out/f'iteration_{iteration:06d}';out.mkdir(exist_ok=True)
            reports=[]
            for seed in a.seeds:
                report=out/f'seed_{seed}.json'
                if not report.exists():
                    command=[sys.executable,'-m','tools.evaluate_wide_stage','--checkpoint',str(checkpoint),
                        '--out',str(report),'--level',str(a.level),'--seed',str(seed)]
                    with (out/f'seed_{seed}.log').open('a') as log:
                        result=subprocess.run(command,cwd=a.frozen_runner,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError(f'Evaluation failed: {report}')
                measured=json.loads(report.read_text())
                if measured.get('checkpoint_sha256')!=digest or measured.get('seed')!=seed or measured.get('level')!=a.level:
                    raise ValueError('Evaluation identity mismatch')
                reports.append(measured)
            if sha(checkpoint)!=digest:raise ValueError('Checkpoint changed')
            summary=dict(checkpoint_sha256=digest,iteration=iteration,level=a.level,passed=all(r['passed'] for r in reports),
                reports=[dict(seed=r['seed'],passed=r['passed'],fixed_grid_all_pass=r['fixed_grid_all_pass'],random_success_rate=r['random_success_rate']) for r in reports],
                scope='Intermediate stage only; no final acceptance and no automatic promotion')
            (out/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
        (a.out/'worker_complete.json').write_text(json.dumps(dict(iterations=a.iterations)))


if __name__=='__main__':main()
