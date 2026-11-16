"""Bounded evaluation worker; never trains, promotes, or changes a checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.check_reference_training_gate import check
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.robots import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--iterations',type=int,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--timeout-hours',type=float,default=3.)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+args.timeout_hours*3600
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    identity=dict(dataset_manifest_sha256=sha256(args.dataset/'manifest.json'),run=str(args.run.resolve()),
                  iterations=args.iterations,scope=__doc__,pid=os.getpid())
    json_save(args.out/'worker.json',identity)
    for iteration in args.iterations:
        ck=args.run/f'ckpt_{iteration:06d}.pt';dest=args.out/f'iteration_{iteration:06d}'
        while not ck.is_file():
            if time.monotonic()>deadline:raise TimeoutError(f'Checkpoint did not appear: {ck}')
            time.sleep(5)
        # Runtime.save publishes numbered checkpoints by atomic rename.
        digest=sha256(ck)
        if sha256(args.dataset/'manifest.json')!=identity['dataset_manifest_sha256']:
            raise ValueError('Evaluation dataset manifest changed')
        if (dest/'result.json').exists():
            prior=json.loads((dest/'result.json').read_text())
            if prior.get('checkpoint_sha256')!=digest:raise ValueError('Previously evaluated checkpoint changed')
            continue
        dest.mkdir(parents=True,exist_ok=True)
        commands=[
            [sys.executable,'-m','supermimic.acp.evaluate','--ckpt',str(ck),
             '--motion_dir',str(args.dataset/'motions'),'--contacts_dir',str(args.dataset/'contacts'),
             '--out',str(dest/'reference.json')],
            [sys.executable,'-m','tools.evaluate_tracker_standing','--checkpoint',str(ck),
             '--motion-dir',str(args.dataset/'motions'),'--contacts-dir',str(args.dataset/'contacts'),
             '--out',str(dest/'standing.json')],
        ]
        for index,command in enumerate(commands):
            with (dest/f'process_{index}.log').open('a') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode:raise RuntimeError(f'Evaluation exited {result.returncode}; see {dest}/process_{index}.log')
        gate=check(json.loads((dest/'reference.json').read_text()),manifest,ck)
        gate.update(report_sha256=sha256(dest/'reference.json'),manifest_sha256=identity['dataset_manifest_sha256'])
        json_save(dest/'reference_gate.json',gate)
        standing=json.loads((dest/'standing.json').read_text())
        if sha256(ck)!=digest or standing.get('checkpoint_sha256')!=digest:raise ValueError('Checkpoint identity changed')
        result=dict(iteration=iteration,checkpoint=str(ck),checkpoint_sha256=digest,
                    reference_passed=gate['passed'],reference_successful_fraction=gate['successful_fraction'],
                    standing_passed=standing['passed'],standing_success_rate=standing['success_rate'],
                    passed=gate['passed'] and standing['passed'],scope='Stage tracking and standing only; no aiming acceptance')
        json_save(dest/'result.json',result);print(json.dumps(result),flush=True)
    json_save(args.out/'worker_complete.json',dict(evaluated=args.iterations,scope='Evaluation completion is not policy acceptance'))


if __name__=='__main__':main()
