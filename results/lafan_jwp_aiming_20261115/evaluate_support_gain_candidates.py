"""Bounded physical gain calibration on development seeds; no automatic curriculum promotion."""
import argparse,fcntl,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--frozen-runner',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(checkpoints=[str(x.resolve()) for x in a.checkpoints],seeds=[17,53],devices=['cuda:0','cuda:1'],
                      fixed_count=1,random_count=100,seconds=30,stop_after_first_passing_gain=True,
                      scope='Balance-stage development calibration; no final aiming or moving acceptance.')
        (a.out/'worker.json').write_text(json.dumps(manifest,indent=2));results=[];selected=None
        for checkpoint in a.checkpoints:
            meta=json.loads(checkpoint.with_suffix('.json').read_text());gain=meta['calibration']['gain']
            digest=sha256(checkpoint)
            if digest!=meta['checkpoint_sha256']:raise ValueError('Candidate identity changed')
            out=a.out/checkpoint.stem;out.mkdir(exist_ok=True)
            def evaluate(pair):
                seed,device=pair;report=out/f'seed_{seed}.json'
                if not report.exists():
                    cmd=[sys.executable,'-m','tools.evaluate_wide_stage','--checkpoint',str(checkpoint.resolve()),
                         '--out',str(report.resolve()),'--level','0','--seed',str(seed),'--device',device]
                    with (out/f'seed_{seed}.log').open('a') as log:
                        status=subprocess.run(cmd,cwd=a.frozen_runner,stdout=log,stderr=subprocess.STDOUT)
                    if status.returncode:raise RuntimeError(f'Evaluation failed: {report}')
                r=json.loads(report.read_text())
                if r['checkpoint_sha256']!=digest or r['seed']!=seed or r['level']!=0:
                    raise ValueError('Physical report identity mismatch')
                return r
            with ThreadPoolExecutor(max_workers=2) as pool:
                reports=list(pool.map(evaluate,[(17,'cuda:0'),(53,'cuda:1')]))
            assert sha256(checkpoint)==digest
            row=dict(gain=gain,axes=meta['calibration'].get('axes','all'),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest,
                     passed=all(r['passed'] for r in reports),
                     reports=[{k:r[k] for k in ('seed','passed','fixed_grid_all_pass','random_success_rate')} for r in reports],
                     scope=manifest['scope'])
            (out/'result.json').write_text(json.dumps(row,indent=2));results.append(row)
            print(json.dumps(row),flush=True)
            if row['passed']:
                selected=str(checkpoint.resolve());break
        (a.out/'complete.json').write_text(json.dumps(dict(results=results,selected_checkpoint=selected,
            not_evaluated=[str(x.resolve()) for x in a.checkpoints[len(results):]],
            final_policy_acceptance=False,scope=manifest['scope']),indent=2))


if __name__=='__main__':main()
