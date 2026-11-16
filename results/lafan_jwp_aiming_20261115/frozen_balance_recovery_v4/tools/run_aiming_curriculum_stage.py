"""Own one frozen, explicitly configured ACP curriculum stage, with optional resource sequencing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['run_dir']);out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,digest in cfg['bound_files'].items():
            if sha256(path)!=digest:raise ValueError(f'Changed input: {path}')
        deadline=time.monotonic()+cfg.get('wait_timeout_hours',6)*3600
        for path in cfg.get('wait_for_process_results',[]):
            result=Path(path)
            (out/'controller_status.json').write_text(json.dumps(dict(status='waiting_for_resources',dependency=path)))
            while not result.exists():
                if time.monotonic()>deadline:raise TimeoutError(f'Dependency not finished: {path}')
                time.sleep(5)
            if json.loads(result.read_text()).get('exit_code')!=0:
                raise RuntimeError(f'Dependency did not finish cleanly: {path}')
        arguments=list(cfg['train_arguments']);last=out/'train/last.pt'
        if last.exists():
            for flag in ('--init','--resume'):
                if flag in arguments:
                    i=arguments.index(flag);del arguments[i:i+2]
            arguments+=['--resume',str(last)]
        command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2',
            '--master_addr=127.0.0.1',f"--master_port={cfg['master_port']}",'--max_restarts=0',
            '-m','supermimic.acp.train_wide_aiming',*arguments,'--log-dir',str(out/'train')]
        (out/'launch.json').write_text(json.dumps(dict(command=command,config=str(a.config.resolve()),controller_pid=os.getpid()),indent=2))
        (out/'controller_status.json').write_text(json.dumps(dict(status='running')))
        result=subprocess.run(command,check=False)
        (out/'process_result.json').write_text(json.dumps(dict(exit_code=result.returncode,scope='Process completion is not policy acceptance')))
        raise SystemExit(result.returncode)


if __name__=='__main__':main()
