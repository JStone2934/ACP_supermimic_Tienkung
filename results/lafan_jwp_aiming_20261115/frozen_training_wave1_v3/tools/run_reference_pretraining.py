"""Own one immutable ACP reference-training run and resume its last checkpoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,type=Path)
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    out=Path(cfg['run_dir']);out.mkdir(parents=True,exist_ok=True)
    with (out/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,expected in cfg['bound_files'].items():
            if sha(path)!=expected:raise ValueError(f'Run input changed: {path}')
        command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2',
                 '--master_addr=127.0.0.1',f"--master_port={cfg['master_port']}", '--max_restarts=0',
                 '-m','supermimic.acp.train_stage1',*cfg['train_arguments'],'--log_dir',str(out/'train')]
        last=out/'train/last.pt'
        if last.exists():command+=['--resume',str(last)]
        (out/'launch.json').write_text(json.dumps(dict(command=command,controller_pid=os.getpid(),
                                                     source_config=str(args.config.resolve())),indent=2))
        result=subprocess.run(command,check=False)
        (out/'process_result.json').write_text(json.dumps(dict(exit_code=result.returncode,
            scope='Process completion is not policy acceptance'),indent=2))
        raise SystemExit(result.returncode)


if __name__=='__main__':main()
