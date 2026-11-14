"""Own one continuous training child; poll it without tying its lifetime to SSH."""
import argparse,json,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default=str(ROOT/'logs/tienkung3_29dof/standing_aiming_curriculum/continuous_20261111'));args=ap.parse_args()
RUN=Path(args.run_dir).resolve()
RUN.mkdir(parents=True,exist_ok=True)
lock=(RUN/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
train=RUN/'train';attempt=0
while True:
    resume=train/'last.pt'
    cmd=[sys.executable,'-m','supermimic.acp.train_stage1','--task','standing_aiming_curriculum','--aim_config',str(RUN/'config.json'),'--log_dir',str(train)]
    cmd+=['--resume',str(resume)] if resume.exists() else ['--finetune',str(ROOT/'data/tienkung3_29dof/standing_aiming/ik_supervision/20261111_c0/best.pt')]
    attempt+=1
    with (RUN/f'attempt_{attempt:04d}_{int(time.time())}.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while True:
            code=child.poll()
            status=dict(pid=child.pid,exit_code=code,checked_at=time.time(),attempt=attempt,checkpoint=str(resume),checkpoint_mtime=resume.stat().st_mtime if resume.exists() else None)
            tmp=RUN/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(RUN/'status.json')
            if code is not None:break
            time.sleep(60)
    time.sleep(60)
