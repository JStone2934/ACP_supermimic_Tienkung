"""Read-only minute health snapshot; never restarts a live training process."""
import argparse,json,time,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default=str(root/'logs/tienkung3_29dof/standing_aiming_curriculum/continuous_20261111'));args=ap.parse_args()
r=Path(args.run_dir).resolve()
logs=sorted(r.glob('attempt_*.log'),key=lambda p:p.stat().st_mtime)
last=None
if logs:
    with logs[-1].open('rb') as f:
        f.seek(max(0,logs[-1].stat().st_size-32768))
        for line in f.read().decode(errors='replace').splitlines():
            if line.startswith('{"iteration"'):
                try:last=json.loads(line)
                except ValueError:pass
service=subprocess.check_output(['systemctl','--user','show','aiming-training.service','-p','ActiveState','-p','MainPID','-p','MemoryCurrent','-p','NRestarts','-p','ControlGroup'],text=True)
props=dict(line.split('=',1) for line in service.splitlines() if '=' in line)
cg=Path('/sys/fs/cgroup')/props.get('ControlGroup','').lstrip('/')
events=(cg/'memory.events').read_text() if (cg/'memory.events').exists() else None
value=dict(checked_at=time.time(),service=props,latest_logged_update=last,log=str(logs[-1]) if logs else None,log_age_seconds=time.time()-logs[-1].stat().st_mtime if logs else None,memory_events=events)
tmp=r/'health.json.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(r/'health.json')
