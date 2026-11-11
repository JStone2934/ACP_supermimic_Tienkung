"""Sequential full-data execution probes under one bounded server unit."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from tools.run_stage1_training import verify_checkpoint


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_profile_entry(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('profiling insertion no longer matches training source: '+old)
        source = source.replace(old,new)
    replace('import argparse', 'import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd()))\nimport argparse')
    replace('dev = rt.device', 'dev = rt.device\nfrom tools.stage1_phase_profile import PhaseProfile\nphase_profile = PhaseProfile(rt)')
    for fn in ('wm_update', 'policy_update', 'collect'):
        replace(f'def {fn}(', f'@phase_profile.section("{fn}")\ndef {fn}(')
    replace('for it in range(start_it, args.iterations):',
            'for it in range(start_it, args.iterations):\n    phase_profile.iteration(it-start_it)')
    replace('critic_input_c = torch.compile(ctc.critic_input)',
            'critic_input_c = torch.compile(ctc.critic_input)\n'
            'reward_batch = phase_profile.section("reward_forward")(reward_batch)\n'
            'rt.step = phase_profile.section("optimizer_and_sync")(rt.step)\n'
            'torch.autograd.backward = phase_profile.section("backward")(torch.autograd.backward)')
    replace('rt.close()', 'phase_profile.finish()\nrt.close()')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    root=Path(cfg['out']);root.mkdir(parents=True,exist_ok=True)
    source=Path('supermimic/acp/train_stage1.py')
    if sha(source)!=cfg['source_sha256'] or sha(cfg['checkpoint'])!=cfg['checkpoint_sha256']:
        raise ValueError('probe source/checkpoint binding changed')
    if any(sha(p)!=h for p,h in cfg['source_hashes'].items()):
        raise ValueError('bound training sources changed')
    # No second long-running trainer may share these GPUs during comparison.
    status=subprocess.run(['systemctl','--user','is-active',cfg['production_unit']],capture_output=True,text=True)
    if status.stdout.strip()=='active':
        raise RuntimeError('stop the production owner at its saved checkpoint before probing')
    frozen=root/'frozen_profiled_stage1.py';frozen.write_text(frozen_profile_entry(source.read_text()))
    reports={}
    for chunk in cfg['chunks']:
        name=f'chunk_{chunk}';out=root/name;done=out/'verified.json'
        if done.exists():
            reports[name]=json.loads(done.read_text());continue
        if out.exists():
            raise RuntimeError(f'incomplete probe {out}; inspect before retry')
        command=list(cfg['base_command'])
        pos=command.index('--module');command[pos:pos+2]=[str(frozen.resolve())]
        def flag(key,value):
            if key in command:
                i=command.index(key);command[i+1]=str(value)
            else:command.extend([key,str(value)])
        for key,value in dict(log_dir=out.resolve(),resume=cfg['checkpoint'],iterations=cfg['start_iteration']+75,
                              reward_chunk_size=chunk,benchmark_warmup=25,benchmark_steps=50,
                              log_every=25,ckpt_every=cfg['start_iteration']+75).items():
            flag('--'+key,value)
        if '--allow_contact_geometry_change' in command:command.remove('--allow_contact_geometry_change')
        (root/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        state=dict(time=dt.datetime.now().astimezone().isoformat(),stage='execution_benchmark',state='running',
                   candidate=name,production_checkpoint=cfg['checkpoint'],run_dir=str(root))
        (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        Path('docs/stage1/progress.md').write_text('# Stage1 当前进度\n\n'+
            f"北京时间 {state['time']}，正在单独测量 `{name}`，正式训练已在完整第 {cfg['start_iteration']} 轮断点停止并保留。\n\n"+
            '全部 2960 个样本、512 环境、256/256 batch、32 步 horizon 和优化器更新次数保持相同。\n\n'+
            f"测试目录：`{root}`。测试与正式训练不同时运行，测试结果通过验证后再恢复正式任务。\n")
        print(f'{name}: start',flush=True)
        with (root/(name+'.stdout.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed ({result.returncode}); see log')
        expected={k:v+cfg['expected_increment'][k] for k,v in cfg['source_updates'].items()}
        ck=verify_checkpoint(out/'last.pt',cfg['start_iteration']+75,cfg['data_hash'],expected,'spheres224')
        rows=list(csv.DictReader((out/'train_log.csv').open()))
        if not rows or not all(math.isfinite(float(x)) for row in rows for x in row.values()):
            raise ValueError('nonfinite or missing probe log')
        benchmark=json.loads((out/'benchmark.json').read_text())
        if benchmark['updates']!=dict(wm=500,pi=50,critic=1000):raise ValueError('benchmark update counts changed')
        report=dict(time=dt.datetime.now().astimezone().isoformat(),chunk_size=chunk,checkpoint=ck,
                    seconds_per_iteration=benchmark['elapsed_s']/50,benchmark=benchmark)
        done.write_text(json.dumps(report,indent=2)+'\n')
        reports[name]=report
        (root/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(f'{name}: verified {report["seconds_per_iteration"]:.4f} s/it',flush=True)


if __name__=='__main__':main()
