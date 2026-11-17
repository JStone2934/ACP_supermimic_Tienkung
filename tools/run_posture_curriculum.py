"""Single-owner unlimited curriculum; finite evaluation intervals, no run budget."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from supermimic.acp.robots import sha256
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import validate_checkpoint
from supermimic.acp.networks import Critic
from supermimic.acp.aiming_fullbody_io import save_checkpoint
from supermimic.acp.posture_curriculum import RESET_PROBABILITIES,conditions,next_phase,validate_evidence


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def actor_count(ck):return ck.get('distributed',{}).get('updates',{}).get('pi',0)


def matches_phase(ck,state):
    return (ck['config'].get('formal_phase_id')==state['phase_id'] and
            ck['config']['curriculum_level']==state['level'] and
            ck['config']['reset_mixture']['probability']==RESET_PROBABILITIES[state['reset_index']])


def next_target(previous_target,actual_count,interval=100):
    if actual_count<previous_target:raise ValueError('Cannot advance an unfinished block')
    return actual_count+interval


def prepare_transition(ck,source,new_level,new_index,phase_id,dest,reports):
    old_level=ck['config']['curriculum_level']
    old_index=RESET_PROBABILITIES.index(ck['config']['reset_mixture']['probability'])
    if next_phase(old_level,old_index)!=(new_level,new_index) or phase_id!=ck['config']['formal_phase_id']+1:
        raise ValueError('Cannot skip a curriculum phase')
    evidence=[json.loads(Path(p).read_text()) for p in reports]
    for condition in conditions(old_level,old_index):
        validate_evidence(source,[r for r in evidence if r.get('reset_pose_condition')==condition],old_level,condition)
    cfg=dict(ck['config']);cfg['reset_mixture']=dict(cfg['reset_mixture'],probability=RESET_PROBABILITIES[new_index])
    cfg.update(curriculum_level=new_level,formal_phase_id=phase_id,
               switch_interval_steps=500 if new_level==6 else None)
    task=WideAimingTask('cpu',cfg)
    critic=Critic(task.critic_dim)
    ck=dict(ck,config=cfg,task_config=task.metadata(),critic=critic.state_dict(),
            critic_warmup_done=0,model_warm_until=ck['iteration']+100,
            source_checkpoint=dict(path=str(source.resolve()),sha256=sha256(source)))
    # Preserve actor/model optimizers and update counts; refresh value regression
    # when the reward stage or reset distribution changes.
    ck['optimizers']=[ck['optimizers'][0],ck['optimizers'][1],torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr']).state_dict()]
    ck['curriculum_transition']=dict(previous_level=old_level,previous_reset_index=old_index,
        evidence=[dict(path=str(Path(p).resolve()),sha256=sha256(p)) for p in reports])
    validate_checkpoint(ck,task);save_checkpoint(dest,ck)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,default=29788)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner=out/'runner';statefile=out/'controller_state.json'
    # Immutable source snapshot: restart must not silently change algorithms.
    manifest=json.loads((out/'runner_manifest.json').read_text())
    for name,digest in manifest.items():
        if sha256(runner/name)!=digest:raise ValueError('Frozen runner changed: '+name)
    state=(json.loads(statefile.read_text()) if statefile.exists() else
           dict(phase='training',level=0,reset_index=0,phase_id=0,actor_target=100,
                checkpoint=str(out/'init.pt'),best={},completed_blocks=0))
    def save(**updates):
        state.update(updates);state['updated_at_unix']=time.time();write_json(statefile,state)
        write_json(out/'progress.json',dict(state,training_status_path=str(out/'train/status.json'),total_iteration_limit=None))
    def invoke(cmd,log):
        with open(log,'a') as f:subprocess.run(cmd,cwd=runner,stdout=f,stderr=subprocess.STDOUT,check=True)
    def evaluate(candidate,condition,final=False):
        seeds=[101,211] if final else [17,53];digest=sha256(candidate)
        folder=out/'evaluations'/f'phase{state["phase_id"]}_actor{state["actor_target"]}'/('final' if final else condition)
        folder.mkdir(parents=True,exist_ok=True);jobs=[];paths=[]
        for gpu,seed in enumerate(seeds):
            path=folder/f'seed{seed}.json';paths.append(path)
            if path.exists():
                r=json.loads(path.read_text())
                if r.get('checkpoint_sha256')==digest and r.get('reset_pose_condition')==condition and r.get('level')==(None if final else state['level']):continue
                raise ValueError('Existing report identity mismatch: '+str(path))
            cmd=[sys.executable,'-m','tools.evaluate_posture_curriculum','--checkpoint',str(candidate),'--out',str(path),'--seed',str(seed),'--device',f'cuda:{gpu}','--reset-pose',condition]
            if final:cmd+=['--final']
            log=open(path.with_suffix('.log'),'a');jobs.append((subprocess.Popen(cmd,cwd=runner,stdout=log,stderr=subprocess.STDOUT),log))
        for child,log in jobs:
            rc=child.wait();log.close()
            if rc:raise RuntimeError('Physical evaluation exited '+str(rc))
        reports=[json.loads(path.read_text()) for path in paths]
        try:validate_evidence(candidate,reports,state['level'],condition,final=final);passed=True
        except ValueError:passed=False
        return passed,paths,reports
    try:
        while state['phase']!='complete':
            checkpoint=Path(state['checkpoint']);last=out/'train/last.pt'
            recent=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
            if recent is not None and matches_phase(recent,state):ck=recent;checkpoint=last
            else:ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if not matches_phase(ck,state):raise ValueError('Controller/checkpoint phase mismatch')
            if actor_count(ck)<state['actor_target']:
                save(phase='training',checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
                mode='--resume' if ck['stage']=='training' else '--init'
                invoke([sys.executable,'-m','torch.distributed.run','--nproc_per_node=2',f'--master_port={a.port}',
                    '-m','supermimic.acp.train_wide_aiming',mode,str(checkpoint),'--log-dir',str(out/'train'),
                    '--iterations','0','--actor-update-target',str(state['actor_target']),
                    '--keep-last-checkpoints','5','--compile-training'],out/'training.log')
                ck=torch.load(last,map_location='cpu',weights_only=False)
            candidate=out/'train'/f'ckpt_{ck["iteration"]:06d}.pt'
            if not candidate.exists():
                shutil.copyfile(last,str(candidate)+'.tmp');os.replace(str(candidate)+'.tmp',candidate)
            if actor_count(ck)<state['actor_target']:raise ValueError('Trainer exited before evaluation boundary')
            save(phase='evaluating',checkpoint=str(candidate),checkpoint_sha256=sha256(candidate),actual_actor_updates=actor_count(ck),iteration=ck['iteration'])
            passed=True;paths=[];score=0
            for condition in conditions(state['level'],state['reset_index']):
                ok,pp,rr=evaluate(candidate,condition);passed &= ok;paths+=pp
                score+=sum(sum(s['passed'] for s in r['scenarios']) for r in rr)
            key=str(state['phase_id']);best=state['best'].get(key)
            if best is None or score>best['score']:
                dest=out/'best'/f'phase{key}.pt';dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(candidate,str(dest)+'.tmp');os.replace(str(dest)+'.tmp',dest)
                state['best'][key]=dict(path=str(dest),score=score,sha256=sha256(dest),reports=[str(x) for x in paths])
            nxt=next_phase(state['level'],state['reset_index'])
            if passed and nxt is None:
                save(phase='final_evaluation')
                passed,pp,rr=evaluate(candidate,'reference',final=True);paths+=pp
                if passed:
                    dest=out/'accepted.pt';shutil.copyfile(candidate,dest)
                    save(phase='complete',accepted_checkpoint=str(dest),reports=[str(x) for x in paths],
                         original_task_acceptance=True,posture_hard_acceptance=False);break
            history=out/'reviews.jsonl'
            with history.open('a') as f:f.write(json.dumps(dict(phase_id=state['phase_id'],level=state['level'],reset_index=state['reset_index'],actor_updates=actor_count(ck),passed=passed,reports=[str(x) for x in paths]))+'\n')
            if passed:
                # All evidence has been checked against this exact checkpoint.
                level,index=nxt;phase_id=state['phase_id']+1
                milestone=out/'milestones'/f'phase{state["phase_id"]}.pt';milestone.parent.mkdir(exist_ok=True);shutil.copyfile(candidate,milestone)
                transition=out/'transitions'/f'phase{phase_id}.pt';transition.parent.mkdir(exist_ok=True)
                prepare_transition(ck,milestone,level,index,phase_id,transition,paths)
                state.setdefault('milestone_reports',[]).extend(str(x) for x in paths)
                save(phase='training',level=level,reset_index=index,phase_id=phase_id,checkpoint=str(transition),
                     actor_target=next_target(state['actor_target'],actor_count(ck)),completed_blocks=state['completed_blocks']+1)
            else:
                save(phase='training',checkpoint=str(candidate),actor_target=next_target(state['actor_target'],actor_count(ck)),
                     completed_blocks=state['completed_blocks']+1,last_review_passed=False)
            # Trajectories are large; reports survive, recent five block folders
            # and all best/milestone evidence remain inspectable.
            protected={str(Path(x).parent.parent) for b in state['best'].values() for x in b['reports']}
            protected.update(str(Path(x).parent.parent) for x in state.get('milestone_reports',[]))
            blocks=sorted((out/'evaluations').glob('phase*_actor*'),key=lambda x:x.stat().st_mtime)
            for block in blocks[:-5]:
                if str(block) not in protected:
                    for trajectory in block.rglob('*.npz'):trajectory.unlink()
    except Exception as exc:
        save(phase='error',error=repr(exc));raise


if __name__=='__main__':main()
