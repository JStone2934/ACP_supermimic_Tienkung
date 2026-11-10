"""Refine provenance-bound complete candidates, then rerun every quality gate."""
from __future__ import annotations
import argparse,concurrent.futures,fcntl,json,multiprocessing,os,shutil,time,traceback
from pathlib import Path
import numpy as np
from supermimic.acp.php_conversion import (VERSION,COLLISION_POLICY,atomic_json,atomic_npz,digest)
from supermimic.acp.native_pose_polish import polish
from supermimic.acp.motion import load_php_npz
from supermimic.acp.robots import ROOT,get_robot,sha256
from supermimic.acp.terrain import load_terrain,terrain_path
from supermimic.acp.retarget_fidelity import POLICY as FIDELITY_POLICY,POSTURE_POLICY
from supermimic.acp.root_continuity import POLICY as ROOT_POLICY
from supermimic.acp.source_contacts import SOURCE_CONTACT_POLICY
from supermimic.acp.contact_intent import POLICY as CONTACT_INTENT_POLICY,PATCH_POLICY
from tools.convert_php import load_json,verify_task,source_fingerprint,output_hashes,aggregate
from tools.check_php_conversion import evaluate_task,collision_evidence

FIXED_ROOT_POLISH_POLICY=dict(version='native-joint-polish-fixed-root-v1',trigger_m=.002,
                  target_penetration_m=.0015,max_nfev=120,root_position_bound_m=0.,joint_change_bound_rad=.4,
                  scope='Native pose refinement of collision frames; complete root poses and other frames exact; all original gates must pass after refinement')

POLISH_POLICY=dict(version='native-joint-polish-fixed-root-velocity-v1',trigger_m=.002,
                  target_penetration_m=.0015,max_nfev=120,root_position_bound_m=0.,joint_change_bound_rad=.4,
                  velocity_stencil='numpy-gradient-first-order-edges',velocity_margin_ratio=1e-7,
                  scope='Velocity-bounded native pose refinement of collision frames; complete root poses and other frames exact; all original gates must pass after refinement')

LEGACY_POLISH_POLICY={'version': 'native-pose-polish-fixed-root-rotation-v1', 'trigger_m': 0.002, 'target_penetration_m': 0.0015, 'max_nfev': 120, 'root_position_bound_m': 0.1, 'joint_change_bound_rad': 0.4, 'scope': 'Native pose refinement of collision frames; root quaternions and other frames exact; all original gates must pass after refinement'}
SUPPORTED_POLISH_POLICIES=(LEGACY_POLISH_POLICY,FIXED_ROOT_POLISH_POLICY,POLISH_POLICY)

def seed_receipt(parent,task_id):
    parent=Path(parent).resolve();plan=load_json(parent/'plan.json')
    task=next(t for t in plan['tasks'] if t['id']==task_id)
    state=load_json(parent/'tasks'/(task_id+'.json'))
    if state['status'] not in ('accepted','rejected'):raise ValueError('seed has no committed quality result')
    verify_task(parent,state)
    robot=get_robot('tienkung3_29dof');robot.validate_metadata(plan['robot_metadata'])
    for field,policy in [('fidelity_policy',FIDELITY_POLICY),('posture_policy',POSTURE_POLICY),
                         ('source_contact_validation_policy',SOURCE_CONTACT_POLICY),('collision_policy',COLLISION_POLICY)]:
        if plan.get(field)!=policy:raise ValueError('seed policy mismatch: '+field)
    for relative,checksum in plan['source_robot_files'].items():
        if sha256(ROOT/relative)!=checksum:raise ValueError('source robot changed')
    record=task['record'];candidate=parent/'candidates'/(record['key']+'_motion.npz')
    quality=load_json(parent/state['quality_path'])
    if quality['source']!=record or quality['motion_hash']!=sha256(candidate):raise ValueError('seed quality identity mismatch')
    for path,expected in [(record['source'],record['source_hash']),(record['labels'],record['labels_hash']),
                          (terrain_path(record['source']),record['terrain_hash'])]:
        if sha256(path)!=expected:raise ValueError('original source asset changed')
    receipt=dict(parent=str(parent),parent_task_id=task_id,parent_plan_sha256=sha256(parent/'plan.json'),
                 parent_identity=plan['identity'],quality_sha256=state['quality_hash'],motion_sha256=sha256(candidate),
                 intent_sha256=sha256(str(candidate).replace('_motion.npz','_intent.npz')),
                 terrain_sha256=sha256(terrain_path(str(candidate))))
    return receipt,task,plan

def verify_seed(task):
    receipt,parent_task,_=seed_receipt(task['seed']['parent'],task['seed']['parent_task_id'])
    if receipt!=task['seed'] or parent_task['record']!=task['record'] or parent_task['solve_variant']!=task['solve_variant']:
        raise ValueError('seed provenance changed')
    return Path(receipt['parent'])/'candidates'/(task['record']['key']+'_motion.npz')

def refine(task,root):
    started=time.monotonic();root=Path(root);seed=verify_seed(task);robot=get_robot('tienkung3_29dof')
    p,quat,q,fps=load_php_npz(seed,robot);record=task['record'];motion=np.concatenate((p,quat,q),axis=1)
    if len(motion)!=record['frames'] or fps!=50:raise ValueError('seed frame/fps mismatch')
    with np.load(str(seed).replace('_motion.npz','_intent.npz')) as z:intent={k:z[k].copy() for k in z.files}
    v=task['solve_variant'];result,stats=polish(robot,motion,load_terrain(str(seed))[v],intent['contact'][v],intent['offsets'][v])
    dest=root/'candidates'/(record['key']+'_motion.npz')
    atomic_npz(dest,joint_pos=result,joint_names=np.array(robot.joint_names),fps=[fps],robot_metadata=json.dumps(robot.metadata()),
               data_version=VERSION,source=record['source'],source_frames=record['source_frames'],task_id=task['id'],
               seed_receipt=json.dumps(task['seed']),native_polish_policy=json.dumps(POLISH_POLICY))
    shutil.copyfile(terrain_path(record['source']),terrain_path(str(dest)))
    intent['task_id']=np.array(task['id'])
    atomic_npz(str(dest).replace('_motion.npz','_intent.npz'),**intent)
    return dict(native_polish=stats,solve_seconds=time.monotonic()-started,candidate_files=output_hashes(root,task))

def audit_seed_provenance(root,plan):
    root=Path(root)
    declared=plan.get('native_polish_policy')
    if declared not in SUPPORTED_POLISH_POLICIES:raise ValueError('missing or unsupported native polish policy')
    fixed_root=declared['version'] in ('native-joint-polish-fixed-root-v1','native-joint-polish-fixed-root-velocity-v1')
    robot=get_robot('tienkung3_29dof')
    for task in plan['tasks']:
        seed=verify_seed(task);state=load_json(root/'tasks'/(task['id']+'.json'))
        candidate=root/'candidates'/(task['record']['key']+'_motion.npz')
        p,r,q,fps=load_php_npz(seed,robot);cp,cr,cq,cfps=load_php_npz(candidate,robot)
        if fps!=cfps or cp.shape!=p.shape or not np.array_equal(cr,r) or (fixed_root and not np.array_equal(cp,p)):raise ValueError('polish changed timing/root pose')
        selected=state['native_polish']['selected_frames'];mask=np.ones(len(p),bool);mask[selected]=False
        if not np.array_equal(cp[mask],p[mask]) or not np.array_equal(cq[mask],q[mask]):raise ValueError('polish changed unselected frames')
        with np.load(candidate) as z:
            if json.loads(str(z['seed_receipt'].item()))!=task['seed'] or json.loads(str(z['native_polish_policy'].item()))!=declared:raise ValueError('candidate provenance missing')
        with np.load(str(seed).replace('_motion.npz','_intent.npz')) as a,np.load(str(candidate).replace('_motion.npz','_intent.npz')) as b:
            if set(a.files)!=set(b.files):raise ValueError('intent fields changed')
            for k in a.files:
                if k!='task_id' and not np.array_equal(a[k],b[k]):raise ValueError('source intent changed: '+k)
    return len(plan['tasks'])

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seeds',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--scan',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args(argv)
    if args.workers<1:ap.error('workers must be positive')
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True);lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seeds=load_json(args.seeds);items=[seed_receipt(s['parent'],s['task_id']) for s in seeds]
    if len({t['record']['key'] for _,t,_ in items})!=len(items):raise ValueError('duplicate output key')
    identity=dict(version=VERSION,kind='native_pose_polish',native_polish_policy=POLISH_POLICY,code=source_fingerprint(),
                  robot_metadata=get_robot('tienkung3_29dof').metadata(),source_robot_files=items[0][2]['source_robot_files'],
                  collision_policy=COLLISION_POLICY,fidelity_policy=FIDELITY_POLICY,posture_policy=POSTURE_POLICY,
                  contact_intent_policy=CONTACT_INTENT_POLICY,contact_patch_policy=PATCH_POLICY,
                  source_contact_validation_policy=SOURCE_CONTACT_POLICY,root_continuity_policy=ROOT_POLICY)
    tasks=[dict(id=digest([identity,receipt,t['record'],t['solve_variant']])[:24],record=t['record'],solve_variant=t['solve_variant'],seed=receipt) for receipt,t,_ in items]
    plan=dict(run_id=root.name,identity=digest([identity,[t['id'] for t in tasks]]),**identity,tasks=tasks,execution=dict(cpu_workers=args.workers,simulation_environments=0))
    if (root/'plan.json').exists():
        if not args.resume or load_json(root/'plan.json')['identity']!=plan['identity']:raise ValueError('existing plan requires unchanged resume')
        plan=load_json(root/'plan.json')
    else:atomic_json(root/'plan.json',plan)
    if args.scan:print(json.dumps(dict(sources=len(tasks),frames=sum(t['record']['frames'] for t in tasks),slots=sum(t['record']['variants'] for t in tasks))),flush=True);return
    for relative,checksum in plan['code'].items():
        source=ROOT/relative
        if sha256(source)!=checksum:raise ValueError('runtime code changed')
        target=root/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    atomic_json(root/'collision_policy.json',collision_evidence())
    states={}
    for t in tasks:
        path=root/'tasks'/(t['id']+'.json');s=load_json(path) if path.exists() else dict(task_id=t['id'],status='pending')
        if s['status'] in ('accepted','rejected'):verify_task(root,s)
        elif s['status']=='failed':raise ValueError('diagnose failed task before choosing a new run')
        else:s['status']='pending'
        states[t['id']]=s;atomic_json(path,s)
    aggregate(root,plan,states)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={};todo=iter([t for t in tasks if states[t['id']]['status']=='pending'])
        def submit_next():
            t=next(todo,None)
            if t is None:return
            s=states[t['id']];s['status']='solving';atomic_json(root/'tasks'/(t['id']+'.json'),s)
            pending[pool.submit(refine,t,root)]=(t,'refine')
        for _ in range(args.workers):submit_next()
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                t,stage=pending.pop(f);s=states[t['id']]
                try:
                    s.update(f.result())
                    if stage=='refine':s['status']='checking';pending[pool.submit(evaluate_task,t,root)]=(t,'check')
                    else:verify_task(root,s);print(s['status']+': '+t['record']['key'],flush=True);submit_next()
                except Exception:s.update(status='failed',error=traceback.format_exc());submit_next()
                atomic_json(root/'tasks'/(t['id']+'.json'),s)
            print(json.dumps(aggregate(root,plan,states)['counts']),flush=True)
    if any(s['status']=='failed' for s in states.values()):raise SystemExit(2)
    audit_seed_provenance(root,plan)

if __name__=='__main__':main()
