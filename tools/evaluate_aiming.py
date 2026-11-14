"""Independent single-actor evaluation; no IK or standing teacher is loaded."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming import setup,policy,json_save,AimingSim
from supermimic.acp.aiming_curriculum import sample_parameters,summarize
from supermimic.acp.env import TrackingSim
from supermimic.acp.mdp import flatten_state


def fixed_parameters(task,central=False):
    rng=np.random.default_rng(20261111);dev=task.alpha.device
    c=dict(mode=0,depth=(3,3),angle=(0,0),rho=.2 if central else .05)
    p=sample_parameters(c,1 if central else 9,rng,dev);p[:,14]=1 if central else 5
    if not central:
        for i,(L,a,b) in enumerate([(3,0,0),(3,15,0),(3,-15,0),(3,0,10),(3,0,-10),(2,0,0),(4,0,0)]):
            p[i,1]=L;p[i,4:6]=torch.tensor(np.deg2rad([a,b]),device=dev)
        p[7,0]=1;p[7,14]=2
        nodes=p[7,16:].reshape(8,6)
        for j in range(5):nodes[j,:3]=torch.tensor([j*6,np.deg2rad(10*(j%2)),0],device=dev)
        p[8,0]=2;p[8,6]=np.deg2rad(5);p[8,8]=1;p[8,14]=2
    return p

@torch.no_grad()
def evaluate(student,task,out,central=False,params=None):
    fixed_suite=params is None and not central
    params=fixed_parameters(task,central) if params is None else params
    n=len(params);dev=task.alpha.device
    base=TrackingSim(n,task.mdp.motion,task.mdp,str(dev),eval_mode=True,control_task=task)
    env=AimingSim(base,task,lambda _:params);hist=torch.zeros(n,10,99,device=dev);gs=torch.full((n,),50,device=dev)
    states=[tuple(x.clone() for x in base.state())];goals=[env.goal().clone()];actions=[]
    end=torch.full((n,),1500,dtype=torch.long,device=dev);terminated=torch.zeros(n,dtype=torch.bool,device=dev)
    old=student.training;student.eval()
    for t in range(1500):
        o=base.observe();hist=o[:,None].repeat(1,10,1) if t==0 else torch.cat((hist[:,1:],o[:,None]),1)
        raw=student(hist,task.command(base.state(),base.f,base.v,gs,env.goal(),env.previous))
        if not torch.isfinite(raw).all():raise FloatingPointError('non-finite actor action')
        tr=env.step(raw);states.append(tuple(x.clone() for x in base.state()));goals.append(env.goal().clone());actions.append(tr['a'].cpu())
        first=tr['term']&~terminated;end[first]=t+1;terminated|=tr['term']
        if bool(terminated.all()):break
    reports=[]
    for i in range(n):
        e=int(end[i])+1;seq=[tuple(x[i] for x in S) for S in states[:e]];g=[x[i] for x in goals[:e]]
        r=summarize(task,seq,g,params[i].cpu().numpy(),bool(terminated[i]));r['scenario']=i
        r['passed']=bool(r['standing_pass'] and all(r['attempt_success']) and (not fixed_suite or r['post_capture_p95_m']<=.05))
        reports.append(r)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(central=central,fixed_suite=fixed_suite,passed=all(r['passed'] for r in reports),scenarios=reports,task_config=task.metadata())
    json_save(out.with_suffix('.json'),report)
    np.savez_compressed(out.with_suffix('.npz'),s=torch.stack([flatten_state(s) for s in states]).cpu().numpy(),goal=torch.stack(goals).cpu().numpy(),a=torch.stack(actions).numpy(),params=params.cpu().numpy(),end=end.cpu().numpy(),n0=task.n0.cpu().numpy(),e1=task.e1.cpu().numpy(),e2=task.e2.cpu().numpy(),task_config=np.array(json.dumps(task.metadata())))
    student.train(old)
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ckpt',required=True);ap.add_argument('--out',required=True);ap.add_argument('--central',action='store_true');args=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=True
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False);task=setup(ck['config']);pi=policy(task,ck['pi'])
    r=evaluate(pi,task,args.out,args.central);print(json.dumps(r['scenarios'],indent=2))

if __name__=='__main__':main()
