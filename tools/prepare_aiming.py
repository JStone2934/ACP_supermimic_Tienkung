"""Generate actual teacher trajectories and initialize the one shared aiming actor.
Run with --smoke for a small pipeline exercise; normal defaults prepare C0 only.
"""
import argparse,json,math,copy
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming import setup,policy,json_save,SOURCE,DEFAULT_CONFIG
from supermimic.acp.aiming_curriculum import sample_parameters
from supermimic.acp.env import TrackingSim
from supermimic.acp.aiming import AimingSim
from supermimic.acp.networks import Policy
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.mdp import flatten_state,unflatten_state
from supermimic.acp.robots import sha256
from supermimic.rotations import mat_to_quat

class Teacher:
    def __init__(self,task):
        self.task=task;kin=task.mdp.kin
        self.bodies=[kin.joint_idx.index(int(j)) for j in task.terminal]
        self.tcp=kin.body_names.index('right_tcp_link')
        self.axes=kin.axes[self.bodies];self.anchors=kin.anchors[self.bodies]
        self.geometry=torch.compile(self._geometry)

    def _geometry(self,S,q,g):
        task=self.task;qs=S[4].scatter(-1,task.terminal.expand_as(q),q)
        bp,bR,_,_=task.mdp.kin.forward(*S[:4],qs,S[5],all_bodies=True)
        h=bp[:,self.tcp];d=-bR[:,self.tcp,:,2];d=d/d.norm(dim=-1,keepdim=True).clamp_min(1e-8);delta=g-h;D=delta.norm(dim=-1,keepdim=True).clamp_min(1e-6);u=delta/D
        axis=(bR[:,self.bodies]@self.axes[...,None]).squeeze(-1)
        anchor=bp[:,self.bodies]+(bR[:,self.bodies]@self.anchors[...,None]).squeeze(-1)
        Jh=torch.linalg.cross(axis,h[:,None]-anchor);Jd=torch.linalg.cross(axis,d[:,None].expand_as(axis))
        J=(Jd+(Jh-u[:,None]*(Jh*u[:,None]).sum(-1,keepdim=True))/D[:,None]).transpose(-1,-2)
        return d-u,J

    @torch.no_grad()
    def solve(self,S,g,previous):
        q=previous.clone();eye=torch.eye(3,device=q.device);iterations=torch.zeros(len(q),device=q.device)
        for retry in range(2):
            damping=.01*10**retry
            for it in range(50):
                r,J=self.geometry(S,q,g);good=r.norm(dim=-1)<=2*math.sin(math.radians(.25)/2)
                if bool(good.all()):break
                grad=(J.transpose(-1,-2)@r[...,None]).squeeze(-1)+1e-4*(q-previous)+1e-5*q
                step=torch.linalg.solve(J.transpose(-1,-2)@J+(damping*damping+1.1e-4)*eye,-grad[...,None]).squeeze(-1).clamp(-.1,.1)
                objective=.5*r.square().sum(-1)+.5e-4*(q-previous).square().sum(-1)+.5e-5*q.square().sum(-1)
                accepted=good.clone()
                for scale in (1.,.5,.25,.125):
                    cand=(q+scale*step).clamp(self.task.mdp.soft_lo[self.task.terminal],self.task.mdp.soft_hi[self.task.terminal])
                    rr,_=self.geometry(S,cand,g)
                    score=.5*rr.square().sum(-1)+.5e-4*(cand-previous).square().sum(-1)+.5e-5*cand.square().sum(-1)
                    take=(score<objective)&~accepted
                    q=torch.where(take[:,None],cand,q);accepted|=take
                iterations+=~good
                if not bool((accepted&~good).any()):break
            if bool(good.all()):break
        r,_=self.geometry(S,q,g)
        error=torch.rad2deg(2*torch.asin((r.norm(dim=-1)/2).clamp(0,1)))
        margin=torch.minimum(q-self.task.mdp.soft_lo[self.task.terminal],self.task.mdp.soft_hi[self.task.terminal]-q).min(-1).values
        valid=(error<=.25)&(margin>=math.radians(1))&torch.isfinite(q).all(-1)
        return q,valid,error,margin,iterations

def history(hist,o,fresh):
    return torch.where(fresh[:,None,None],o[:,None].expand(-1,10,-1),torch.cat((hist[:,1:],o[:,None]),1))

def make_env(task,n,sampler,eval_mode=True):
    b=TrackingSim(n,task.mdp.motion,task.mdp,str(task.alpha.device),control_task=task,eval_mode=eval_mode)
    return AimingSim(b,task,sampler)

def collect(task,ref,config,out,smoke=False):
    rng=np.random.default_rng(config['seed']);dev=task.alpha.device;n=config['sequences'];steps=config['sequence_steps']
    teacher=Teacher(task)
    # Four baseline state sources. Entire source time windows, not adjacent frames, define splits.
    central=dict(mode=0,depth=(3,3),angle=(0,0),rho=.2)
    baseline=make_env(task,4,lambda n:sample_parameters(central,n,rng,dev))
    hist=torch.zeros(4,10,99,device=dev);gs=torch.full((4,),50,device=dev);source=[]
    for t in range(50 if smoke else 1500):
        with torch.no_grad():
            o=baseline.base.observe();hist=history(hist,o,baseline.base.ep_step==0)
            raw=ref(hist,task.ctc.command(baseline.base.state(),baseline.base.f,baseline.base.v,gs,noise=False))
            raw[:,task.terminal]=(-task.mdp.default_q[task.terminal])/task.alpha[task.terminal]
            baseline.step(raw)
            source.append(flatten_state(baseline.base.state()).cpu())
    source=torch.stack(source);torch.save(source,out/'standing_sources.pt')
    # Each sequence has an independent physical rollout. Split source windows before selecting starts.
    nval=max(1,round(n*.2));split=torch.arange(n)>=n-nval
    starts=[];source_ids=[]
    block=max(1,len(source)//5)
    for i in range(n):
        window=4 if split[i] else int(rng.integers(4));src=int(rng.integers(4));t=min(len(source)-1,window*block+int(rng.integers(block)))
        starts.append(source[t,src]);source_ids.append([src,window,t])
    params=[]
    for i in range(n):
        mode=int(rng.choice([0,1,2],p=[.5,.25,.25]));angle=(5,3) if rng.random()<.5 else (15,10)
        c=dict(mode=mode,depth=(2,4),angle=angle,rho=float(rng.choice([.2,.15,.1,.075,.05])),two_dim=True,speed=5,jump=(3,5),dwell=(2,2))
        params.append(sample_parameters(c,1,rng,dev))
    params=torch.cat(params);env=make_env(task,n,lambda _:params)
    S=unflatten_state(torch.stack(starts).to(dev));b=env.base
    ids=torch.arange(n,device=dev)
    b.robot.write_root_link_pose_to_sim(torch.cat((S[0]+b.scene.env_origins,mat_to_quat(S[1])),-1),env_ids=ids)
    b.robot.write_root_link_velocity_to_sim(torch.cat(S[2:4],-1),env_ids=ids)
    b.robot.write_joint_position_to_sim(S[4][:,b.to_internal],env_ids=ids);b.robot.write_joint_velocity_to_sim(S[5][:,b.to_internal],env_ids=ids)
    b.refresh();env.previous=S[4][:,task.terminal].clone()
    hist=torch.zeros(n,10,99,device=dev);gs=torch.full((n,),50,device=dev)
    data={k:[] for k in ('s','hist','goal','previous','q_star','executed_q','body','valid','error_deg','margin','iterations')}
    alive=torch.ones(n,dtype=torch.bool,device=dev)
    root0=b.state()[0].clone();feet0=task.mdp.kin.forward(*b.state())[0][:,task.feet].clone()
    for t in range(steps):
        S=b.state();o=b.observe();hist=history(hist,o,b.ep_step==0);g=env.goal();previous=env.previous.clone()
        qs,valid,error,margin,it=teacher.solve(S,g[:,:3],previous)
        with torch.no_grad():
            a=ref(hist,task.ctc.command(S,b.f,b.v,gs,noise=False));body=a[:,task.body].clone()
            a[:,task.terminal]=(qs-task.mdp.default_q[task.terminal])/task.alpha[task.terminal]
            state=flatten_state(S).clone();obs=hist.clone()
            transition=env.step(a)
            vals=(state,obs,g,previous,qs,env.previous,body,valid&alive,error,margin,it)
            for k,x in zip(data,vals):data[k].append(x.cpu())
            new=b.state();fp=task.mdp.kin.forward(*new)[0][:,task.feet]
            alive &= ~transition['term'] & ((new[0][:,:2]-root0[:,:2]).norm(dim=-1)<=.1) & ((fp[...,:2]-feet0[...,:2]).norm(dim=-1).max(-1).values<=.05)
        if (t+1)%25==0:print(f'collect {t+1}/{steps}, valid={float(valid.float().mean()):.3f}, standing={float(alive.float().mean()):.3f}',flush=True)
    data={k:torch.stack(v).transpose(0,1).contiguous() for k,v in data.items()}
    data.update(validation=split,source_ids=source_ids,params=params.cpu(),source_sha256=sha256(config['source']),robot_metadata=task.mdp.robot.metadata())
    torch.save(data,out/'supervision.pt')
    json_save(out/'dataset.json',dict(sequences=n,steps=steps,candidates=n*steps,valid=int(data['valid'].sum()),validation_sequences=nval,source_sha256=data['source_sha256'],source_ids=source_ids,actual_dynamics=True,geometric_invalid=int((data['error_deg']>.25).sum()),excluded_unstable_frames=int((~data['valid']).sum()),standing_sequences_final=int(alive.sum())))
    return data

def augment(task,data,out,config):
    rng=np.random.default_rng(config['seed']+301);n,T=data['valid'].shape;extra=100
    times=torch.tensor(rng.integers(0,T,(n,extra)));rows=torch.arange(n)[:,None]
    fields=('s','hist','goal','previous','q_star','executed_q','body','valid','error_deg','margin','iterations')
    new={k:data[k][rows,times].clone() for k in fields}
    new['valid'][data['validation']]=False  # keep the original independent validation distribution
    teacher=Teacher(task);ids=new['valid'].flatten().nonzero().flatten();flat={k:v.flatten(0,1) for k,v in new.items()}
    for chunk in ids.split(256):
        S=unflatten_state(flat['s'][chunk].to(task.alpha.device));prev=flat['previous'][chunk].to(task.alpha.device)
        par=sample_parameters(dict(mode=0,depth=(2,4),angle=(15,10),rho=.2),len(chunk),rng,task.alpha.device)
        g=task.target(par,torch.zeros(len(chunk),device=task.alpha.device));g[:,6:8]=flat['goal'][chunk,6:8].to(task.alpha.device)
        q,valid,err,margin,it=teacher.solve(S,g[:,:3],prev)
        for key,val in [('goal',g),('q_star',q),('valid',valid),('error_deg',err),('margin',margin),('iterations',it)]:flat[key][chunk]=val.cpu()
    result=dict(data)
    for k in fields:result[k]=torch.cat((data[k],new[k]),1)
    result['sample_kind']=torch.cat((torch.zeros(n,T,dtype=torch.int8),torch.ones(n,extra,dtype=torch.int8)),1)
    result['dataset_path']=str((out/'supervision_augmented.pt').resolve())
    torch.save(result,out/'supervision_augmented.pt')
    json_save(out/'augmentation.json',dict(valid_labels=int(new['valid'].sum()),kind='offline goal relabel on actual state/history; not dynamics transitions',validation_unchanged=True))
    print('Offline goal relabels:',int(new['valid'].sum()),flush=True)
    return result

def supervise(task,ref,data,config,out,initial=None):
    dev=task.alpha.device;student=policy(task);student.migrate(ref.state_dict(),task)
    if initial: student.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['pi'])
    delta=data['goal'][...,:3][data['valid']]-task.h_ref.cpu()
    L=(delta*task.n0.cpu()).sum(-1)
    target_pool=torch.stack((L,torch.atan2((delta*task.e1.cpu()).sum(-1),L),torch.atan2((delta*task.e2.cpu()).sum(-1),L)),-1)[::4]
    torch.save(target_pool,out/'target_pool.pt')
    dataset_path=data.get('dataset_path',str((out/'supervision.pt').resolve()))
    dataset_sha256=sha256(dataset_path)
    opt=torch.optim.Adam(student.parameters(),lr=config['supervised_lr'])
    valid=data['valid'];val=data['validation'][:,None].expand_as(valid)
    train=(valid&~val).flatten().nonzero().flatten();test=(valid&val).flatten().nonzero().flatten()
    if not len(train) or not len(test):raise RuntimeError('No valid training/validation labels; see dataset.json')
    ds={k:v.flatten(0,1).to(dev) for k,v in data.items() if k in ('s','hist','goal','previous','q_star','body')}
    train,test=train.to(dev),test.to(dev);B=config['supervised_batch'];best=None
    def loss_batch(idx):
        S=unflatten_state(ds['s'][idx]);f=torch.zeros(len(idx),dtype=torch.long,device=dev);gs=f+50
        cmd=task.command(S,f,f,gs,ds['goal'][idx],ds['previous'][idx]);a=student(ds['hist'][idx],cmd)
        q=task.mdp.default_q[task.terminal]+task.alpha[task.terminal]*a[:,task.terminal]
        qc=q.clamp(task.mdp.soft_lo[task.terminal],task.mdp.soft_hi[task.terminal]);S=list(S);S[4]=S[4].scatter(-1,task.terminal.expand_as(q),qc)
        h,d,_=task.hand(S);u=ds['goal'][idx,:3]-h;u=u/u.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        c=(d*u).sum(-1).clamp(-1,1);dq=q-ds['q_star'][idx];db=a[:,task.body]-ds['body'][idx]
        loss=(dq/task.half).square().mean()+config.get('direction_weight',1.)*(1-c).mean()+db.square().mean()
        return loss,torch.rad2deg(torch.acos(c.detach())),torch.rad2deg(dq.detach()),db.detach()
    loss_batch=torch.compile(loss_batch)
    for epoch in range(config['epochs']):
        order=train[torch.randperm(len(train),device=dev)];total=0
        for idx in order.split(B):
            opt.zero_grad();loss,*_=loss_batch(idx);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())
        with torch.no_grad():
            result=[loss_batch(idx)[1:] for idx in test.split(B)]
            angle,dq,db=[torch.cat([x[j] for x in result]) for j in range(3)]
            report=dict(epoch=epoch+1,loss=total/math.ceil(len(train)/B),direction_p95_deg=float(torch.quantile(angle,.95)),joint_rmse_deg=dq.square().mean(0).sqrt().tolist(),body_rmse=float(db.square().mean().sqrt()))
            report['passed']=report['direction_p95_deg']<=1 and max(report['joint_rmse_deg'])<=5 and report['body_rmse']<=.05
        ck=dict(pi=student.state_dict(),task_config=task.metadata(),config=config,stage='supervised',source_sha256=data['source_sha256'],dataset=data.get('dataset_path',str((out/'supervision.pt').resolve())),robot_metadata=task.mdp.robot.metadata(),target_pool=target_pool,dataset_sha256=dataset_sha256,report=report)
        torch.save(ck,out/'last.pt');key=(report['body_rmse']>.05,report['direction_p95_deg'],max(report['joint_rmse_deg']))
        if best is None or key<best:best=key;torch.save(ck,out/'best.pt');json_save(out/'supervised.json',report)
        print(json.dumps(report),flush=True)
        if report['passed']:break
    return out/'best.pt'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--reuse-data',action='store_true');p.add_argument('--augment-targets',action='store_true');p.add_argument('--init-supervised');args=p.parse_args()
    config=json.loads(Path(args.config).read_text());torch.set_num_threads(8);torch.manual_seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=True
    if args.smoke:config.update(sequences=8,sequence_steps=25,epochs=2,supervised_batch=64)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);json_save(out/'config.json',config)
    task=setup(config);source=torch.load(config['source'],map_location='cpu',weights_only=False)
    ref=Policy(99,task.ctc.cmd_dim,robot=task.mdp.robot).cuda();ref.load_state_dict(source['pi']);ref.eval();freeze_normalizers(ref)
    data=torch.load(out/('supervision_augmented.pt' if config.get('use_augmented_data',False) else 'supervision.pt'),weights_only=False) if args.reuse_data else collect(task,ref,config,out,args.smoke)
    if args.augment_targets:data=augment(task,data,out,config)
    ck=supervise(task,ref,data,config,out,args.init_supervised)
    print('C0 checkpoint:',ck,flush=True)

if __name__=='__main__':main()
