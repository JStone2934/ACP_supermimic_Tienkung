"""Single actor curriculum trainer. Initialization consumes the C0 checkpoint."""
import argparse,json,math,random,copy,itertools,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .aiming import setup,policy,json_save,AimingSim,DEFAULT_CONFIG
from .aiming_curriculum import Curriculum,LEVELS,metric_report,sample_parameters,set_target_pool
from .env import TrackingSim
from .networks import WorldModel,Policy,Critic,run_mlp
from .standing import freeze_normalizers
from .buffer import ReplayBuffer
from .mdp import unflatten_state,normalize_xi
from .robots import sha256

class AimCritic(Critic):
    def __init__(self,base_dim):
        super().__init__(base_dim);self.base_dim=base_dim
        self.net[0]=torch.nn.Linear(base_dim+82,256)
    def forward(self,x):
        return run_mlp(self.net,torch.cat((self.norm(x[...,:self.base_dim]),x[...,self.base_dim:]),-1)).squeeze(-1)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config','--aim_config',default=DEFAULT_CONFIG);ap.add_argument('--log_dir',required=True)
    group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--finetune');group.add_argument('--resume')
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    torch.set_num_threads(8);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
    if args.smoke:cfg.update(num_envs=8,wm_batch=8,pi_batch=4,horizon=4,buffer_steps=100,wm_batches=1,wm_only_iters=2,iterations=4,ckpt_every=4,resume_wm_iters=2,critic_epochs=1,critic_minibatches=1)
    ck=torch.load(args.resume or args.finetune,map_location='cpu',weights_only=False)
    if ck.get('stage') not in ('supervised','curriculum'):raise ValueError('Use a C0 or curriculum checkpoint')
    if args.resume and ck.get('stage')!='curriculum':raise ValueError('Use --finetune for C0 weights')
    set_target_pool(ck.get('target_pool'))
    task=setup(cfg);dev='cuda:0';mdp=task.mdp;ctc=task.ctc;pi=policy(task,ck['pi'])
    source=torch.load(cfg['source'],map_location='cpu',weights_only=False)
    ref=Policy(99,ctc.cmd_dim,robot=mdp.robot).to(dev);ref.load_state_dict(source['pi']);freeze_normalizers(ref);ref.eval();ref.requires_grad_(False)
    wm=WorldModel(mdp.scan_dim,robot=mdp.robot).to(dev);wm.load_state_dict(ck['wm'] if args.resume else source['wm']);freeze_normalizers(wm)
    v= AimCritic(mdp.inv_dim+mdp.critic_extra+ctc.cmd_dim).to(dev)
    optim=[torch.optim.Adam(m.parameters(),lr=lr) for m,lr in ((pi,cfg['actor_lr']),(wm,cfg['wm_lr']),(v,cfg['critic_lr']))]
    optim[0].param_groups[0]['betas']=(.9,.95)
    cur=Curriculum(cfg['seed'],ck.get('curriculum') if args.resume else None)
    start=ck.get('iteration',0) if args.resume else 0;updates=ck.get('actor_updates',0) if args.resume else 0
    recovery=ck.get('recovery',dict(active=False,bad=0,good=0,release=None,last_stable=0)) if args.resume else dict(active=False,bad=0,good=0,release=None,last_stable=0)
    if args.resume:
        v.load_state_dict(ck['critic'])
        for opt,state in zip(optim,ck['optimizers']):opt.load_state_dict(state)
        torch.set_rng_state(ck['rng_torch']);torch.cuda.set_rng_state_all(ck['rng_cuda']);np.random.set_state(ck['rng_numpy']);random.setstate(ck['rng_python'])
        cur.pending={}  # discard unfinished windows with physical/replay restart
    out=Path(args.log_dir);out.mkdir(parents=True,exist_ok=True);json_save(out/'config.json',cfg)
    base=TrackingSim(cfg['num_envs'],mdp.motion,mdp,dev,control_task=task)
    env=AimingSim(base,task,lambda n:cur.sample(n,dev));N=base.N
    buffer=ReplayBuffer(cfg['buffer_steps'],N,dev,extra={'aim_params':(64,torch.float32),'terminal_prev':(3,torch.float32)},robot=mdp.robot)
    hist=torch.zeros(N,10,99,device=dev);gs=torch.full((N,),50,device=dev)
    records=torch.zeros(N,1501,11,device=dev);root0=torch.zeros(N,3,device=dev);feet0=torch.zeros(N,2,3,device=dev)
    def measure(S,goal,ids):
        bp=mdp.kin.forward(*S)[0][:,task.feet]
        e,ang,valid,_=task.metrics(S,goal)
        return torch.cat(((S[0][:,:2]-root0[ids,:2]).norm(dim=-1)[:,None],(bp[...,:2]-feet0[ids,:,:2]).norm(dim=-1),torch.rad2deg(torch.acos(S[1][:,2,2].clamp(-1,1)))[:,None],torch.rad2deg((S[4][:,task.fixed]-task.target_q[task.fixed]).abs()),e[:,None],ang[:,None],valid[:,None]),-1)
    @torch.no_grad()
    def collect(n):
        nonlocal hist
        for _ in range(n):
            ids=torch.arange(N,device=dev);fresh=base.ep_step==0;S=base.state()
            root0[fresh]=S[0][fresh];feet0[fresh]=mdp.kin.forward(*S)[0][fresh][:,task.feet]
            records[ids[fresh],0]=measure(S,env.goal(),ids)[fresh]
            o=base.observe();hist=torch.where(fresh[:,None,None],o[:,None].expand_as(hist),torch.cat((hist[:,1:],o[:,None]),1))
            a=pi(hist,task.command(S,base.f,base.v,gs,env.goal(),env.previous))
            noise=cfg['terminal_noise_rad']*max(0,1-updates/cfg['noise_decay_updates']) if updates else 0
            a[:,task.terminal]+=noise*torch.randn_like(a[:,task.terminal])/task.alpha[task.terminal]
            tr=env.step(a);tr['goal_steps']=gs.clone();buffer.add(tr)
            records[ids,tr['ep_step']+1]=measure(unflatten_state(tr['s1']),tr['aim_goal1'],ids)
            for i in tr['done'].nonzero().flatten().tolist():
                x=records[i,:int(tr['ep_step'][i])+2].cpu().numpy();p=tr['aim_params'][i].cpu().numpy()
                r=metric_report(x[:,0],x[:,1:3],x[:,3],x[:,4:8],x[:,8],x[:,9],x[:,10].astype(bool),p,bool(tr['term'][i]));cur.settle(p,r)
    def critic_input(S,f,variant,xi,params,t,prev):
        g=task.target(params,t.float()*.02);cmd=task.command(S,f,variant,torch.full_like(f,50),g,prev)
        scale=torch.ones(64,device=dev);scale[0]=3;scale[1]=3;scale[2:4]=.2;scale[10:12]=2*math.pi;scale[16::6]=30
        proc=params/scale
        proc=torch.cat((proc[:,:12],torch.zeros_like(proc[:,12:16]),proc[:,16:]),-1)
        proc=proc.scatter(-1,torch.arange(16,64,6,device=dev).expand(len(proc),-1),proc[:,16::6].clamp_max(1))
        return torch.cat((ctc.critic_input(S,f,variant,torch.full_like(f,50),normalize_xi(xi)),cmd[:,-17:],proc,t[:,None]/1500),-1)
    def wm_loss(w):
        S=unflatten_state(w['s'][:,0]);loss=0;std=wm.out_norm.std().clamp_min(.003)
        for j in range(8):
            S=wm.step(S,w['a'][:,j],normalize_xi(w['xi'][:,j]),mdp.scan(S,mdp.motion.boxes(w['f'][:,j],w['v'][:,j])))
            err=(wm.state_error(S,unflatten_state(w['s1'][:,j]))/std).square().mean(-1)
            loss+=(err*w['mask'][:,j]).mean()/8
        return loss
    wm_loss=torch.compile(wm_loss)
    def step(S,h,f,variant,xi,p,t,prev,ap):
        g=task.target(p,t.float()*.02)
        raw=pi(h,task.command(S,f,variant,torch.full_like(f,50),g,prev));a,q=task.compose_action(raw,xi,prev)
        with torch.no_grad():teacher=ref(h,ctc.command(S,f,variant,torch.full_like(f,50),noise=False))
        body=(raw[:,task.body]-teacher[:,task.body]).square().mean(-1)
        S1=wm.step(S,a,normalize_xi(xi),mdp.scan(S,mdp.motion.boxes(f,variant)))
        g1=task.target(p,(t+1).float()*.02);r=task.aim_reward(S1,f+1,variant,a,ap,g1)
        done=torch.maximum(mdp.soft_termination(S1,f+1),(t+1>=1500).float())
        return S1,torch.cat((h[:,1:],mdp.obs(S1,a,xi)[:,None]),1),a,q,r,done,body
    # Compile recomputation and operation together, as in the existing reward memory helper.
    def recomputed(*a):return checkpoint(step,*a,use_reentrant=False)
    imag=torch.compile(recomputed,options={'max_fusion_size':32})
    def update_actor(weight):
        sample,h=buffer.sample_starts(cfg['pi_batch'],10);S=unflatten_state(sample['s']);hist0=h['o'];f=sample['f'];variant=sample['v'];xi=sample['xi'];params=sample['aim_params'];t=sample['ep_step'].long();prev=sample['terminal_prev'];ap=sample['a_prev']
        anchor=hist0.new_zeros(())
        if cfg.get('replay_body_anchor',0.):
            with torch.no_grad():teacher=ref(hist0,ctc.command(S,f,variant,torch.full_like(f,50),noise=False))
            raw=pi(hist0,task.command(S,f,variant,torch.full_like(f,50),task.target(params,t.float()*.02),prev))
            anchor=(raw[:,task.body]-teacher[:,task.body]).square().mean()
        for model in (wm,v):model.requires_grad_(False)
        alive=torch.ones(len(f),device=dev);J=0;aux=0;count=0;xs=[];rs=[];ds=[];masks=[]
        for j in range(cfg['horizon']):
            xs.append(critic_input(S,f,variant,xi,params,t,prev).detach());masks.append(alive.detach())
            S,hist0,a,prev,r,d,b=imag(S,hist0,f,variant,xi,params,t,prev,ap)
            J=J+(.99**j)*alive*r;aux=aux+(alive*b).sum();count=count+alive.sum();alive=alive*(1-d);ap=a;f=f+1;t=t+1
            rs.append(r.detach());ds.append(d.detach())
        last=critic_input(S,f,variant,xi,params,t,prev)
        J=J+(.99**cfg['horizon'])*alive*v(last)
        loss=-J.mean()+weight*(aux/count.clamp_min(1)+cfg.get('replay_body_anchor',0.)*anchor)
        optim[0].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pi.parameters(),1,error_if_nonfinite=True);optim[0].step()
        for model in (wm,v):model.requires_grad_(True)
        X=torch.stack(xs);mask=torch.stack(masks);v.norm.update(X[...,:v.base_dim])
        with torch.no_grad():
            V=v(torch.cat((X,last.detach()[None]),0));ret=V[-1];ys=[]
            for j in reversed(range(cfg['horizon'])):
                ret=rs[j]+.99*(1-ds[j])*(.05*V[j+1]+.95*ret);ys.append(ret)
            Y=torch.stack(ys[::-1])
        Xf=X.flatten(0,1);Yf=Y.flatten();mf=mask.flatten()
        for _ in range(cfg.get('critic_epochs',5)):
            for idx in torch.randperm(len(Xf),device=dev).chunk(cfg.get('critic_minibatches',4)):
                optim[2].zero_grad();vl=((v(Xf[idx])-Yf[idx]).square()*mf[idx]).sum()/mf[idx].sum().clamp_min(1);vl.backward();torch.nn.utils.clip_grad_norm_(v.parameters(),1,error_if_nonfinite=True);optim[2].step()
        return float(loss.detach()),float(vl.detach()),float((aux/count.clamp_min(1)).detach()),float(anchor.detach())
    collect(64);best=ck.get('best') if args.resume else None
    warm=cfg['resume_wm_iters'] if args.resume else cfg['wm_only_iters']
    for it in (itertools.count(start) if cfg['iterations'] < 0 else range(start,cfg['iterations'] if not (args.smoke and args.resume) else start+4)):
        decay=.03+.97*.5*(1+math.cos(math.pi*min(it,10000)/10000))
        for opt,name in zip(optim,('actor_lr','wm_lr','critic_lr')):
            for group in opt.param_groups:group['lr']=cfg[name]*decay
        collect(2 if it%2==0 else 3)
        for _ in range(cfg['wm_batches']):
            optim[1].zero_grad();wl=wm_loss(buffer.sample_windows(cfg['wm_batch'],8));wl.backward();torch.nn.utils.clip_grad_norm_(wm.parameters(),1,error_if_nonfinite=True);optim[1].step()
        weight=max(0,1-updates/cfg['body_decay_updates'])
        recovery_weight=cfg.get('recovery_body_weight',1.)
        if recovery['active']:weight=recovery_weight
        elif recovery['release'] is not None:weight=max(weight,recovery_weight*max(0,1-(updates-recovery['release'])/500))
        losses=None
        if it-start>=warm:losses=update_actor(weight);updates+=1
        if (it+1)%25==0 or args.smoke:print(json.dumps(dict(iteration=it+1,wm=float(wl.detach()),actor_critic=losses,level=LEVELS[cur.level]['name'],lambda_body=weight)),flush=True)
        if (it+1)%cfg['ckpt_every']==0 or it+1==cfg['iterations']:
            report=None
            if not args.smoke:
                from tools.evaluate_aiming import evaluate
                with torch.random.fork_rng(devices=[0]):
                    report=evaluate(pi,task,out/f'eval_{it+1:06d}')
                    current=evaluate(pi,task,out/f'current_{it+1:06d}',params=sample_parameters(LEVELS[cur.level],3,np.random.default_rng(710),dev))
                stable=all(r['standing_pass'] for r in current['scenarios']);recovery['bad']=0 if stable else recovery['bad']+1;recovery['good']=recovery['good']+1 if stable else 0
                if stable:recovery['last_stable']=cur.level
                if recovery['bad']>=2 and not recovery['active']:
                    import subprocess,sys
                    subprocess.run([sys.executable,'-m','tools.evaluate_standing','--ckpt',cfg['source'],'--out',str(out/'baseline_recheck')],check=True)
                    baseline=json.loads((out/'baseline_recheck.json').read_text())
                    if not baseline['standing_30s_pass']:raise RuntimeError('Baseline no longer passes; see baseline_recheck.json')
                    recovery['active']=True;cur.paused=True;cur.level=recovery.get('last_stable',0)
                elif recovery['active'] and recovery['good']>=2:
                    recovery['active']=False;recovery['release']=updates;cur.paused=False;cur.windows={};cur.pending={}
            saved=dict(stage='curriculum',pi=pi.state_dict(),wm=wm.state_dict(),critic=v.state_dict(),optimizers=[o.state_dict() for o in optim],iteration=it+1,actor_updates=updates,config=cfg,task_config=task.metadata(),curriculum=cur.state_dict(),recovery=recovery,source_sha256=sha256(cfg['source']),robot_metadata=mdp.robot.metadata(),data_hash=mdp.motion.data_hash,supervision_dataset=ck.get('dataset',ck.get('supervision_dataset')),target_pool=ck.get('target_pool'),dataset_sha256=ck.get('dataset_sha256'),reference=ref.state_dict(),migration=ck.get('migration'),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),best=best)
            if report:
                rs=report['scenarios'];key=(sum(not x['standing_pass'] for x in rs),-sum(x['passed'] for x in rs),sum(x['post_capture_p95_m'] for x in rs))
                if best is None or key<tuple(best):best=key;saved['best']=best;torch.save(saved,out/'best.pt')
            torch.save(saved,out/'last.pt.tmp');os.replace(out/'last.pt.tmp',out/'last.pt');torch.save(saved,out/f'ckpt_{it+1:06d}.pt')
            for old in sorted(out.glob('ckpt_*.pt'))[:-3]:old.unlink()
            if cfg['iterations'] >= 0 and report and report['passed'] and weight==0 and len(LEVELS)-1 in cur.mastered:
                loaded=torch.load(out/'last.pt',map_location='cpu',weights_only=False);pi.load_state_dict(loaded['pi'])
                confirmed=evaluate(pi,task,out/'confirmed')
                if confirmed['passed']:json_save(out/'completion.json',dict(passed=True,iteration=it+1));return
    json_save(out/'completion.json',dict(passed=False,reason='smoke completed' if args.smoke else 'iteration budget reached',actor_updates=updates,level=cur.level))

if __name__=='__main__':main()
