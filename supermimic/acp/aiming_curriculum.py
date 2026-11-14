"""Explicit, small curriculum; only settled physical episodes advance it."""
import math
import numpy as np
import torch

def levels():
    out=[]
    def add(name,mode=0,angle=(15,10),depth=(3,3),rho=.2,hold=5,**kw):
        out.append(dict(name=name,mode=mode,angle=angle,depth=depth,rho=rho,hold=hold,**kw))
    for h in (1,3,5): add(f'C1-h{h}',angle=(0,0),hold=h)
    for a in ((5,3),(10,6),(15,10)): add(f'C2-{a[0]}',angle=a)
    for d in ((2.5,3.5),(2,4)): add(f'C3-{d[0]}',depth=d)
    previous=.2
    for r in (.15,.10,.075,.05):
        add(f'C4-{r}-review',rho=previous)
        add(f'C4-{r}-precision',rho=r)
        for d in ((2.5,3.5),(2,4)): add(f'C4-{r}-{d[0]}',rho=r,depth=d)
        previous=r
    for name,jump,dwell,r in [('A1',(3,5),(8,12),.15),('A2',(10,15),(8,12),.15),('A3',(10,15),(4,8),.15),('A4-10',(10,15),(4,8),.1),('A4-5',(10,15),(4,8),.05)]:
        add('C5'+name,mode=1,rho=r,hold=2,jump=jump,dwell=dwell)
    for speed in (2,5,10): add(f'C5B-sine-{speed}',mode=2,rho=.15,speed=speed,two_dim=False,hold=2)
    add('C5B-2d',mode=2,rho=.15,speed=10,two_dim=True,hold=2)
    add('C5B-hermite',mode=3,rho=.15,speed=10,hold=2)
    for r in (.10,.05): add(f'C5B-precision-{r}',mode=3,rho=r,speed=10,hold=2)
    for d in ((3,3),(2.5,3.5),(2,4)): add(f'C6-{d[0]}',mode=-1,depth=d,rho=.05,hold=2,speed=10,jump=(10,15),dwell=(4,8))
    return out

LEVELS=levels()
TARGET_POOL=None

def set_target_pool(pool):
    global TARGET_POOL
    TARGET_POOL=np.asarray(pool) if pool is not None else None

def sample_parameters(level,n,rng,device):
    """64 numbers per episode, including a bounded eight-node target schedule."""
    p=np.zeros((n,64),dtype=np.float32)
    for i in range(n):
        c=level; mode=c['mode']
        if mode==-1: mode=int(rng.choice([0,1,3],p=[.5,.25,.25]))
        depth=rng.uniform(*c['depth']); bounds=np.deg2rad(c['angle'])
        pool=TARGET_POOL
        if pool is not None:pool=pool[(np.abs(pool[:,1:])<=bounds+1e-6).all(-1)]
        ab=pool[int(rng.integers(len(pool))),1:].copy() if pool is not None and len(pool) and bounds.max()>0 else rng.uniform(-bounds,bounds)
        p[i,:6]=[mode,depth,c['rho'],c.get('sigma',c['rho']),*ab]
        nodes=p[i,16:].reshape(8,6); nodes[:,0]=np.arange(8)*100+100
        nodes[0,:3]=[0,*ab]
        if mode==1:
            time=0
            for j in range(1,8):
                dwell=rng.uniform(*c.get('dwell',(8,12)))
                time+=dwell
                if time+c.get('dwell',(8,12))[1]>30: break
                # Finite candidate pool; inability to meet jump leaves original segment.
                candidates=pool[rng.integers(len(pool),size=256),1:] if pool is not None and len(pool) else rng.uniform(-bounds,bounds,(256,2))
                rays=np.c_[np.ones(256),np.tan(candidates)]; rays/=np.linalg.norm(rays,axis=1)[:,None]
                old=np.r_[1,np.tan(ab)]; old/=np.linalg.norm(old)
                angles=np.rad2deg(np.arccos(np.clip(rays@old,-1,1)))
                lo,hi=c.get('jump',(3,5)); valid=np.flatnonzero((angles>=lo)&(angles<=hi))
                if not len(valid): break
                ab=candidates[valid[0]]; nodes[j,:3]=[time,*ab]
        elif mode==2:
            amp=np.deg2rad(rng.uniform(3,5,2)); amp[1]*=bool(c.get('two_dim',False))
            speed=np.deg2rad(c.get('speed',5))
            freq=speed/max(np.linalg.norm(amp),1e-6)*np.array([1,rng.uniform(.7,1.3)])
            # Conservative combined speed bound; measured again from exact reference rays below.
            freq*=min(1.,.9*speed/max(np.linalg.norm(amp*freq),1e-8))
            p[i,4:6]=0; p[i,6:8]=amp; p[i,8:10]=freq;p[i,10:12]=rng.uniform(0,2*np.pi,2)
        elif mode==3:
            xy=rng.uniform(-bounds,bounds,(8,2)); vel=np.zeros_like(xy)
            # Zero shared knot tangents are C1 and keep every cubic inside its endpoint box.
            duration=np.maximum(4.,1.5*np.linalg.norm(np.diff(xy,axis=0),axis=1)/(.8*np.deg2rad(c.get('speed',10))))
            acc=np.deg2rad(max(5,4*c.get('speed',10)))
            duration=np.maximum(duration,np.sqrt(6*np.linalg.norm(np.diff(xy,axis=0),axis=1)/acc))
            duration*=max(1.,30./duration.sum())
            nodes[:,0]=np.r_[0,np.cumsum(duration)];nodes[:,1:3]=xy;nodes[:,3:5]=vel
    return torch.tensor(p,device=device)

class Curriculum:
    def __init__(self,seed=0,state=None):
        self.rng=np.random.default_rng(seed);self.level=0;self.windows={};self.pending={};self.frontier={};self.history=[];self.mastered=[];self.transition=False;self.previous_rho=.2;self.paused=False
        if state:
            for k,v in state.items():
                if k!='rng':setattr(self,k,v)
            self.rng.bit_generator.state=state['rng']

    def state_dict(self):
        return {k:v for k,v in vars(self).items() if k!='rng'}|{'rng':self.rng.bit_generator.state}

    def sample(self,n,device):
        allp=[]
        for _ in range(n):
            bucket=int(self.rng.choice(3,p=[.2,.6,.2])); lev=self.level
            if bucket==0 and self.mastered: lev=int(self.rng.choice(self.mastered))
            elif bucket==2 and self.level+1<len(LEVELS): lev+=1
            else: bucket=1
            c=dict(LEVELS[lev])
            if lev==self.level and self.transition:c['sigma']=max(c['rho'],self.previous_rho)
            p=sample_parameters(c,1,self.rng,device);p[:,12]=lev;p[:,13]=bucket;p[:,14]=5 if c['mode']==-1 and int(p[0,0])==0 else c['hold'];allp.append(p)
        return torch.cat(allp)

    def settle(self,params,report):
        lev,bucket=int(params[12]),int(params[13]);mode=int(params[0]);key=f'{lev}:{mode}'
        successes=report['attempt_success']
        if bucket==2:
            self.frontier.setdefault(key,[]).extend(successes);self.frontier[key]=self.frontier[key][-64:];return
        if bucket!=1 or lev!=self.level:return
        self.pending.setdefault(key,[]).extend([(bool(x), bool(abs(float(params[3])-float(params[2]))<1e-6)) for x in successes])
        if len(self.pending[key])<64:return
        batch=self.pending[key][:64];self.pending[key]=self.pending[key][64:]
        wins=sum(x[0] for x in batch); final_width=all(x[1] for x in batch); high,low=self.windows.get(key,[0,0])
        high,low=(high+1,0) if wins>=55 else ((0,low+1) if wins<=38 else (0,0))
        if not final_width:high=0;self.transition=False
        self.windows[key]=[high,low];self.history.append(dict(level=lev,mode=mode,wins=wins))
        if self.paused:return
        frontier=[x for k,v in self.frontier.items() if k.startswith(str(lev+1)+':') for x in v]
        advance=high>=2 and (lev==len(LEVELS)-1 or (len(frontier)>=16 and np.mean(frontier)>=.6))
        if LEVELS[lev]['mode']==-1:
            advance=advance and all(self.windows.get(f'{lev}:{m}',[0,0])[0]>=2 for m in (0,1,3))
        if low>=2 and lev>0:
            self.level-=1;self.mastered=[x for x in self.mastered if x<lev];self.windows={};self.pending={}
        elif advance:
            if lev not in self.mastered:self.mastered.append(lev)
            if lev<len(LEVELS)-1:
                self.previous_rho=LEVELS[lev]['rho'];self.level+=1
                self.transition=LEVELS[self.level]['rho']<self.previous_rho
                self.windows={};self.pending={}

def summarize(task, states, goals, params, terminated=False):
    """One complete physical episode. All attempts inherit the whole-episode body result."""
    S=tuple(torch.stack([s[i] for s in states]) for i in range(6));g=torch.stack(goals)
    with torch.no_grad():
        bp=task.mdp.kin.forward(*S)[0]
        root=(S[0][...,:2]-S[0][0,...,:2]).norm(dim=-1)
        foot=(bp[...,task.feet,:2]-bp[0,...,task.feet,:2]).norm(dim=-1)
        tilt=torch.rad2deg(torch.acos(S[1][...,2,2].clamp(-1,1)))
        fixed=torch.rad2deg((S[4][...,task.fixed]-task.target_q[task.fixed]).abs())
        err,angle,valid,_=task.metrics(S,g)
    arr=lambda x:x.detach().cpu().numpy()
    root,foot,tilt,fixed,err,angle,valid=map(arr,(root,foot,tilt,fixed,err,angle,valid))
    return metric_report(root,foot,tilt,fixed,err,angle,valid,params,terminated)

def metric_report(root,foot,tilt,fixed,err,angle,valid,params,terminated=False):
    full=(len(err)-1)*.02>=30 and not terminated
    body=bool(full and root.max()<=.10 and foot.max()<=.05 and tilt.max()<=15 and np.percentile(fixed,95,axis=0).max()<=5)
    p=np.asarray(params); mode=int(p[0]);rho=float(p[2]);hold=float(p[14] or (2 if mode else 5))
    starts=[0]
    if mode==1:starts += [math.ceil(float(t)*50-1e-5) for t in p[16:].reshape(8,6)[1:,0] if t<30 and t<=(len(err)-1)*.02]
    ends=starts[1:]+[1500];attempts=[];captures=[]
    hit=valid&(err<=rho); longest=0
    for start,end in zip(starts,ends):
        run=0;capture=None
        for j in range(start,min(end+1,len(hit))):
            run=run+1 if hit[j] else 0;longest=max(longest,run)
            # Samples include both endpoints: duration is (run-1)*dt.
            if (run-1)*.02>=hold and capture is None:capture=(j-run+1-start)*.02
        limit=min(3,(end-start)*.02-hold) if mode==1 else 3
        captures.append(capture);attempts.append(bool(body and capture is not None and capture<=limit))
    tail=hit[150:];run=0;lost=0
    for h in tail:run=0 if h else run+1;lost=max(lost,run)
    if mode>=2:attempts=[bool(attempts[0] and len(tail)>0 and tail.mean()>=.9 and lost*.02<=.5)]
    def q95(x):
        if not len(x): return float('inf')
        return float(np.sort(x)[min(len(x)-1,math.ceil(.95*len(x))-1)])
    return dict(standing_pass=body,terminated=terminated,seconds=(len(err)-1)*.02,
                root_max_m=float(root.max()),feet_max_m=foot.max(0).tolist(),tilt_max_deg=float(tilt.max()),fixed_p95_deg=np.percentile(fixed,95,axis=0).tolist(),
                error_mean_m=float(err.mean()),error_p95_m=q95(err),post_capture_p95_m=q95(err[150:]),angle_p95_deg=q95(angle),valid_fraction=float(valid.mean()),
                longest_hit_s=max(0,longest-1)*.02,hit_fraction=float(tail.mean()) if len(tail) else 0,longest_loss_s=lost*.02,capture_s=captures,attempt_success=attempts,
                radius_hit_fraction={str(r):float((err[150:]<=r).mean()) if len(err)>150 else 0 for r in (.2,.1,.05)})
