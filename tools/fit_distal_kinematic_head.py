"""Offline differentiable-kinematics distal network experiment, not a physical policy."""
import argparse,json,time
from pathlib import Path
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.wide_curriculum import sample,grid
from supermimic.acp.robots import sha256
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--iterations',type=int,default=2000);p.add_argument('--resume-head',type=Path);p.add_argument('--learning-rate',type=float,default=.002);a=p.parse_args();assert not a.out.exists();torch.set_num_threads(2);torch.manual_seed(391);torch.backends.cuda.matmul.allow_tf32=False
ck=torch.load(a.source,map_location=a.device,weights_only=False);T=WideAimingTask(a.device,ck['config']);ids=T.arm_ids.tolist()[4:];center=T.initial_q[ids];lo=T.mdp.soft_lo[ids];hi=T.mdp.soft_hi[ids];mid=(lo+hi)/2;half=(hi-lo)/2
net=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,3)).to(a.device)
with torch.no_grad():net[-1].weight.zero_();net[-1].bias.copy_(torch.atanh(((center-mid)/half).clamp(-.99,.99)))
if a.resume_head:
 previous=torch.load(a.resume_head,map_location=a.device,weights_only=False);assert previous['source_sha256']==sha256(a.source);net.load_state_dict(previous['network'])
op=torch.optim.Adam(net.parameters(),lr=a.learning_rate);train_gen=torch.Generator(device=a.device).manual_seed(391);test_gen=torch.Generator(device=a.device).manual_seed(1791)
def context(params):
 S=T.initial_state(len(params));ctx=T.context(S,params,4.);return S,ctx
train_params=sample(4,4096,a.device,train_gen)[0];test_params=torch.cat((grid(4,a.device),sample(4,512,a.device,test_gen)[0]));Sall,Call=context(train_params);St,Ct=context(test_params)
def prediction(S,ctx):
 # Two feedforward refinements use current hand position to match runtime target direction input.
 q=S[4];direction=None
 for _ in range(2):
  h,_,R=T.hand((*S[:4],q,S[5]));direction=torch.nn.functional.normalize((R.transpose(-1,-2)@(ctx['aim_target_w']-h)[...,None]).squeeze(-1),dim=-1)
  distal=mid+half*torch.tanh(net(direction));q=S[4].clone();q[:,ids]=distal
 h,d,R=T.hand((*S[:4],q,S[5]));want=torch.nn.functional.normalize(ctx['aim_target_w']-h,dim=-1)
 return q,d,want
start=time.time();best=float('inf');beststate=None
for it in range(1,a.iterations+1):
 idx=torch.randint(4096,(256,),device=a.device,generator=train_gen);S=tuple(v[idx] for v in Sall);ctx={k:v[idx] for k,v in Call.items()};q,d,want=prediction(S,ctx);angular=(d-want).square().sum(-1);loss=angular.mean()+.1*angular.max()+1e-5*(q[:,ids]-center).square().mean();op.zero_grad();loss.backward();op.step()
 if it%250==0 or it==a.iterations:
  with torch.no_grad():
   q,_,_=prediction(St,Ct);err=T.metrics((*St[:4],q,St[5]),Ct)[0];score=float(torch.quantile(err,.95));report=dict(iteration=it,loss=float(loss),grid_within5cm=int((err[:75]<=.05).sum()),random_within5cm=float((err[75:]<=.05).float().mean()),p95_error_m=score,max_error_m=float(err.max()),elapsed_s=time.time()-start)
   if score<best:best=score;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};bestreport=report
  print(json.dumps(report),flush=True)
a.out.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(scope=__doc__,source_sha256=sha256(a.source),resume_head_sha256=sha256(a.resume_head) if a.resume_head else None,network=beststate,mid=mid.cpu(),half=half.cpu(),joint_ids=ids,report=bestreport,physical_acceptance=False),a.out);a.out.with_suffix('.json').write_text(json.dumps(dict(scope=__doc__,source_sha256=sha256(a.source),report=bestreport,physical_acceptance=False),indent=2))
