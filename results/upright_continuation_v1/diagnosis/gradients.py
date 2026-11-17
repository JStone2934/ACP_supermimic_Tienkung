import sys,json
from pathlib import Path
ROOT=Path('/home/js/robot/locomotion/standing-aiming');P=ROOT/'results/upright_continuation_v1';sys.path.insert(0,str(P/'runner'))
import numpy as np,torch
from torch.utils.checkpoint import checkpoint
from supermimic.acp.aiming_fullbody import FullbodyPolicy,flat_boxes
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.mdp import unflatten_state,normalize_xi
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.posture_transfer import attach_teacher,retention_cost

torch.set_num_threads(2);dev='cpu';c=torch.load(P/'train/ckpt_000100.pt',map_location=dev,weights_only=False);n=torch.load(P/'train/ckpt_000105.pt',map_location=dev,weights_only=False)
task=WideAimingTask(dev,c['config']);attach_teacher(task,c['config']);pi=FullbodyPolicy(task.robot);pi.load_state_dict(c['pi']);wm=WorldModel(task.mdp.scan_dim,robot=task.robot);wm.load_state_dict(c['wm']);wm.requires_grad_(False);freeze_normalizers(pi,wm)
z=np.load(P/'train/wm_validation/000100.npz');d={k:torch.tensor(z[k]) for k in z.files};assert not d['done'].any()
S0=unflatten_state(d['s'][0]);ctx=task.context(S0,torch.tensor([[3.,0.,0.,.1]]).expand(8,-1),d['aim_weight'][0]);chest=ctx['aim_target_w']-3*ctx['aim_normal_w'];ctx.update(aim_target_w=d['aim_target_w'][0],aim_tolerance=d['aim_tolerance'][0],aim_normal_w=torch.nn.functional.normalize(d['aim_target_w'][0]-chest,dim=-1))
times=[64,96];cs=[]
for t in range(97):
 ctx=task.observe_context(unflatten_state(d['s'][t]),ctx)
 if t in times:cs.append({k:v.clone() for k,v in ctx.items()})
 ctx=task.advance_context(ctx,1)
ctx={k:torch.cat([x[k] for x in cs]) for k in ctx};S=unflatten_state(torch.cat([d['s'][t] for t in times]));xi=torch.cat([d['xi'][t] for t in times]);ap=torch.cat([d['a'][t-1] for t in times]);clock=torch.cat([d['ep_step'][t] for t in times]).long()
hist=torch.cat([torch.stack([task.mdp.obs(unflatten_state(d['s'][j]),d['a'][j-1],d['xi'][j]) for j in range(t-9,t+1)],1) for t in times]);cmd=task.command(S,ctx)
with torch.no_grad():
 a0=pi(hist,cmd);pi.load_state_dict(n['pi']);a1=pi(hist,cmd);delta=(a1-a0)*task.alpha
 report=dict(same_input_action_change=dict(rms_deg=float(delta.square().mean().sqrt()*180/torch.pi),max_deg=float(delta.abs().max()*180/torch.pi),per_joint_rms_deg=dict(zip(task.robot.joint_names,(delta.square().mean(0).sqrt()*180/torch.pi).tolist()))),normalizers_unchanged=all(torch.equal(c['pi'][k],n['pi'][k]) for k in c['pi'] if 'norm' in k))
 pi.load_state_dict(c['pi'])
groups=['aim','stability','chest','pose','request','regularization','retention']
params=[p for p in pi.parameters() if p.requires_grad];names=[k for k,p in pi.named_parameters() if p.requires_grad];change=[n['pi'][k]-c['pi'][k] for k in names]
def step(s,h,prev,t):
 cc=task.advance_context(ctx,t-clock);nc=task.advance_context(ctx,t+1-clock);cmd=task.command(s,cc);raw=pi(h,cmd);a,_=task.compose_action(raw,xi);ns=wm.step(s,a,normalize_xi(xi),task.mdp.scan(s,flat_boxes(s)));_,v=task.reward_terms(ns,a,prev,nc,raw=raw,xi=xi)
 with torch.no_grad():teacher=task.retention_teacher(h.detach(),cmd.detach())
 retain=-task.dt*retention_cost(raw,teacher,task.alpha,2.)
 chosen={'aim','chest_upright','chest_excess','posture_reference','request_limit','action_rate','joint_limit'}
 vals=torch.stack([v['aim'],sum(x for k,x in v.items() if k not in chosen),v['chest_upright']+v['chest_excess'],v['posture_reference'],v['request_limit'],v['action_rate']+v['joint_limit'],retain],-1)
 return ns,torch.cat((h[:,1:],task.mdp.obs(ns,a,xi)[:,None]),1),a,vals,task.soft_termination(ns,t+1,nc)
def obj():
 s,h,prev,t=S,hist,ap,clock;alive=torch.ones(len(t));o=torch.zeros(len(t),7)
 for j in range(32):
  s,h,prev,v,done=checkpoint(step,s,h,prev,t,use_reentrant=False);o=o+.99**j*alive[:,None]*v;alive=alive*(1-done);t=t+1
 return o.mean(0)
def gradients(values):
 out={}
 for j,key in enumerate(groups):
  gs=torch.autograd.grad(values[j],params,retain_graph=True,allow_unused=True);norm=sum(g.square().sum() for g in gs if g is not None).sqrt();dot=sum((g*dd).sum() for g,dd in zip(gs,change) if g is not None)
  out[key]=dict(value=float(values[j].detach()),gradient_norm=float(norm),update_dot_gradient=float(dot))
 return out
v=obj();report['source']=gradients(v);print('source done',flush=True)
pi.load_state_dict(n['pi']);v=obj();report['candidate']=gradients(v);report['scope']='16 recorded physical starts; 32-step eager CPU rollout; fixed pre-update world model; gradients all trainable parameters; source-gradient dots with finite five-update displacement are not per-step causality'
(P/'diagnosis/gradients.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
