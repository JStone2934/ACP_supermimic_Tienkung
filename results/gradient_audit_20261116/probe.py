"""Read-only CPU gradient probe of the actual frozen Take006 runner/checkpoint."""
import sys, json
from pathlib import Path
ROOT=Path('/home/js/robot/locomotion/standing-aiming')
sys.path.insert(0,str(ROOT/'results/lafan_jwp_aiming_20261115/frozen_take006_upright_v2'))
import torch
torch.set_num_threads(2)
torch.manual_seed(17)
from supermimic.acp.aiming_fullbody import FullbodyPolicy, flat_boxes
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel,Critic
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.wide_curriculum import sample
from supermimic.acp.mdp import normalize_xi
from supermimic.acp.train_aiming_fullbody import imagination_step,bound_bootstrap
path=ROOT/'logs/tienkung3_29dof/lafan_jwp_20261115/take006_upright/train/ckpt_001800.pt'
ck=torch.load(path,map_location='cpu',weights_only=False);cfg=ck['config']
task=WideAimingTask('cpu',cfg)
pi=FullbodyPolicy(task.robot);pi.load_state_dict(ck['pi'])
wm=WorldModel(task.mdp.scan_dim,robot=task.robot);wm.load_state_dict(ck['wm']);wm.requires_grad_(False)
critic=Critic(task.critic_dim);critic.load_state_dict(ck['critic']);critic.requires_grad_(False)
freeze_normalizers(pi,wm,critic)
params=[pi.dec[-1].weight,pi.dec[-1].bias]
def grad(v):
 if not v.requires_grad:return dict(requires_grad=False,norm=None,finite=True)
 gs=torch.autograd.grad(v.sum(),params,retain_graph=True,allow_unused=True)
 flat=torch.cat([g.flatten() for g in gs if g is not None])
 return dict(requires_grad=True,norm=float(flat.norm()),finite=bool(flat.isfinite().all()))
S=task.initial_state(2);xi=torch.zeros(2,task.robot.xi_dim);xi[:,0]=1
ap=(S[4]-task.mdp.default_q)/task.alpha
history=task.mdp.obs(S,ap,xi)[:,None].repeat(1,10,1)
ctx=task.context(S,*sample(0,2,'cpu'))
raw=pi(history,task.command(S,ctx));a,_=task.compose_action(raw,xi)
S1=wm.step(S,a,normalize_xi(xi),task.mdp.scan(S,flat_boxes(S)))
terms,contrib=task.reward_terms(S1,a,ap,ctx,raw=raw,xi=xi)
out=dict(checkpoint=str(path),scope='CPU eager; two nominal initial states, frozen production runner; output-layer gradients, not replay-distribution or compiled audit',config={k:cfg.get(k) for k in ['curriculum_level','policy_training_scope','actor_value_gradient','soft_termination_mode','bound_value_bootstrap']},terms={})
for k in terms:out['terms'][k]=dict(value=float(terms[k].mean().detach()),raw_gradient=grad(terms[k]),weighted_gradient=grad(contrib[k]))
out['aim_if_enabled']=grad(terms['aim']*4*task.dt)
requested=task.mdp.default_q+xi[:,1:30]+task.alpha*raw
saturated=(requested<task.mdp.soft_lo)|(requested>task.mdp.soft_hi)
action_derivative=torch.autograd.grad(a.sum(),raw,retain_graph=True)[0]
out['action_clip']=dict(saturated_count=int(saturated.sum()),total=saturated.numel(),zero_derivative_count=int((action_derivative==0).sum()),request_limit_weight=task.weights.get('request_limit',0.))
out['soft_termination']=[]
from supermimic.rotations import so3_exp
for deg in [0.,28.,29.,30.,31.,32.,35.]:
 angle=torch.tensor(deg*3.141592653589793/180,requires_grad=True)
 rv=torch.stack((angle*0,angle,angle*0))[None].expand(2,-1)
 test=(S[0],so3_exp(rv),*S[2:])
 d=task.soft_termination(test,torch.ones(2,dtype=torch.long),ctx)
 g=torch.autograd.grad(d.sum(),angle)[0]
 out['soft_termination'].append(dict(tilt_deg=deg,failure=float(d.mean().detach()),derivative_per_rad=float(g)))
alive=torch.ones(2);objective=torch.zeros(2);t=torch.zeros(2,dtype=torch.long);start=t.clone()
for j in range(32):
 S,history,ap,r,d=imagination_step(pi,wm,task,S,history,ap,t,xi,ctx,start)
 objective=objective+cfg['gamma']**j*alive*r
 alive=alive*(1-d);t=t+1
last=task.critic_features(S,history,ap,xi,t,task.advance_context(ctx,t-start))
tail=cfg['gamma']**32*alive*bound_bootstrap(critic(last),t,task,cfg)
out['rollout32']=dict(reward=float(objective.mean().detach()),tail=float(tail.mean().detach()),survival=alive.detach().tolist(),reward_gradient=grad(objective.mean()),tail_gradient=grad(tail.mean()),combined_gradient=grad((objective+tail).mean()))
(ROOT/'results/gradient_audit_20261116/probe.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
