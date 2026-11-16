"""Read-only eager/compiled loss and parameter-gradient equivalence on recorded physical states."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import FullbodyPolicy
from supermimic.acp.networks import WorldModel
from supermimic.acp.train_aiming_fullbody import wm_loss,imagination_step,compiled_imagination_step
from supermimic.acp.mdp import unflatten_state
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.robots import sha256


def main():
    torch.set_num_threads(2);torch.manual_seed(82);torch.backends.cuda.matmul.allow_tf32=False
    root=Path.cwd();b=root/'results/lafan_jwp_aiming_20261115';dev='cuda:1'
    path=b/'precise_translated_load_candidates/ckpt_000002.pt'
    digest=sha256(path);ck=torch.load(path,map_location=dev,weights_only=False)
    cfg=dict(ck['config'],soft_termination_mode='compact',curriculum_level=5,policy_training_scope='body_outputs')
    task=WideAimingTask(dev,cfg);pi=FullbodyPolicy(task.robot).to(dev);pi.load_state_dict(ck['pi'])
    from supermimic.acp.aiming_adapters import configure_policy_training
    configure_policy_training(pi,cfg)
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(dev);wm.load_state_dict(ck['wm']);freeze_normalizers(pi,wm)
    data=np.load(root/'logs/tienkung3_29dof/lafan_jwp_20261115/balance_recovery_conservative/train/wm_validation/003250.npz')
    windows={k:torch.as_tensor(np.concatenate([data[k][start:start+8].transpose(1,0,2) for start in [0,64]],0).copy(),device=dev) for k in ['s','s1','a','xi']}
    done=torch.as_tensor(np.concatenate([data['done'][start:start+8].T for start in [0,64]],0).copy(),device=dev)
    windows['mask']=torch.cat((torch.ones(16,1,device=dev),(~done[:,:-1]).float().cumprod(1)),1)
    report={'scope':'Read-only prospective body-output stage5 compile test; not promotion or physical policy acceptance','checkpoint_sha256':digest,'weights_changed':False,'formal_training':False,'matmul_tf32':False,'soft_termination_mode':'compact','comparisons':{}}
    def measure(name,eager,compiled,params):
        torch.cuda.synchronize();start=time.monotonic();le=eager();ge=torch.autograd.grad(le,params)
        torch.cuda.synchronize();et=time.monotonic()-start
        start=time.monotonic();lc=compiled();gc=torch.autograd.grad(lc,params)
        torch.cuda.synchronize();cold=time.monotonic()-start
        errors=[(x-y).square().sum() for x,y in zip(ge,gc)]
        relative=float((sum(errors)/sum(x.square().sum() for x in ge).clamp_min(1e-12)).sqrt())
        report['comparisons'][name]={'passed':False,'loss_eager':float(le.detach()),'loss_compiled':float(lc.detach()),'parameter_gradient_relative_l2_error':relative}
        (b/'body_output_compile_benchmark.json').write_text(json.dumps(report,indent=2))
        torch.testing.assert_close(le,lc,rtol=1e-3,atol=1e-5)
        assert relative<.005,(name,relative)
        assert len(ge)==2 and all(g.isfinite().all() for g in ge+gc)
        assert all(torch.count_nonzero(g[22:29])==0 for g in ge+gc), 'Frozen aiming rows received a gradient'
        assert all(g[:22].square().sum()>0 for g in ge), 'Body outputs must receive a learning gradient'
        eager_timings=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.monotonic();loss=eager();torch.autograd.grad(loss,params)
            torch.cuda.synchronize();eager_timings.append(time.monotonic()-start)
        timings=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.monotonic();loss=compiled();torch.autograd.grad(loss,params)
            torch.cuda.synchronize();timings.append(time.monotonic()-start)
        report['comparisons'][name]={'passed':True,'loss_eager':float(le.detach()),'loss_compiled':float(lc.detach()),
            'parameter_gradient_relative_l2_error':relative,'eager_cold_s':et,'eager_warm_median_s':float(np.median(eager_timings)),'compiled_cold_s':cold,'compiled_warm_median_s':float(np.median(timings))}
        (b/'body_output_compile_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['comparisons'][name]),flush=True)
    wm.requires_grad_(False)
    S=unflatten_state(windows['s'][:,0]);xi=windows['xi'][:,0];ap=(S[4]-task.mdp.default_q)/task.alpha
    hist=task.mdp.obs(S,ap,xi)[:,None].repeat(1,10,1);t=torch.zeros(16,device=dev,dtype=torch.long)
    ctx=task.context(S,torch.tensor([[3.,0.,0.,.15]],device=dev).expand(16,-1),4.)
    ctx['support_age_steps'][:8]=-1;ctx['support_age_steps'][8:]=24
    ctx['feet_anchor_w'][:8,:,0]-=.02
    t.fill_(24)
    report['support_test']='Half locked with2cm position error; half immediately before capture boundary'
    def eager_step(*args):return checkpoint(lambda *inputs:imagination_step(pi,wm,task,*inputs),*args,use_reentrant=False)
    fast_step=compiled_imagination_step(pi,wm,task)
    def objective(step,horizon):
        state,history,action,clock=S,hist,ap,t;alive=torch.ones(16,device=dev);reward=alive*0
        for j in range(horizon):
            state,history,action,r,d=step(state,history,action,clock,xi,ctx,t)
            reward=reward+.99**j*alive*r;alive=alive*(1-d);clock=clock+1
        return -reward.mean()
    for horizon in [32]:
        measure(f'actor{horizon}',lambda:objective(eager_step,horizon),lambda:objective(fast_step,horizon),[p for p in pi.parameters() if p.requires_grad])
    assert sha256(path)==digest
    report['passed']=True;(b/'body_output_compile_benchmark.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
