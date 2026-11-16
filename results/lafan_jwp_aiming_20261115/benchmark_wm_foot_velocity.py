"""Read-only eager/compiled gradient check for optional Cartesian foot-velocity supervision."""
import json,time
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.train_aiming_fullbody import wm_loss
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.robots import sha256


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    b=Path('results/lafan_jwp_aiming_20261115');run=Path('logs/tienkung3_29dof/lafan_jwp_20261115/balance_recovery_conservative/train')
    path=run/'ckpt_002500.pt';digest=sha256(path);dev='cuda:1'
    ck=torch.load(path,map_location=dev,weights_only=False);task=WideAimingTask(dev,dict(ck['config'],wm_foot_velocity_weight=.05))
    wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(dev);wm.load_state_dict(ck['wm']);freeze_normalizers(wm)
    recorded=b/'balance_conservative2500_late_dynamics.npz';data=np.load(recorded)
    windows={k:torch.as_tensor(np.concatenate([data[k][s:s+16].transpose(1,0,2) for s in [500,1000]],0).copy(),device=dev) for k in ['s','s1','a','xi']}
    done=torch.as_tensor(np.concatenate([data['done'][s:s+16].T for s in [500,1000]],0).copy(),device=dev)
    windows['mask']=torch.cat((torch.ones(len(done),1,device=dev),(~done[:,:-1]).float().cumprod(1)),1)
    fast=torch.compile(wm_loss,dynamic=False,options={'max_fusion_size':32});params=list(wm.parameters())
    def evaluate(fn):
        torch.cuda.synchronize();start=time.monotonic();loss=fn(wm,task,windows);grad=torch.autograd.grad(loss,params)
        torch.cuda.synchronize();return loss,grad,time.monotonic()-start
    loss0,grad0,eager_cold=evaluate(wm_loss);loss1,grad1,compiled_cold=evaluate(fast)
    relative=float((sum((a-b).square().sum() for a,b in zip(grad0,grad1))/sum(a.square().sum() for a in grad0).clamp_min(1e-12)).sqrt())
    result=dict(scope=__doc__,checkpoint_sha256=digest,recorded_transitions=str(recorded),recorded_transitions_sha256=sha256(recorded),
        weights_changed=False,policy_acceptance=False,wm_foot_velocity_weight=.05,velocity_scale_m_s=.05,window_steps=16,batch=len(done),matmul_tf32=False,
        loss_eager=float(loss0.detach()),loss_compiled=float(loss1.detach()),parameter_gradient_relative_l2_error=relative,
        eager_cold_s=eager_cold,compiled_cold_s=compiled_cold,passed=False)
    out=b/'wm_foot_velocity_compile_benchmark.json';out.write_text(json.dumps(result,indent=2))
    torch.testing.assert_close(loss0,loss1,rtol=1e-3,atol=1e-5);assert relative<.005,relative
    result['eager_warm_median_s']=float(np.median([evaluate(wm_loss)[2] for _ in range(3)]))
    result['compiled_warm_median_s']=float(np.median([evaluate(fast)[2] for _ in range(3)]))
    assert sha256(path)==digest;result['passed']=True;out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
