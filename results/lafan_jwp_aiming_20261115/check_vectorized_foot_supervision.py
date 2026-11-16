"""Read-only parity check for batching auxiliary FK over a dynamics window."""
import json,time
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.mdp import unflatten_state,normalize_xi
from supermimic.acp.aiming_fullbody import flat_boxes
from supermimic.acp.train_aiming_fullbody import wm_loss
from supermimic.acp.robots import sha256


def vectorized_loss(wm,task,windows):
    S=unflatten_state(windows['s'][:,0]);loss=S[0].new_zeros(());scale=wm.out_norm.std().clamp_min(.003);predictions=[]
    weight=task.config.get('wm_foot_velocity_weight',0.)
    for j in range(windows['a'].shape[1]):
        truth=unflatten_state(windows['s'][:,j])
        S=tuple(torch.where(windows['mask'][:,j].reshape(-1,*([1]*(x.ndim-1))).bool(),x,y) for x,y in zip(S,truth))
        S=wm.step(S,windows['a'][:,j],normalize_xi(windows['xi'][:,j]),task.mdp.scan(S,flat_boxes(S)))
        err=(wm.state_error(S,unflatten_state(windows['s1'][:,j]))/scale).square().mean(-1)
        loss+=(err*windows['mask'][:,j]).sum()
        if weight:predictions.append(S)
    if weight:
        predicted=tuple(torch.stack([s[k] for s in predictions],1).flatten(0,1) for k in range(6))
        target=unflatten_state(windows['s1'].flatten(0,1))
        pv=task.kin.forward(*predicted)[2][:,task.feet];tv=task.kin.forward(*target)[2][:,task.feet]
        err=((pv-tv)/.05).square().mean((-1,-2))
        loss+=weight*(err*windows['mask'].flatten()).sum()
    return loss/windows['mask'].sum().clamp_min(1)


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    b=Path('results/lafan_jwp_aiming_20261115');ckpath=Path('logs/tienkung3_29dof/lafan_jwp_20261115/balance_recovery_conservative/train/ckpt_002500.pt');ck=torch.load(ckpath,map_location='cuda:1',weights_only=False)
    task=WideAimingTask('cuda:1',dict(ck['config'],wm_foot_velocity_weight=.05));wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to('cuda:1');wm.load_state_dict(ck['wm']);freeze_normalizers(wm)
    data=np.load(b/'balance_conservative2500_late_dynamics.npz')
    windows={k:torch.tensor(np.concatenate([data[k][s:s+16].transpose(1,0,2) for s in [500,1000]],0),device='cuda:1') for k in ['s','s1','a','xi']}
    done=torch.tensor(np.concatenate([data['done'][s:s+16].T for s in [500,1000]],0),device='cuda:1');windows['mask']=torch.cat((torch.ones(len(done),1,device='cuda:1'),(~done[:,:-1]).float().cumprod(1)),1)
    results=[];grads=[]
    for fn in [wm_loss,vectorized_loss]:
        torch.cuda.synchronize();start=time.monotonic();loss=fn(wm,task,windows);g=torch.autograd.grad(loss,list(wm.parameters()));torch.cuda.synchronize()
        results.append(dict(loss=float(loss.detach()),elapsed_s=time.monotonic()-start));grads.append(g)
    a,c=grads;relative=float((sum((x-y).square().sum() for x,y in zip(a,c))/sum(x.square().sum() for x in a)).sqrt())
    torch.testing.assert_close(torch.tensor(results[0]['loss']),torch.tensor(results[1]['loss']),rtol=1e-5,atol=1e-7);assert relative<1e-4,relative
    report=dict(scope=__doc__,checkpoint_sha256=sha256(ckpath),results=results,gradient_relative_l2_error=relative,passed=True,weights_changed=False,policy_acceptance=False)
    (b/'wm_foot_velocity_vectorized_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
