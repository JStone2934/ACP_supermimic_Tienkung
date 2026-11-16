"""Read-only finite action-response comparison; diagnostic, not policy acceptance.

Position prediction on observed actions does not establish that action derivatives
are accurate. Paired physical worlds start from identical recorded states, with
equal cleared solver warm starts, and differ in one PD joint target only.
Finite differences cross contact boundaries and are not exact contact Jacobians.
"""
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.rotations import mat_to_quat
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody import flat_boxes
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.mdp import unflatten_state,normalize_xi
from supermimic.acp.networks import WorldModel
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.wide_curriculum import sample
from supermimic.acp.robots import sha256


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(913)
    b=Path('results/lafan_jwp_aiming_20261115');dev='cuda:1'
    source=Path('logs/tienkung3_29dof/lafan_jwp_20261115/balance_recovery_quiet/train/ckpt_001500.pt')
    digest=sha256(source);ck=torch.load(source,map_location=dev,weights_only=False)
    task=WideAimingTask(dev,ck['config']);wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(dev)
    wm.load_state_dict(ck['wm']);freeze_normalizers(wm);wm.eval()
    data=np.load(b/'balance_pre_actor_horizon_validation.npz')
    times=np.array([0,15,30,50,75,100,150,200]);worlds=np.arange(8)
    states=unflatten_state(torch.as_tensor(data['s'][times,worlds],device=dev))
    actions=torch.as_tensor(data['a'][times,worlds],device=dev)
    n=len(times)*29*2;env=FullbodySim(task,n,lambda count:sample(0,count,dev),eval_mode=True)
    ids=torch.arange(n,device=dev);S=tuple(x.repeat_interleave(58,0) for x in states)
    p,R,v,w,q,qd=S
    perturb=torch.eye(29,device=dev)[None,:,None,:]*torch.tensor([-1.,1.],device=dev)[None,None,:,None]
    result=dict(scope=__doc__,checkpoint_sha256=digest,weights_changed=False,policy_acceptance=False,
                recorded_time_indices=times.tolist(),recorded_world_indices=worlds.tolist(),responses={})
    arrays={}
    for eps in [.005,.02]:
        env.sim.reset(ids);env.scene.reset(ids)
        env.robot.write_root_link_pose_to_sim(torch.cat((p+env.scene.env_origins,mat_to_quat(R)),-1),env_ids=ids)
        env.robot.write_root_link_velocity_to_sim(torch.cat((v,w),-1),env_ids=ids)
        env.robot.write_joint_position_to_sim(q[:,env.to_internal],env_ids=ids)
        env.robot.write_joint_velocity_to_sim(qd[:,env.to_internal],env_ids=ids)
        env.sim.forward();initial=tuple(x.clone() for x in env.state())
        raw=(actions[:,None,None,:]+eps*perturb/task.alpha).reshape(n,29)
        action,_=task.compose_action(raw,env.xi)
        predicted=wm.step(initial,action,normalize_xi(env.xi),task.mdp.scan(initial,flat_boxes(initial)))
        transition=env.step(raw);physical=unflatten_state(transition['s1'])
        values=[]
        for next_state in [physical,predicted]:
            delta=wm.targets(initial,next_state).reshape(len(times),29,2,-1)
            values.append((delta[:,:,1]-delta[:,:,0])/(2*eps))
        truth,estimate=values;summary={}
        for name,lo,hi in [('root_position',0,3),('root_rotation',3,6),('root_velocity',6,9),
                           ('root_angular_velocity',9,12),('joint_position',12,41),('joint_velocity',41,70)]:
            x,y=truth[...,lo:hi].flatten(),estimate[...,lo:hi].flatten()
            summary[name]=dict(cosine=float((x*y).sum()/(x.norm()*y.norm()).clamp_min(1e-12)),
                physical_rms=float(x.square().mean().sqrt()),predicted_rms=float(y.square().mean().sqrt()),
                relative_l2_error=float((x-y).norm()/x.norm().clamp_min(1e-12)))
        j=torch.arange(29,device=dev);x=truth[:,j,41+j];y=estimate[:,j,41+j]
        valid=x.abs()>1e-3
        summary['direct_joint_velocity_sign_agreement']=float(((x*y>0)&valid).sum()/valid.sum().clamp_min(1))
        summary['per_joint']=[dict(name=name,physical_response=float(x[:,i].mean()),model_response=float(y[:,i].mean()))
                              for i,name in enumerate(task.robot.joint_names)]
        result['responses'][str(eps)]=summary
        arrays[f'physical_{eps}']=truth.cpu().numpy();arrays[f'predicted_{eps}']=estimate.cpu().numpy()
        print(json.dumps({'epsilon_rad':eps,**{k:v for k,v in summary.items() if k!='per_joint'}}),flush=True)
    assert sha256(source)==digest
    (b/'balance_action_response.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(b/'balance_action_response.npz',**arrays)


if __name__=='__main__':main()
