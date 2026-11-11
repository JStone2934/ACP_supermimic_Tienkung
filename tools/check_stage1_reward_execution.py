"""Check real Tienkung paper reward and state gradients across execution chunk sizes."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.robots import get_robot
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP
from supermimic.acp.task import ContactCfg, ContactTask, PaperReward, PaperRewardCfg
from supermimic.acp.memory import reward_batch, checkpoint_reward


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(224)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    chosen={}
    for entry in manifest['entries']:
        chosen.setdefault(entry['source_key'].split('/')[0],entry['motion'])
    robot=get_robot('tienkung3_29dof');kin=RobotKinematics('cpu',robot)
    motion=MotionLib(list(chosen.values()),kin,'cpu',str(args.dataset/'motions'))
    mdp=MDP(kin,motion,torch.tensor(robot.default_q))
    task=ContactTask(ContactCfg(contacts_dir=str(args.dataset/'contacts'),contact_geometry='spheres224'),mdp,str(args.dataset/'motions'),'cpu')
    reward=PaperReward(PaperRewardCfg(),task)
    f=torch.linspace(1,motion.num_frames-2,129).long();v=torch.zeros_like(f)
    ref=motion.ref_state(f)
    state=tuple((x+.001*torch.randn_like(x)).requires_grad_() for x in ref)
    previous=tuple(x.detach().clone() for x in ref)
    action=(torch.randn(129,robot.num_joints)*.01).requires_grad_()
    old=torch.zeros_like(action);xi=torch.zeros(129,robot.xi_dim)
    def fn(S,Sp,f,v,a,ap,app,xi):
        return reward(S,Sp,f,v,a,ap,app,xi),mdp.soft_termination(S,f)
    inputs=(state,previous,f,v,action,old,old,xi)
    base=reward_batch(fn,*inputs,chunk_size=0)
    grad=torch.autograd.grad(sum(x.sum() for x in base),(*state,action))
    result={}
    for chunk in (64,256,1024):
        value=reward_batch(checkpoint_reward(fn),*inputs,chunk_size=chunk)
        actual=torch.autograd.grad(sum(x.sum() for x in value),(*state,action))
        for a,b in zip(value,base):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        for a,b in zip(actual,grad):torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
        result[str(chunk)]=dict(max_output_abs=max(float((a-b).abs().max().detach()) for a,b in zip(value,base)),
                                max_gradient_abs=max(float((a-b).abs().max()) for a,b in zip(actual,grad)))
    out=dict(passed=True,robot=robot.robot_id,contact_geometry='spheres224',poses=129,categories=list(chosen),
             check='actual PaperReward plus soft termination, state and action gradients; CPU FP32; nondivisible chunk boundary',comparisons=result)
    args.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
