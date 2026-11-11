"""Bounded real-simulator curriculum check: inference only, no optimizer or training updates.

Run under torch.distributed.run for dual-GPU reset/promotion state validation.
"""
import argparse
import json
from pathlib import Path

import torch
from supermimic.acp.distributed import Runtime
from supermimic.acp.env import TrackingSim
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP, unflatten_state
from supermimic.acp.motion import MotionLib
from supermimic.acp.networks import Policy
from supermimic.acp import networks
from supermimic.acp.robots import resolve_robot
from supermimic.acp.task import ContactCfg, ContactTask
from supermimic.acp.transition_curriculum import TransitionCurriculum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--plan',required=True);p.add_argument('--log_dir',required=True)
    p.add_argument('--num_envs',type=int,default=32);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261111)
    a=p.parse_args();rt=Runtime(a);dev=rt.device
    ck=torch.load(a.checkpoint,map_location=dev,weights_only=False);robot=resolve_robot(None,ck)
    kin=RobotKinematics(dev,robot);motion=MotionLib.from_dir(ck['args']['motion_dir'],kin,dev)
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=dev))
    ctc=ContactTask(ContactCfg(**ck['ctc_cfg']),mdp,ck['args']['motion_dir'],dev)
    curr=TransitionCurriculum(a.plan,motion,ctc,a.num_envs,dev,rt)
    pi=Policy(robot.obs_dim,ctc.cmd_dim,hist=ck['args']['hist'],robot=robot).to(dev)
    pi.load_state_dict(ck['pi']);pi.eval();rt.register({'pi':pi},{})
    networks.AMP_DTYPE=torch.bfloat16 if ck['args']['amp']=='bf16' else None
    env=TrackingSim(a.num_envs,motion,mdp,dev,runtime=rt,transition_curriculum=curr)
    env.phase.copy_(ck['curriculum_phase']);env.reset(torch.arange(a.num_envs,device=dev))
    assert env.max_episode_steps>500 and not env.curriculum
    hist=None;gs=torch.full((a.num_envs,),50,device=dev);n_done=0;local_truncations=0
    with torch.no_grad():
        for step in range(a.steps):
            o=env.observe()
            hist=(o[:,None].repeat(1,pi.hist,1) if hist is None else
                  torch.where((env.ep_step==0)[:,None,None],o[:,None].expand(-1,pi.hist,-1),torch.cat((hist[:,1:],o[:,None]),1)))
            action=pi(hist,ctc.command(env.state(),env.f,env.v,gs,noise=False))
            end=curr.episode_end.clone();kind=curr.kind.clone()
            out=env.step(action);S1=unflatten_state(out['s1']);f1=out['f']+1
            torch.testing.assert_close(out['r'],mdp.reward(S1,f1,out['a'],out['a_prev']),rtol=1e-4,atol=1e-6)
            assert torch.equal(out['term'],mdp.termination(S1,f1))
            assert torch.equal(out['done'],out['term']|(f1>=end))
            n_done+=int(out['done'].sum());local_truncations+=int(((kind==0)&(f1>=end)&~out['term']).sum())
            if step%32==0:rt.print('simulation step',step,flush=True)
    synced=rt.check_sync({'transition_curriculum':curr.state_dict()})
    # Verify serialization and restoration of shared progress without saving/updating weights.
    state=curr.state_dict()
    restored=TransitionCurriculum(a.plan,motion,ctc,a.num_envs,dev,rt);restored.load_state_dict(state)
    for k in ('start','attempts','successes','promotions','totals','quality_counts'):
        torch.testing.assert_close(getattr(curr,k),getattr(restored,k),rtol=0,atol=0)
    totals=rt.sum(torch.tensor([n_done,local_truncations],device=dev)).tolist()
    rt.write_json('curriculum_check.json',dict(status='passed',world_size=rt.world_size,steps=a.steps,
        checkpoint=a.checkpoint,optimizer_updates=0,done_episodes=totals[0],local_truncations=totals[1],
        unchanged_reward_and_failure_termination=True,state_sha256=synced,**curr.summary()))
    rt.close()

if __name__=='__main__':main()
