"""Counterfactual pose-command sensitivity at fixed reference proprioception."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP
from supermimic.acp.task import ContactTask,ContactCfg
from supermimic.acp.networks import Policy


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    digest=sha256(a.checkpoint);ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if not ck['ctc_cfg'].get('reference_pose'):raise ValueError('Explicit reference-pose interface required')
    robot=get_robot('tienkung3_29dof');kin=RobotKinematics(a.device,robot)
    motion=MotionLib.from_dir(str(a.dataset/'motions'),kin,a.device)
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=a.device))
    cfg=ck['ctc_cfg']|dict(contacts_dir=str(a.dataset/'contacts'))
    ctc=ContactTask(ContactCfg(**cfg),mdp,str(a.dataset/'motions'),a.device)
    pi=Policy(99,ctc.cmd_dim,hist=10,robot=robot,pose_action_skip=ck['args'].get('pose_action_skip',False)).to(a.device)
    pi.load_state_dict(ck['pi']);pi.eval()
    frames=torch.arange(0,motion.num_frames-1,97,device=a.device)
    frames=torch.minimum(frames,motion.clip_end[motion.clip_id[frames]]-1)
    xi=torch.zeros(len(frames),robot.xi_dim,device=a.device);xi[:,0]=1
    alpha=torch.tensor(robot.action_scales,device=a.device);history=[]
    for offset in range(-9,1):
        f=torch.maximum(frames+offset,motion.clip_start[motion.clip_id[frames]])
        history.append(mdp.obs(motion.ref_state(f),(motion.q[f]-mdp.default_q)/alpha,xi))
    history=torch.stack(history,1)
    command=ctc.command(motion.ref_state(frames),frames,torch.zeros_like(frames),torch.full_like(frames,50),noise=False)
    target=command[:,-29:];requested=mdp.default_q+alpha*pi(history,command)
    altered=command.clone();generator=torch.Generator(device=a.device).manual_seed(123)
    altered[:,-29:]=target[torch.randperm(len(frames),device=a.device,generator=generator)]
    delta=alpha*(pi(history,altered)-pi(history,command));goal_delta=altered[:,-29:]-target
    latent=pi.enc(pi.goal_norm(command)).tanh()
    report=dict(checkpoint_sha256=digest,data_hash=motion.data_hash,samples=len(frames),permutation_seed=123,
        reference_state_target_rmse_rad=float((requested-target).square().mean().sqrt()),
        command_change_rmse_rad=float(goal_delta.square().mean().sqrt()),response_rmse_rad=float(delta.square().mean().sqrt()),
        response_gain=float((delta*goal_delta).sum()/goal_delta.square().sum()),
        latent_saturation_fraction=float((latent.abs()>.95).float().mean()),pose_action_skip=hasattr(pi,'pose_skip'),
        scope=__doc__+' Diagnostic only, not physical acceptance')
    if sha256(a.checkpoint)!=digest:raise ValueError('Checkpoint changed')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
