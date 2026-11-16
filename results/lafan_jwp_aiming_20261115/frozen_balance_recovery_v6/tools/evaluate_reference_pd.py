"""Physical diagnostic: replay reference joint PD targets without a learned actor.

This is an oracle/reference-dependent baseline, never a delivered policy.
It measures whether a learned residual is needed for balance and motion tracking.
"""
import argparse
import json
from pathlib import Path
import torch
from supermimic.acp.env import TrackingSim,make_sim
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP,RewardCfg
from supermimic.acp.robots import get_robot


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion-dir',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    robot=get_robot('tienkung3_29dof');kin=RobotKinematics(a.device,robot)
    motion=MotionLib.from_dir(a.motion_dir,kin,a.device)
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=a.device),RewardCfg(term_root_distance=.5))
    n=len(motion.names);env=TrackingSim(n,motion,mdp,a.device,eval_mode=True,sim_cfg=make_sim(robot=robot,physics_substeps=16))
    env.reset(torch.arange(n,device=a.device),motion.clip_start.clone())
    count=torch.zeros(n,device=a.device);rs=count.clone();js=count.clone();finite=torch.ones(n,dtype=torch.bool,device=a.device)
    while not bool(env.done_eval.all()):
        f1=torch.minimum(env.f+1,motion.clip_end[motion.clip_id[env.f]])
        alive=~env.done_eval.clone()
        action=(motion.q[f1]-mdp.default_q-env.xi[:,1:30])/env.alpha
        env.step(action);state=env.state()
        finite &= ~alive|torch.stack([x.flatten(1).isfinite().all(-1) for x in state]).all(0)
        rs+=torch.where(alive,(state[0]-motion.root_p[f1]).square().sum(-1),0)
        js+=torch.where(alive,(state[4]-motion.q[f1]).square().mean(-1),0);count+=alive
        env.f=torch.minimum(env.f,motion.clip_end-1)
    rows={name:dict(completed=not bool(env.failed_eval[i]),alive_steps=int(count[i]),
                   root_rmse_m=float((rs[i]/count[i].clamp_min(1)).sqrt()),
                   joint_rmse_rad=float((js[i]/count[i].clamp_min(1)).sqrt()),finite=bool(finite[i]))
          for i,name in enumerate(motion.names)}
    result=dict(scope=__doc__,physics_substeps=16,data_hash=motion.data_hash,robot_metadata=robot.metadata(),per_clip=rows,
                completed=sum(x['completed'] for x in rows.values()),clips=n,
                mean_root_rmse_m=sum(x['root_rmse_m'] for x in rows.values())/n,
                mean_joint_rmse_rad=sum(x['joint_rmse_rad'] for x in rows.values())/n)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_clip','robot_metadata','scope')}))


if __name__=='__main__':main()
