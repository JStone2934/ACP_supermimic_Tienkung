"""Independent 30-second standing check from quiet, double-support data poses.

The tracker receives fixed contact points and a fixed root goal. No future
motion frames enter its command. This is a reference-tracker stage gate, not
acceptance of the eventual reference-free aiming policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from supermimic.acp.env import TrackingSim, make_sim, RESET_POS, RESET_ROT, RESET_JOINT_POS
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP
from supermimic.acp.motion import MotionLib
from supermimic.acp.networks import Policy
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.task import ContactCfg, ContactTask, goal_command
from supermimic.rotations import mat_to_quat


def quiet_frames(motion, ctc, limit=100, seed=0):
    """One-second quiet windows; round-robin clips, spaced by at least 1s."""
    pools=[]
    for clip,(lo,hi) in enumerate(zip(motion.clip_start.tolist(),motion.clip_end.tolist())):
        variant=int(motion.first_variants(torch.tensor([clip],device=motion.q.device))[0])
        quiet=(motion.root_v[lo:hi+1].norm(dim=-1)<=.1)&(motion.root_w[lo:hi+1].norm(dim=-1)<=.3)
        quiet &= motion.qd[lo:hi+1].abs().amax(-1)<=1.
        quiet &= ctc.labels[lo:hi+1,variant,:2].all(-1)
        if len(quiet)<50:continue
        windows=F.conv1d(quiet.float()[None,None],torch.ones(1,1,50,device=quiet.device))[0,0]>=50
        candidates=windows.nonzero().flatten().cpu().tolist()
        spaced=[]
        for f in candidates:
            if not spaced or f-spaced[-1]>=50:spaced.append(f)
        pools.append([lo+f+25 for f in spaced])
    generator=np.random.default_rng(seed)
    for pool in pools:generator.shuffle(pool)
    generator.shuffle(pools);chosen=[]
    while len(chosen)<limit and any(pools):
        for pool in pools:
            if pool and len(chosen)<limit:chosen.append(pool.pop())
    if not chosen:raise ValueError('No one-second quiet double-support windows; cannot run standing gate')
    return torch.tensor(chosen,device=motion.q.device,dtype=torch.long)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--motion-dir',required=True)
    ap.add_argument('--contacts-dir',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--select-only',action='store_true');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--count',type=int,default=100);ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    robot=get_robot('tienkung3_29dof');kin=RobotKinematics(args.device,robot)
    motion=MotionLib.from_dir(args.motion_dir,kin,args.device)
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=args.device))
    ctc=ContactTask(ContactCfg(contacts_dir=args.contacts_dir,contact_geometry='spheres224'),mdp,args.motion_dir,args.device)
    frames=quiet_frames(motion,ctc,args.count,args.seed);n=len(frames)
    seed_states=motion.ref_state(frames)
    seeds=[dict(clip=motion.names[int(motion.clip_id[f])],frame=int(f-motion.clip_start[motion.clip_id[f]]),
                initial_q=seed_states[4][i].cpu().tolist(),initial_height=float(seed_states[0][i,2]),
                initial_quat=mat_to_quat(seed_states[1][i]).cpu().tolist()) for i,f in enumerate(frames)]
    result=dict(robot_metadata=robot.metadata(),data_hash=motion.data_hash,seed=args.seed,
                motion_dir=str(Path(args.motion_dir).resolve()),contacts_dir=str(Path(args.contacts_dir).resolve()),
                quiet_window_s=1.,unique_seed_count=n,seeds=seeds,scope=__doc__)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.select_only:
        args.out.write_text(json.dumps(result,indent=2));print(f'Selected {n} distinct quiet poses');return
    # Few long aiming clips may contain only a handful of quiet poses. Evaluate
    # independent reset perturbations of those poses, and disclose both counts.
    unique_count=n
    scenario_seed_ids=torch.arange(args.count,device=args.device)%n
    frames=frames[scenario_seed_ids];n=len(frames)
    if args.checkpoint is None:ap.error('--checkpoint required for physical evaluation')
    digest=sha256(args.checkpoint);ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    robot.validate_metadata(ck['robot_metadata'])
    if ck['args'].get('hist')!=10 or ck.get('task_config') is not None:raise ValueError('Expected ten-frame, full-joint tracker')
    if ck['ctc_cfg'].get('blind') or ck['ctc_cfg'].get('contact_geometry')!='spheres224':raise ValueError('Incompatible contact command')
    pi=Policy(robot.obs_dim,93,hist=10,robot=robot).to(args.device);pi.load_state_dict(ck['pi']);pi.eval()
    env=TrackingSim(n,motion,mdp,args.device,eval_mode=True,
                    sim_cfg=make_sim(robot=robot,physics_substeps=ck['args'].get('physics_substeps')))
    ids=torch.arange(n,device=args.device)
    env.reset_noise=True
    env.reset(ids,frames)
    env.robot.write_root_link_velocity_to_sim(torch.zeros(n,6,device=args.device),env_ids=ids)
    env.robot.write_joint_velocity_to_sim(torch.zeros(n,29,device=args.device),env_ids=ids);env.refresh()
    initial=tuple(x.clone() for x in env.state());anchor=initial[0].clone()
    yaw=torch.atan2(initial[1][:,1,0],initial[1][:,0,0])
    targets=ctc.contact_point(initial,motion.boxes(frames,env.v)).clone()
    valid=torch.zeros(n,9,dtype=torch.bool,device=args.device);valid[:,:2]=True
    feet_ids=[kin.tracked_names.index(robot.contact_bodies[k][0]) for k in ('left_foot','right_foot')]
    feet0=kin.forward(*initial)[0][:,feet_ids]
    failed=torch.zeros(n,dtype=torch.bool,device=args.device)
    max_root=torch.zeros(n,device=args.device);max_feet=max_root.clone();max_tilt=max_root.clone()
    history=None;trace=[]
    for step in range(1500):
        env.f.copy_(frames);state=env.state();obs=env.observe()
        history=obs[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        cmd=ctc.command(state,frames,env.v,torch.zeros_like(frames),noise=False,targets_w=targets,valid=valid,
                        goal=goal_command(state,anchor,yaw,.5))
        action=pi(history,cmd);bad=~action.isfinite().all(-1)
        action=torch.where(action.isfinite(),action,torch.zeros_like(action))
        transition=env.step(action);state=env.state()
        finite=torch.stack([x.flatten(1).isfinite().all(-1) for x in state]).all(0)
        drift=(state[0][:,:2]-anchor[:,:2]).norm(dim=-1)
        feet=(kin.forward(*state)[0][:,feet_ids]-feet0).norm(dim=-1).amax(-1)
        tilt=torch.rad2deg(torch.acos(state[1][:,2,2].clamp(-1,1)))
        failed |= bad|~finite|transition['term']|(drift>.5)|(feet>.05)|(tilt>15.)
        max_root=torch.maximum(max_root,drift.nan_to_num(nan=1e6));max_feet=torch.maximum(max_feet,feet.nan_to_num(nan=1e6));max_tilt=torch.maximum(max_tilt,tilt.nan_to_num(nan=180.))
        if step%10==0:trace.append(torch.cat((state[0],mat_to_quat(state[1]),state[4]),-1).cpu().numpy())
        if failed.all():break
    if sha256(args.checkpoint)!=digest:raise ValueError('Checkpoint changed during evaluation')
    passed=(~failed)&(step==1499)
    result.update(checkpoint=str(args.checkpoint),checkpoint_sha256=digest,physics_substeps=env.physics_substeps,
                  success_rate=float(passed.float().mean()),passed=bool(n>=100 and passed.float().mean()>=.95),
                  scenario_count=n,unique_seed_count=unique_count,scenario_seed_ids=scenario_seed_ids.cpu().tolist(),
                  reset_noise=dict(position_half_ranges=RESET_POS,rotation_half_ranges_rad=RESET_ROT,joint_half_range_rad=RESET_JOINT_POS,initial_velocity='zero'),
                  continuous_without_reset=True,elapsed_sim_seconds=(step+1)*.02,
                  scenarios=[dict(passed=bool(passed[i]),max_root_drift_m=float(max_root[i]),max_foot_drift_m=float(max_feet[i]),max_tilt_deg=float(max_tilt[i])) for i in range(n)])
    args.out.write_text(json.dumps(result,indent=2));np.savez_compressed(args.out.with_suffix('.npz'),trajectory=np.stack(trace),sample_dt=.2)
    print(json.dumps({k:result[k] for k in ('passed','success_rate','elapsed_sim_seconds')}))


if __name__=='__main__':main()
