"""Record deterministic short G1 checkpoint playback, also runnable against the pre-migration worktree."""
import argparse
import dataclasses
import glob
import json
import os
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.g1 import MOTION_DIR, default_joint_pos_list
from supermimic.acp.kinematics import G1Kinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP, OBS_DIM
from supermimic.acp.task import ContactTask, ContactCfg
from supermimic.acp.env import TrackingSim
from supermimic.acp.networks import Policy


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True); ap.add_argument('--compare'); args=ap.parse_args()
    torch.set_num_threads(8); torch.manual_seed(0); dev='cuda:0'; ck=torch.load(args.checkpoint,map_location=dev)
    motion_dir=ck['args'].get('motion_dir') or MOTION_DIR
    files=sorted(glob.glob(os.path.join(motion_dir,'motions-locomotion/*_motion.npz')))[:3]
    kin=G1Kinematics(dev); motion=MotionLib(files,kin,dev,motion_dir)
    mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev)); cfg=ContactCfg(**ck['ctc_cfg'])
    ctc=ContactTask(cfg,mdp,motion_dir,dev); pi=Policy(OBS_DIM,ctc.cmd_dim,hist=ck['args']['hist']).to(dev)
    pi.load_state_dict(ck['pi']); pi.eval()
    env=TrackingSim(len(files),motion,mdp,dev,eval_mode=True); env.reset(torch.arange(len(files),device=dev),motion.clip_start)
    hist=None; actions=[]; states=[]; rewards=[]
    with torch.no_grad():
        for _ in range(20):
            obs=env.observe(); hist=obs[:,None].repeat(1,pi.hist,1) if hist is None else torch.cat((hist[:,1:],obs[:,None]),1)
            cmd=ctc.command(env.state(),env.f,env.v,torch.full((len(files),),50,device=dev),noise=False)
            a=pi(hist,cmd); out=env.step(a)
            actions.append(a.cpu().numpy()); states.append(out['s1'].cpu().numpy()); rewards.append(out['r'].cpu().numpy())
    output=dict(actions=np.array(actions),states=np.array(states),rewards=np.array(rewards))
    np.savez(args.out,**output)
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),clips=motion.names,steps=20,finite=all(np.isfinite(x).all() for x in output.values()))
    if args.compare:
        ref=np.load(args.compare); report['max_errors']={k:float(np.abs(x-ref[k]).max()) for k,x in output.items()}
        report['passed']=report['finite'] and all(x<1e-3 for x in report['max_errors'].values())
    else: report['passed']=report['finite']
    Path(args.out+'.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
