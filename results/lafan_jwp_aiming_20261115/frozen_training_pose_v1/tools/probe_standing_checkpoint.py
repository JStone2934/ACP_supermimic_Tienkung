"""Zero-shot standing evaluation of an existing tracker, without training or resets."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from supermimic.acp.robots import get_robot, sha256

ARM = ('shoulder_pitch_r_joint','shoulder_roll_r_joint','shoulder_yaw_r_joint',
       'elbow_pitch_r_joint','elbow_yaw_r_joint','wrist_pitch_r_joint','wrist_roll_r_joint')
ARM_Q = (.2,-.1,0.,-np.pi/2,0.,0.,0.)


def prepare(root):
    import sys
    robot = get_robot('tienkung3_29dof')
    for pose in ('default','aiming'):
        motions=root/pose/'motions'; folder=motions/'synthetic'; folder.mkdir(parents=True,exist_ok=True)
        q=np.array(robot.default_q)
        if pose=='aiming':
            for name,value in zip(ARM,ARM_Q): q[robot.joint_names.index(name)]=value
        jp=np.tile(np.r_[0,0,robot.stand_height,1,0,0,0,q],(2001,1))
        np.savez_compressed(folder/'standing_motion.npz',joint_pos=jp,joint_names=np.array(robot.joint_names),
                            fps=np.array([50]),robot_metadata=json.dumps(robot.metadata()),
                            source='constant pose for zero-shot checkpoint evaluation')
        np.save(folder/'standing_terrain.npy',np.zeros((0,1,10)))
        subprocess.run([sys.executable,'-m','supermimic.acp.make_labels','--robot','tienkung3_29dof',
                        '--motion_dir',str(motions),'--out',str(root/pose/'contacts'),'--workers','1'],check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--data-root',type=Path,default=Path('data/tienkung3_29dof/standing_checkpoint_probe'))
    ap.add_argument('--ckpt',type=Path)
    ap.add_argument('--pose',choices=['default','aiming'],default='default')
    ap.add_argument('--action-noise',type=float,default=0.)
    ap.add_argument('--seconds',type=float,default=30.)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--out',type=Path,default=Path('results/analysis/standing_checkpoint_probe'))
    args=ap.parse_args()
    if args.prepare:
        prepare(args.data_root); return
    if args.ckpt is None: ap.error('--ckpt required')
    if not 0 < args.seconds <= 30: ap.error('--seconds must be in (0,30]')
    from supermimic.acp.env import TrackingSim
    from supermimic.acp.kinematics import RobotKinematics
    from supermimic.acp.mdp import MDP
    from supermimic.acp.motion import MotionLib
    from supermimic.acp.task import ContactCfg, ContactTask
    import supermimic.acp.networks as networks
    from supermimic.rotations import mat_to_quat
    torch.set_num_threads(8); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    dev='cuda:0'; ck=torch.load(args.ckpt,map_location='cpu')
    robot=get_robot('tienkung3_29dof'); robot.validate_metadata(ck['robot_metadata'])
    networks.AMP_DTYPE=torch.bfloat16 if ck['args'].get('amp')=='bf16' else None
    kin=RobotKinematics(dev,robot); motion_dir=args.data_root/args.pose/'motions'
    motion=MotionLib.from_dir(str(motion_dir),kin,dev)
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=dev))
    cfg=dict(ck['ctc_cfg']); cfg['contacts_dir']=str(args.data_root/args.pose/'contacts')
    ctc=ContactTask(ContactCfg(**cfg),mdp,str(motion_dir),dev)
    hist_len=ck.get('init_args',ck['args'])['hist']
    pi=networks.Policy(mdp.obs_dim,ctc.cmd_dim,hist=hist_len,robot=robot).to(dev)
    pi.load_state_dict(ck['pi']); pi.eval()
    env=TrackingSim(1,motion,mdp,dev,eval_mode=True,reset_noise=False)
    env.reset(torch.arange(1,device=dev),motion.clip_start.clone())
    ids=torch.tensor([robot.joint_names.index(n) for n in ARM],device=dev)
    target=torch.tensor(ARM_Q,device=dev)
    def compose(a):
        if args.pose=='aiming':
            a=a.clone(); a[:,ids]=(target-env.default_q[ids]-env.xi[:,1+ids])/env.alpha[ids]
        return a
    if args.pose=='aiming': env.a_prev[:]=compose(torch.zeros_like(env.a_prev))
    gs=torch.full((1,),50,device=dev); hist=None
    traj={k:[] for k in ('root_p','root_quat','q','f','alive','tgt_w','tgt_valid')}
    metrics={k:[] for k in ('root_drift_m','foot_drift_m','tilt_deg','arm_error_deg','root_height_m')}
    feet=[kin.tracked_names.index(robot.contact_bodies[n][0]) for n in ('left_foot','right_foot')]
    root0=env.state()[0].clone(); feet0=kin.forward(*env.state())[0][:,feet].clone()
    def record():
        S=env.state(); f=env.f
        for key,val in [('root_p',S[0]),('root_quat',mat_to_quat(S[1])),('q',S[4]),('f',f),
                        ('alive',torch.ones(1,dtype=torch.bool,device=dev)),
                        ('tgt_w',ctc.next_pos[f,env.v]),('tgt_valid',ctc.next_valid[f,env.v])]:
            traj[key].append(val.detach().cpu().numpy())
        fp=kin.forward(*S)[0][:,feet]
        metrics['root_drift_m'].append(float((S[0][:,:2]-root0[:,:2]).norm()))
        metrics['foot_drift_m'].append((fp[:,:,:2]-feet0[:,:,:2]).norm(dim=-1)[0].cpu().tolist())
        metrics['tilt_deg'].append(float(torch.rad2deg(torch.acos(S[1][0,2,2].clamp(-1,1)))))
        metrics['arm_error_deg'].append(torch.rad2deg((S[4][0,ids]-target).abs()).cpu().tolist())
        metrics['root_height_m'].append(float(S[0][0,2]))
    terminated=False; steps=0
    with torch.no_grad():
        record()
        for step in range(round(args.seconds/robot.control_dt)):
            o=env.observe(); hist=o[:,None].repeat(1,hist_len,1) if hist is None else torch.cat((hist[:,1:],o[:,None]),1)
            a=pi(hist,ctc.command(env.state(),env.f,env.v,gs,noise=False))
            if args.action_noise: a=a+args.action_noise*torch.randn_like(a)
            a=compose(a)
            if not torch.isfinite(a).all(): raise RuntimeError('nonfinite action')
            result=env.step(a); steps=step+1; record()
            if bool(result['term'][0]): terminated=True; break
            if bool(result['done'][0]): raise RuntimeError('unexpected early timeout')
    args.out.mkdir(parents=True,exist_ok=True)
    stem=f'{args.pose}_noise{args.action_noise:g}_seed{args.seed}'
    np.savez_compressed(args.out/f'{stem}.npz',robot_metadata=json.dumps(robot.metadata()),names=np.array(motion.names),
        variant=env.v.cpu().numpy(),boxes=motion.clip_boxes[torch.arange(1,device=dev),env.v].cpu().numpy(),
        **{k:np.stack(v) for k,v in traj.items()},ref_root_p=motion.root_p.cpu().numpy(),
        ref_root_R=motion.root_R.cpu().numpy(),ref_q=motion.q.cpu().numpy(),
        clip_start=motion.clip_start.cpu().numpy(),clip_len=motion.clip_len.cpu().numpy())
    report={'checkpoint':str(args.ckpt),'checkpoint_sha256':sha256(args.ckpt),'iteration':ck['iteration'],
        'robot_metadata':robot.metadata(),'pose':args.pose,'right_arm_override':args.pose=='aiming',
        'action_noise':args.action_noise,'seed':args.seed,'requested_seconds':args.seconds,
        'elapsed_sim_seconds':steps*robot.control_dt,'terminated':terminated,'continuous_without_reset':True,
        'max_root_drift_m':max(metrics['root_drift_m']),'max_foot_drift_m':np.max(metrics['foot_drift_m'],axis=0).tolist(),
        'max_tilt_deg':max(metrics['tilt_deg']),'min_root_height_m':min(metrics['root_height_m']),
        'held_arm_p95_error_deg':np.percentile(metrics['arm_error_deg'],95,axis=0).tolist() if args.pose=='aiming' else None,
        'metrics':metrics,'scope':'Zero-shot transfer to a new constant reference; no training. Action noise is diagnostic only.'}
    report['standing_30s_pass']=bool(not terminated and steps>=1500 and report['max_root_drift_m']<=.1 and max(report['max_foot_drift_m'])<=.05 and report['max_tilt_deg']<=15)
    (args.out/f'{stem}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','robot_metadata')},indent=2),flush=True)

if __name__=='__main__': main()
