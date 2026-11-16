"""Validate robot-specific ACP data, contact reward/backward and GPU reset/step.

The default remains the existing G1 setup. Synthetic fixtures establish numerical
integration only; resets and falls are measured and are not a learned policy result.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import torch
import mujoco
import numpy as np
from supermimic.acp.env import TrackingSim
from supermimic.acp.robots import get_robot
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP, normalize_xi, unflatten_state
from supermimic.acp.motion import MotionLib
from supermimic.acp.task import ContactCfg, ContactTask, PaperReward, PaperRewardCfg
from supermimic.acp.networks import Policy, WorldModel, Critic
from supermimic.acp.buffer import ReplayBuffer


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot',choices=['g1','tienkung3_29dof'],default='g1')
    ap.add_argument('--motion-dir'); ap.add_argument('--contacts-dir')
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--num-envs',type=int,default=16)
    ap.add_argument('--steps',type=int,default=16); ap.add_argument('--out')
    args=ap.parse_args()
    if args.num_envs<1 or args.steps<1: raise ValueError('positive environments and steps required')
    torch.set_num_threads(8); torch.manual_seed(0); dev=args.device; start=time.time()
    robot=get_robot(args.robot); motion_dir=args.motion_dir or robot.motion_dir
    print('Loading robot, motion and contacts...',flush=True)
    kin=RobotKinematics(dev,robot); motion=MotionLib.from_dir(motion_dir,kin,dev)
    mdp=MDP(kin,motion,torch.tensor(robot.default_q,device=dev))
    ctc=ContactTask(ContactCfg(contacts_dir=args.contacts_dir),mdp,motion_dir,dev)
    reward=PaperReward(PaperRewardCfg(),ctc)
    print('Building GPU simulation...',flush=True)
    env=TrackingSim(args.num_envs,motion,mdp,dev,reset_noise=False)
    assert env.sim.mj_model.nu==robot.num_joints
    # Independent CPU affine-actuator oracle, including saturation and canonical mapping.
    mm=env.sim.mj_model; dd=mujoco.MjData(mm); dd.qpos[2]=3.
    rng=np.random.default_rng(0); expected=[]
    for aid in range(mm.nu):
        jid=mm.actuator_trnid[aid,0]; name=mm.joint(jid).name.rsplit('/',1)[-1]
        j=robot.joint_names.index(name); qa=mm.jnt_qposadr[jid]; va=mm.jnt_dofadr[jid]
        dd.qpos[qa]=robot.default_q[j]+rng.uniform(-.05,.05); dd.qvel[va]=rng.uniform(-1,1)
        target=robot.default_q[j]+.01+robot.action_scales[j]*rng.uniform(-8,8)
        dd.ctrl[aid]=target
        expected.append(np.clip(robot.kp[j]*(target-dd.qpos[qa])-robot.kd[j]*dd.qvel[va],-robot.effort[j],robot.effort[j]))
    mujoco.mj_forward(mm,dd)
    force_error=float(np.max(np.abs(dd.actuator_force-np.asarray(expected))))
    if force_error>1e-8: raise AssertionError(f'PD law differs from reward: {force_error}')
    gs=ctc.sample_goal_steps(args.num_envs)
    buffer=ReplayBuffer(min(args.steps,16),args.num_envs,dev,robot=robot)
    falls=resets=0; max_torque=0.; min_z=float('inf')
    def finite(name,x):
        if not torch.isfinite(x).all(): raise AssertionError(f'non-finite {name}')
    for i in range(args.steps):
        with torch.no_grad():
            obs=env.observe(); S=env.state(); f,v=env.f.clone(),env.v.clone()
            cmd=ctc.command(S,f,v,gs,noise=False)
            finite('observation',obs); finite('command',cmd)
            a=torch.zeros(args.num_envs,robot.num_joints,device=dev)
            out=env.step(a)
            for key in ('s','s1','r','o','xi'): finite(key,out[key])
            S1=unflatten_state(out['s1'])
            pr=reward(S1,S,f+1,v,a,a,a,out['xi']); finite('paper reward',pr)
            out['goal_steps']=gs; buffer.add(out)
            falls+=int(out['term'].sum()); resets+=int(out['done'].sum())
            min_z=min(min_z,float(S1[0][:,2].min()))
            forces=env.sim.data.actuator_force
            finite('actuator force',forces); max_torque=max(max_torque,float(forces.abs().max()))
        if i%100==0: print(f'{i+1}/{args.steps} steps, resets={resets}',flush=True)
    print('Checking policy/world model/critic/reward gradients and compilation...',flush=True)
    N=min(args.num_envs,4); f=env.f[:N].clone(); v=env.v[:N].clone(); S=tuple(x[:N].detach() for x in env.state())
    xi=env.xi[:N]; zeros=torch.zeros(N,robot.num_joints,device=dev); obs=mdp.obs(S,zeros,xi)
    cmd=ctc.command(S,f,v,gs[:N],noise=False)
    pi=Policy(mdp.obs_dim,ctc.cmd_dim,hist=2,robot=robot).to(dev)
    wm=WorldModel(mdp.scan_dim,hidden=(128,128),robot=robot).to(dev)
    # Nonzero last layer exercises policy -> dynamics -> analytic reward gradients.
    torch.nn.init.normal_(wm.net[-1].weight,std=1e-4)
    wm.out_norm.var.fill_(1e-3)
    a=pi(obs[:,None].expand(-1,2,-1),cmd)
    a.retain_grad()
    step=torch.compile(wm.step)
    S1=step(S,a,normalize_xi(xi),mdp.scan(S,motion.boxes(f,v)))
    r=reward(S1,S,f,v,a,zeros,zeros,xi)
    critic=Critic(robot.inv_dim+mdp.critic_extra+ctc.cmd_dim).to(dev)
    value=critic(torch.cat((mdp.critic_features(S1,f,normalize_xi(xi),motion.boxes(f,v)),cmd),-1))
    loss=-r.mean()+value.square().mean()+wm.targets(S,S1).square().mean()
    finite('loss',loss); loss.backward()
    finite('action gradient',a.grad)
    counts={}
    for name,net in (('policy',pi),('world_model',wm),('critic',critic)):
        grads=[p.grad for p in net.parameters() if p.grad is not None]
        if not grads: raise AssertionError(f'{name}: missing gradients')
        for g in grads: finite(name+' gradient',g)
        counts[name]=len(grads)
    if args.steps>=2: buffer.sample_windows(2,2)
    torch.cuda.synchronize(dev)
    report=dict(status='PASS',robot_metadata=robot.metadata(),data_hash=motion.data_hash,
                versions={n:importlib.metadata.version(n) for n in ('torch','mjlab','mujoco','mujoco-warp','warp-lang')},
                gpu=torch.cuda.get_device_name(dev),seed=0,clips=len(motion.files),frames=motion.num_frames,
                num_envs=args.num_envs,control_steps=args.steps,contact_command_dim=ctc.cmd_dim,
                dimensions=dict(state=robot.state_dim,xi=robot.xi_dim,obs=robot.obs_dim,wm_out=robot.wm_out),
                gradient_tensors=counts,torch_compile_backward='PASS',falls=falls,resets=resets,min_root_z=min_z,
                max_abs_actuator_force=max_torque,pd_force_max_error=force_error,elapsed_s=time.time()-start,
                scope='numerical integration only; zero-action resets/falls are not trained-policy evaluation')
    if args.out:
        p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
