"""Bounded integration test of distributed wide-task updates; saves no weights."""
import argparse
from pathlib import Path
import torch
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_env import FullbodySim
from supermimic.acp.aiming_fullbody import FullbodyReplay,REPLAY_FIELDS
from supermimic.acp.aiming_fullbody_io import json_save
from supermimic.acp.networks import WorldModel,Critic
from supermimic.acp.tracking_transfer import transfer_tracker
from supermimic.acp.robots import get_robot,sha256
from supermimic.acp.distributed import Runtime
from supermimic.acp.standing import freeze_normalizers
from supermimic.acp.train_aiming_fullbody import wm_loss,update_actor
from supermimic.acp.wide_curriculum import sample


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    cfg=dict(num_envs=64,wm_batch=32,pi_batch=32,horizon=4,hist=10,wm_window=4,buffer_steps=64,
        critic_epochs=1,critic_minibatches=1,seed=0,gamma=.99,lambda_return=.95,actor_grad_clip_value=.1,
        bound_value_bootstrap=True,log_dir=str(args.out.parent/'wide_runtime_smoke'),device=None)
    options=argparse.Namespace(**cfg);rt=Runtime(options)
    for key in ('num_envs','wm_batch','pi_batch'):cfg[key]=getattr(options,key)
    source_hash=sha256(args.checkpoint);ck=torch.load(args.checkpoint,map_location=rt.device,weights_only=False)
    robot=get_robot('tienkung3_29dof');task=WideAimingTask(rt.device,dict(initial_q=list(robot.default_q),initial_height=robot.stand_height,physics_substeps=16))
    pi,_=transfer_tracker(ck,robot,rt.device)
    wm=WorldModel(task.mdp.scan_dim,robot=robot).to(rt.device);wm.load_state_dict(ck['wm']);freeze_normalizers(wm)
    critic=Critic(task.critic_dim).to(rt.device)
    op=torch.optim.Adam(pi.parameters(),lr=1e-5);ow=torch.optim.Adam(wm.parameters(),lr=1e-4);ov=torch.optim.Adam(critic.parameters(),lr=1e-4)
    rt.register(dict(pi=pi,wm=wm,critic=critic),dict(pi=op,wm=ow,critic=ov))
    env=FullbodySim(task,cfg['num_envs'],lambda n:sample(0,n,rt.device))
    replay=FullbodyReplay(64,env.N,rt.device,extra=REPLAY_FIELDS,robot=robot)
    history=env.observe()[:,None].repeat(1,10,1)
    with torch.no_grad():
        for _ in range(32):
            obs=env.observe();history=torch.where((env.ep_step==0)[:,None,None],obs[:,None].expand_as(history),history)
            out=env.step(pi(history,task.command(env.state(),env.context)))
            if not all(out[k].isfinite().all() for k in ('s','s1','a','r')):raise FloatingPointError('Nonfinite smoke transition')
            replay.add(out);history=torch.cat((history[:,1:],env.observe()[:,None]),1)
    loss=wm_loss(wm,task,replay.sample_windows(cfg['wm_batch'],4));ow.zero_grad();loss.backward();rt.step(ow,wm,'wm',max_norm=1.)
    stats=update_actor(pi,wm,critic,task,replay,op,ov,cfg,train_actor=False,runtime=rt)
    stats=update_actor(pi,wm,critic,task,replay,op,ov,cfg,train_actor=True,runtime=rt)
    digest=rt.check_sync({});finite=all(x.isfinite().all() for model in (pi,wm,critic) for x in model.state_dict().values())
    if sha256(args.checkpoint)!=source_hash:raise ValueError('Source checkpoint changed')
    if rt.main:
        report=dict(passed=bool(finite and not rt.skipped),updates=rt.updates,skipped=rt.skipped,state_sha256=digest,
            world_size=rt.world_size,physics_substeps=env.physics_substeps,wm_loss=float(loss.detach()),metrics=stats,
            scope=__doc__,formal_training=False,weights_saved=False,source_checkpoint_sha256=source_hash)
        json_save(args.out,report);print(report,flush=True)
    rt.close()


if __name__=='__main__':main()
